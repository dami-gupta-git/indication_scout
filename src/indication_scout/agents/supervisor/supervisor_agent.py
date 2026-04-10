"""Supervisor agent.

Top-level agent that orchestrates the literature, clinical trials, and mechanism
sub-agents. Given a drug, it surfaces candidate diseases, decides which
to investigate, and delegates to the right sub-agent for each candidate.

The LLM decides:
- Which candidates are worth investigating in depth
- Whether to run literature, clinical trials, or both for each
- When enough evidence has been gathered to stop
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from indication_scout.agents.mechanism.mechanism_output import MechanismOutput
from indication_scout.agents.supervisor.supervisor_output import (
    CandidateFindings,
    SupervisorOutput,
)
from indication_scout.agents.supervisor.supervisor_tools import build_supervisor_tools

SYSTEM_PROMPT = """\
You are a drug repurposing analyst. Given a drug, your job is to find
the most promising new indications and assess them.

You have five tools:

- find_candidates — surfaces candidate diseases for the drug from Open Targets
- analyze_mechanism — fetches the drug's molecular targets, their disease
  associations, and Reactome pathways; drug-level, call once per drug
- analyze_literature — runs a full literature analysis for a drug-disease pair
- analyze_clinical_trials — runs a full clinical trials analysis for a pair
- finalize_supervisor — signals completion; you MUST call this last

Strategy:
1. Call find_candidates and analyze_mechanism in parallel as your first step.
2. Use find_candidates to see which diseases competitor drugs are active in.
   Use analyze_mechanism as background context — the target-disease associations
   and pathways tell you about the biology, not about what to investigate next.
3. Pick 3-5 candidates to investigate in depth with literature and/or trials.
   Use your judgment:
   - Literature first when you want to assess biological/clinical evidence
   - Clinical trials first when you want to know if the space is already crowded
   - Both for candidates that look genuinely promising
4. You can investigate multiple candidates in parallel by batching tool calls.
5. Stop when you have enough evidence to identify the top candidates.

GROUNDING RULE: Your final summary must reference ONLY findings
returned by the sub-agent tools in this run. You may use your domain
knowledge to decide which candidates to investigate, but the summary
itself must not introduce trial names, mechanisms, or facts that did
not come back from the tools.

IMPORTANT: finalize_supervisor MUST be the final tool call. Pass it your
4-6 sentence plain-text summary of the most promising candidates,
referencing relevant mechanistic context from analyze_mechanism where it
supports or contextualises the findings. Do NOT emit a plain
text message after calling finalize_supervisor."""


def build_supervisor_agent(llm, svc, db):
    """Return a compiled supervisor agent."""
    tools = build_supervisor_tools(llm=llm, svc=svc, db=db)
    return create_react_agent(model=llm, tools=tools, prompt=SYSTEM_PROMPT)


async def run_supervisor_agent(agent, drug_name: str) -> SupervisorOutput:
    """Invoke the supervisor and assemble a SupervisorOutput from the run."""
    result = await agent.ainvoke(
        {
            "messages": [
                HumanMessage(content=f"Find repurposing opportunities for {drug_name}")
            ]
        }
    )

    candidates: list[str] = []
    mechanism: MechanismOutput | None = None
    summary: str = ""
    # Map disease → CandidateFindings (build incrementally as tool results come in)
    findings_by_disease: dict[str, CandidateFindings] = {}

    # Build a map from tool_call_id → tool_call args so we can recover the
    # disease_name argument that was passed to each analyze_* call
    tool_call_args: dict[str, dict] = {}
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            for tc in msg.tool_calls:
                tool_call_args[tc["id"]] = tc["args"]

    for msg in result["messages"]:
        if not isinstance(msg, ToolMessage):
            continue

        if msg.name == "find_candidates":
            candidates = msg.artifact

        elif msg.name == "analyze_mechanism":
            mechanism = msg.artifact

        elif msg.name == "analyze_literature":
            args = tool_call_args.get(msg.tool_call_id, {})
            disease = args.get("disease_name", "")
            if disease:
                findings = findings_by_disease.setdefault(
                    disease, CandidateFindings(disease=disease)
                )
                findings.literature = msg.artifact

        elif msg.name == "analyze_clinical_trials":
            args = tool_call_args.get(msg.tool_call_id, {})
            disease = args.get("disease_name", "")
            if disease:
                findings = findings_by_disease.setdefault(
                    disease, CandidateFindings(disease=disease)
                )
                findings.clinical_trials = msg.artifact

        elif msg.name == "finalize_supervisor":
            summary = msg.artifact or ""

    return SupervisorOutput(
        drug_name=drug_name,
        candidates=candidates,
        mechanism=mechanism,
        findings=list(findings_by_disease.values()),
        summary=summary,
    )
