"""Clinical Trials agent

Uses LangGraph's prebuilt create_react_agent for the agent loop. After
the run, walks the message history to pull typed artifacts off the
ToolMessages and assembles them into a ClinicalTrialsOutput.
"""

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from indication_scout.agents.clinical_trials.clinical_trials_output import (
    ClinicalTrialsOutput,
)
from indication_scout.agents.clinical_trials.clinical_trials_tools import (
    build_clinical_trials_tools,
)

SYSTEM_PROMPT = """\
You are a clinical trials analyst assessing whether a drug could be
repurposed for a new indication.

You have five tools:

- detect_whitespace — checks if any trials exist for this drug-indication pair
- search_trials — fetches details on trials matching the drug and indication
- get_landscape — competitive landscape: total trials, top sponsors, phase distribution
- get_terminated — terminated trials split into two groups: drug-wide safety/efficacy
  failures (same count for every indication — reflects the drug's overall failure
  history) and indication-specific terminations (trials that failed in this disease
  space specifically). Report these separately — do not sum them into a single count.
- finalize_analysis — signals completion; you MUST call this last

Decide which tools to call based on what you learn. Typically start with
detect_whitespace. If trials exist, get the details and landscape. If not,
check for terminated trials and the broader landscape. If terminated trials
reveal safety or efficacy failures, that's enough — you can skip landscape.

Batch independent tool calls into a single response when possible.

IMPORTANT: finalize_analysis MUST be the final tool call. Pass it your
2-3 sentence plain-text summary of the findings. Do NOT
emit a plain text message after calling finalize_analysis.

COMPETITIVE INTENSITY RULES — apply these when summarising the landscape:
- Any Phase 3 trial with >500 enrolled participants from a named pharma/biotech
  sponsor signals a HIGH-INVESTMENT, CROWDED space — do not describe it as early
  or uncrowded regardless of total trial count.
- A completed Phase 3 trial means the space has reached late-stage validation;
  flag this explicitly.
- "Few trials" (e.g. 6) does not mean "uncrowded" if those trials are large Phase 3
  programmes. Distinguish trial count from investment scale.
- When reporting competitive intensity, cite the highest-phase trials, their
  enrollment, and their status — not just the total count.

GROUNDING RULE: Your summary must reference ONLY information returned
by the tools in this run. Do not introduce trial names, drug histories,
or facts from your training that were not returned by the tools."""

def build_clinical_trials_agent(llm, date_before=None, max_search_results=None):
    """Return a compiled ReAct agent. No graph wiring required."""
    tools = build_clinical_trials_tools(
        date_before=date_before,
        max_search_results=max_search_results,
    )
    return create_react_agent(model=llm, tools=tools, prompt=SYSTEM_PROMPT)


async def run_clinical_trials_agent(
    agent, drug_name: str, disease_name: str
) -> ClinicalTrialsOutput:
    """Invoke the agent and assemble a ClinicalTrialsOutput from the run."""
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=f"Analyze {drug_name} in {disease_name}")]}
    )

    artifacts: dict = {
        "whitespace": None,
        "landscape": None,
        "trials": [],
        "terminated": [],
        "summary": None,
    }

    field_map = {
        "detect_whitespace": "whitespace",
        "search_trials": "trials",
        "get_landscape": "landscape",
        "get_terminated": "terminated",
        "finalize_analysis": "summary",
    }

    for msg in result["messages"]:
        if isinstance(msg, ToolMessage) and msg.name in field_map:
            artifacts[field_map[msg.name]] = msg.artifact

    summary = artifacts.get("summary") or ""

    return ClinicalTrialsOutput(
        whitespace=artifacts["whitespace"],
        landscape=artifacts["landscape"],
        trials=artifacts["trials"],
        terminated=artifacts["terminated"],
        summary=summary,
    )
