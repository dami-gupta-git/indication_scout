"""Supervisor agent.

Top-level agent that orchestrates the literature, clinical trials, and mechanism sub-agents. Given
a drug, it surfaces candidate diseases, decides which to investigate, and delegates to the right
sub-agent for each candidate.

The LLM decides:
- Which candidates are worth investigating in depth
- Whether to run literature, clinical trials, or both for each
- When enough evidence has been gathered to stop
"""

import logging
from datetime import date
from pathlib import Path
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from indication_scout.agents.mechanism.mechanism_output import MechanismOutput
from indication_scout.agents.supervisor.supervisor_output import (
    CandidateBlurb,
    CandidateFindings,
    SupervisorOutput,
)
from indication_scout.agents.supervisor.supervisor_tools import build_supervisor_tools
from indication_scout.config import get_settings

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"


def _load_system_prompt(holdout_mode: bool) -> str:
    """Return the supervisor system prompt for production or holdout mode."""
    name = "supervisor_holdout.txt" if holdout_mode else "supervisor.txt"
    return (_PROMPTS_DIR / name).read_text()


# Production prompt loaded at import time. Importers (e.g. the probe script
# in scripts/probe_supervisor_t2dm.py) reference this binding directly.
SYSTEM_PROMPT = _load_system_prompt(holdout_mode=False)


def build_supervisor_agent(llm, svc, db, date_before: date | None = None):
    """Return (compiled supervisor agent, get_merged_allowlist, get_auto_findings).

    - get_merged_allowlist: snapshots the closure-scoped competitor + mechanism
      disease allowlist after the agent has finished running.
    - get_auto_findings: snapshots artifacts produced by the holdout-only
      investigate_top_candidates tool. Empty in non-holdout runs. Used by
      run_supervisor_agent to merge into findings_by_disease since those tool
      calls bypass the LangGraph ReAct loop.

    `date_before` is forwarded to the literature and clinical trials sub-agents so PubMed and
    ClinicalTrials.gov queries respect the same temporal cutoff. When `date_before` is set,
    the supervisor loads a holdout-specific system prompt that tells the LLM to treat all
    candidates as open hypotheses (including ones that would look "obvious" today) and not
    skip them based on training knowledge of the drug's eventual primary use.
    """
    tools, get_merged_allowlist, get_auto_findings = build_supervisor_tools(
        llm=llm, svc=svc, db=db, date_before=date_before
    )
    prompt_file = (
        "supervisor_holdout.txt" if date_before is not None else "supervisor.txt"
    )
    logger.info("supervisor prompt: %s (date_before=%s)", prompt_file, date_before)
    prompt = _load_system_prompt(holdout_mode=date_before is not None)
    agent = create_react_agent(model=llm, tools=tools, prompt=prompt)
    return agent, get_merged_allowlist, get_auto_findings


async def run_supervisor_agent(
    agent,
    get_merged_allowlist,
    drug_name: str,
    get_auto_findings=None,
) -> SupervisorOutput:
    """Invoke the supervisor and assemble a SupervisorOutput from the run.

    Filters out tool calls that were rejected by the candidate guard, and canonicalises disease
    names against the merged competitor + mechanism allowlist so that casing variants (e.g.
    "Parkinson disease" vs "parkinson disease") do not produce duplicate findings, and so that
    mechanism-promoted diseases reach the findings list with their correct source tag.

    `get_auto_findings` (holdout-only): a zero-arg callable returning artifacts produced by
    investigate_top_candidates. Those tool calls bypass the LangGraph ReAct loop, so their
    artifacts don't reach result["messages"]. We pull them via the closure and merge into
    findings_by_disease so the report renderer sees them like any other investigation.
    None in non-holdout runs (the tool isn't built and no closure to read from).
    """
    result = await agent.ainvoke(
        {
            "messages": [
                HumanMessage(content=f"Find repurposing opportunities for {drug_name}")
            ]
        }
    )

    mechanism: MechanismOutput | None = None
    summary: str = ""
    blurbs: list[dict] = []
    findings_by_disease: dict[str, CandidateFindings] = {}

    # Build a map from tool_call_id → args so we can recover the disease_name argument passed to
    # each analyze_* call.
    tool_call_args: dict[str, dict] = {}
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            for tc in msg.tool_calls:
                tool_call_args[tc["id"]] = tc["args"]

    # First pass: capture mechanism artifact and the supervisor's final summary.
    # finalize_supervisor's artifact is {"summary": str, "blurbs": list[dict]}.
    for msg in result["messages"]:
        if not isinstance(msg, ToolMessage):
            continue
        if msg.name == "analyze_mechanism":
            mechanism = msg.artifact
        elif msg.name == "finalize_supervisor":
            artifact = msg.artifact or {}
            summary = artifact.get("summary", "") or ""
            blurbs = artifact.get("blurbs", []) or []

    # Single source of truth: the merged allowlist the runtime tools enforced. Keyed by
    # lowercase disease name → (canonical_name, source). Source is "competitor", "mechanism",
    # or "both".
    allowed_lower = get_merged_allowlist()

    def _canonical(
        disease_raw: str,
    ) -> tuple[str, Literal["competitor", "mechanism", "both"]] | None:
        """Return (canonical_name, source) for disease_raw, or None if not allowed."""
        key = disease_raw.lower().strip()
        return allowed_lower.get(key)

    # Second pass: assemble findings, skipping rejected calls and canonicalising keys.
    for msg in result["messages"]:
        if not isinstance(msg, ToolMessage):
            continue
        if msg.name not in {"analyze_literature", "analyze_clinical_trials"}:
            continue

        args = tool_call_args.get(msg.tool_call_id, {})
        disease_raw = args.get("disease_name", "")
        match = _canonical(disease_raw)

        if match is None:
            logger.warning(
                "Skipping rejected %s call for disease=%r (not in allowlist)",
                msg.name,
                disease_raw,
            )
            continue

        canonical, source = match
        findings = findings_by_disease.setdefault(
            canonical, CandidateFindings(disease=canonical, source=source)
        )

        if msg.name == "analyze_literature":
            findings.literature = msg.artifact
        else:  # analyze_clinical_trials
            findings.clinical_trials = msg.artifact

    # Holdout merge: investigate_top_candidates invokes analyze_literature /
    # analyze_clinical_trials directly (not through the ReAct loop), so their
    # artifacts don't reach result["messages"]. Pull them from the closure
    # and merge into findings_by_disease. LLM-driven calls take precedence
    # because they may have been re-runs with refined disease names; we only
    # fill in slots the LLM didn't already populate.
    if get_auto_findings is not None:
        auto = get_auto_findings()
        for disease_lower, artifacts in auto.items():
            match = _canonical(disease_lower)
            if match is None:
                continue
            canonical, source = match
            findings = findings_by_disease.setdefault(
                canonical, CandidateFindings(disease=canonical, source=source)
            )
            if findings.literature is None and artifacts.get("literature") is not None:
                findings.literature = artifacts["literature"]
            if (
                findings.clinical_trials is None
                and artifacts.get("clinical_trials") is not None
            ):
                findings.clinical_trials = artifacts["clinical_trials"]

    # Attach supervisor-written blurbs to the matching CandidateFindings. Blurbs only attach to
    # diseases the supervisor actually investigated this run (i.e. already present in
    # findings_by_disease). Names are canonicalised through the merged allowlist so casing /
    # synonym variants land on the right finding. Holdout runs send an empty blurbs list.
    structured_keys = (
        "stage",
        "literature",
        "blocker",
        "active_programs",
        "key_risk",
        "verdict",
        "watch",
    )
    for entry in blurbs:
        disease_raw = entry.get("disease", "")
        if not disease_raw:
            continue
        prose = (entry.get("prose") or "").strip()
        fields = {k: (entry.get(k) or "").strip() for k in structured_keys}
        if not prose and not any(fields.values()):
            continue
        match = _canonical(disease_raw)
        if match is None:
            logger.warning(
                "Skipping blurb for disease=%r (not in allowlist)", disease_raw
            )
            continue
        canonical, _ = match
        finding = findings_by_disease.get(canonical)
        if finding is None:
            logger.warning(
                "Skipping blurb for disease=%r (not investigated this run)",
                disease_raw,
            )
            continue
        finding.blurb = CandidateBlurb(prose=prose, **fields)

    # Candidates surfaced to downstream consumers = every disease in the merged allowlist,
    # mapped back to its canonical name. Includes mechanism-promoted diseases.
    candidate_diseases = [canonical for (canonical, _) in allowed_lower.values()]

    # Build top_diseases from the supervisor's blurb list (rank-ordered). Drop any blurb
    # disease that isn't in the allowlist or wasn't investigated this run, then hard cap
    # at 5. Enforces top_diseases ⊆ disease_findings ⊆ candidate_diseases.
    top_diseases: list[str] = []
    seen_top: set[str] = set()
    for entry in blurbs:
        disease_raw = entry.get("disease", "")
        if not disease_raw:
            continue
        match = _canonical(disease_raw)
        if match is None:
            continue
        canonical, _ = match
        if canonical not in findings_by_disease:
            logger.warning(
                "Skipping top_diseases entry for disease=%r (not investigated this run)",
                disease_raw,
            )
            continue
        if canonical in seen_top:
            continue
        seen_top.add(canonical)
        top_diseases.append(canonical)
    top_diseases = top_diseases[: get_settings().supervisor_candidate_cap]

    # Reorder disease_findings so top_diseases entries appear first in rank order, with
    # any remaining investigated diseases following in insertion order.
    top_set = set(top_diseases)
    disease_findings = [findings_by_disease[name] for name in top_diseases]
    for canonical, finding in findings_by_disease.items():
        if canonical not in top_set:
            disease_findings.append(finding)

    return SupervisorOutput(
        drug_name=drug_name,
        candidate_diseases=candidate_diseases,
        mechanism=mechanism,
        disease_findings=disease_findings,
        top_diseases=top_diseases,
        summary=summary,
    )
