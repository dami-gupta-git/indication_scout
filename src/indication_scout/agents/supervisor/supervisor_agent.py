"""Supervisor agent.

Orchestrates the literature, clinical trials, and mechanism sub-agents. Given a drug, the LLM
surfaces candidate diseases, picks which to investigate in depth, chooses literature/trials/both
for each, and decides when enough evidence has been gathered to stop.
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


# Production prompt loaded at import time. Importers (e.g. scripts/probe_supervisor_t2dm.py)
# reference this binding directly.
SYSTEM_PROMPT = _load_system_prompt(holdout_mode=False)


def build_supervisor_agent(llm, svc, db, date_before: date | None = None):
    """Return (compiled supervisor agent, get_merged_allowlist, get_auto_findings).

    - get_merged_allowlist: snapshots the competitor + mechanism disease allowlist after the run.
    - get_auto_findings: snapshots artifacts from the holdout-only investigate_top_candidates tool
      (empty in non-holdout runs). run_supervisor_agent merges these into findings_by_disease
      because those tool calls bypass the LangGraph ReAct loop.

    `date_before` is forwarded to the literature and clinical trials sub-agents so PubMed and
    ClinicalTrials.gov queries share the same temporal cutoff. When set, the supervisor loads a
    holdout-specific prompt telling the LLM to treat all candidates as open hypotheses (even
    "obvious" ones) rather than skipping based on training knowledge of the drug's eventual use.
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

    Filters out tool calls rejected by the candidate guard, and canonicalises disease names against
    the merged allowlist so casing variants (e.g. "Parkinson disease" vs "parkinson disease") don't
    produce duplicate findings and mechanism-promoted diseases land with their correct source tag.

    `get_auto_findings` (holdout-only): zero-arg callable returning investigate_top_candidates
    artifacts. Those tool calls bypass the ReAct loop, so their artifacts don't reach
    result["messages"]; we pull them via the closure and merge into findings_by_disease so the
    renderer sees them like any other investigation. None in non-holdout runs.
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

    # tool_call_id → args, to recover the disease_name passed to each analyze_* call.
    tool_call_args: dict[str, dict] = {}
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            for tc in msg.tool_calls:
                tool_call_args[tc["id"]] = tc["args"]

    # First pass: capture mechanism artifact and the supervisor's final summary.
    # finalize_supervisor's artifact: {"summary": str, "blurbs": list[dict]}.
    for msg in result["messages"]:
        if not isinstance(msg, ToolMessage):
            continue
        if msg.name == "analyze_mechanism":
            mechanism = msg.artifact
        elif msg.name == "finalize_supervisor":
            artifact = msg.artifact or {}
            summary = artifact.get("summary", "") or ""
            blurbs = artifact.get("blurbs", []) or []

    # Single source of truth: the merged allowlist the runtime tools enforced. Keyed by lowercase
    # disease name → (canonical_name, source), source ∈ {competitor, mechanism, both}.
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

    # Holdout merge: investigate_top_candidates calls the analyze_* tools directly (not via the
    # ReAct loop), so their artifacts don't reach result["messages"]. Pull them from the closure
    # and merge in. LLM-driven calls take precedence (they may be re-runs with refined names); we
    # only fill slots the LLM didn't already populate.
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
    # diseases investigated this run (already in findings_by_disease). Names are canonicalised
    # through the allowlist so casing / synonym variants land right. Holdout runs send no blurbs.
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

    # Candidates surfaced downstream = every allowlist disease by canonical name (incl.
    # mechanism-promoted).
    candidate_diseases = [canonical for (canonical, _) in allowed_lower.values()]

    # Build top_diseases from the rank-ordered blurb list. Drop any disease not in the allowlist or
    # not investigated this run, then hard cap. Enforces top_diseases ⊆ disease_findings ⊆
    # candidate_diseases.
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

    # Reorder disease_findings: top_diseases first in rank order, then the rest in insertion order.
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
