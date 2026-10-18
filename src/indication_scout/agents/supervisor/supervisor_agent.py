"""Supervisor agent: orchestrates the literature, clinical trials, and mechanism sub-agents for a drug."""

import logging
import time
from collections.abc import Callable, Mapping
from datetime import date
from pathlib import Path
from typing import Any, Literal, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from sqlalchemy.orm import Session, sessionmaker

from indication_scout.agents._react_loop import (
    _trailing_tool_messages,
    build_gated_react_loop,
)
from indication_scout.agents.mechanism.mechanism_output import MechanismOutput
from indication_scout.agents.supervisor.supervisor_output import (
    CandidateBlurb,
    CandidateFindings,
    SupervisorOutput,
)
from indication_scout.agents.supervisor.supervisor_tools import build_supervisor_tools
from indication_scout.config import get_settings
from indication_scout.services.progress import PHASE_SUMMARY, emit_progress
from indication_scout.services.retrieval import RetrievalService

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"


# Appended when supervisor_fanout is on; replaces WORKFLOW step 2 with one parallel fan-out call.
_FANOUT_DIRECTIVE = """

# FAN-OUT MODE (overrides ONLY WORKFLOW step 2)
For step 2 ONLY: do NOT call analyze_literature / analyze_clinical_trials per candidate. Instead,
after find_candidates and analyze_mechanism complete, call investigate_top_candidates ONCE — it
runs literature and clinical trials for every candidate in parallel. Rank from the results it
returns. All other WORKFLOW steps are UNCHANGED: still call get_drug_briefing (step 3), then
critique_ranking (step 4, MANDATORY — finalize_supervisor is rejected until critique_ranking has
run this turn), then finalize_supervisor (step 5).
"""


def _load_system_prompt(fanout: bool = False) -> str:
    """Return the supervisor system prompt, with the fan-out directive appended when `fanout` is set."""
    summary_count = get_settings().supervisor_investigation_cap
    prompt = (
        (_PROMPTS_DIR / "supervisor.txt")
        .read_text()
        .format(summary_count=summary_count)
    )
    if fanout:
        prompt = prompt + _FANOUT_DIRECTIVE
    return prompt


# Loaded at import time; some scripts reference this directly.
SYSTEM_PROMPT = _load_system_prompt()


def _finalize_done(messages: list) -> bool:
    """End the loop once finalize_supervisor succeeds, which requires critique_ranking to have run first."""
    critique_ran = any(
        isinstance(m, ToolMessage) and m.name == "critique_ranking" for m in messages
    )
    for m in _trailing_tool_messages(messages):
        if m.name == "finalize_supervisor" and m.artifact and critique_ran:
            return True
    return False


def build_supervisor_agent(
    llm: BaseChatModel,
    svc: RetrievalService,
    db: Session,
    session_factory: sessionmaker | None = None,
    date_before: date | None = None,
) -> tuple[
    Any,
    Callable[[], dict[str, tuple[str, Literal["competitor", "mechanism", "both"]]]],
    Callable[[], dict[str, dict]],
    Callable[[], dict[str, str]],
]:
    """Return (agent, get_merged_allowlist, get_auto_findings, get_approval_labels).

    - get_merged_allowlist: snapshots the competitor + mechanism disease allowlist after the run.
    - get_auto_findings: snapshots fan-out artifacts (empty when fan-out is off).
    - get_approval_labels: snapshots FDA approval-relationship labels for kept candidates.

    `date_before` is forwarded to the literature and clinical trials sub-agents as a shared temporal cutoff.
    """
    tools, get_merged_allowlist, get_auto_findings, get_approval_labels = (
        build_supervisor_tools(
            llm=llm,
            svc=svc,
            db=db,
            session_factory=session_factory,
            date_before=date_before,
        )
    )
    fanout = get_settings().supervisor_fanout
    logger.info(
        "supervisor prompt: supervisor.txt (date_before=%s, fanout=%s)",
        date_before,
        fanout,
    )
    prompt = _load_system_prompt(fanout=fanout)
    agent = build_gated_react_loop(llm, tools, prompt, _finalize_done)
    return agent, get_merged_allowlist, get_auto_findings, get_approval_labels


async def run_supervisor_agent(
    agent: Any,
    get_merged_allowlist: Callable[
        [], dict[str, tuple[str, Literal["competitor", "mechanism", "both"]]]
    ],
    drug_name: str,
    get_auto_findings: Callable[[], dict[str, dict]] | None = None,
    get_approval_labels: Callable[[], dict[str, str]] | None = None,
    date_before: date | None = None,
) -> SupervisorOutput:
    """Invoke the supervisor and assemble a SupervisorOutput from the run.

    Drops tool calls rejected by the candidate guard and canonicalises disease names against the merged allowlist.
    `get_auto_findings` (fan-out only) supplies artifacts that bypass the ReAct loop; None when fan-out is off.
    """
    _agent_t0 = time.perf_counter()
    result = await agent.ainvoke(
        {
            "messages": [
                HumanMessage(content=f"Find repurposing opportunities for {drug_name}")
            ]
        }
    )
    _agent_elapsed = time.perf_counter() - _agent_t0
    emit_progress(PHASE_SUMMARY, "Ranking candidates and writing the summary")

    # Per-turn LLM accounting; read-only.
    _ai_turns = [m for m in result["messages"] if isinstance(m, AIMessage)]
    _total_out = 0
    for _i, _msg in enumerate(_ai_turns):
        _usage: Mapping[str, Any] = _msg.usage_metadata or {}
        _in_tok = _usage.get("input_tokens", 0)
        _out_tok = _usage.get("output_tokens", 0)
        # cache_read==0 across turns 2+ means prompt caching is silently missing. Fresh writes are reported
        # under the TTL-specific ephemeral keys.
        _details = _usage.get("input_token_details", {})
        _cache_read = _details.get("cache_read", 0)
        _cache_write = (
            _details.get("ephemeral_5m_input_tokens", 0)
            + _details.get("ephemeral_1h_input_tokens", 0)
        ) or _details.get("cache_creation", 0)
        _total_out += _out_tok
        _called = ", ".join(tc["name"] for tc in _msg.tool_calls) or "(final)"
        logger.debug(
            "[LLMTURN] supervisor turn %d/%d: in=%d out=%d cache_read=%d "
            "cache_write=%d -> %s",
            _i + 1,
            len(_ai_turns),
            _in_tok,
            _out_tok,
            _cache_read,
            _cache_write,
            _called,
        )
    logger.debug(
        "[LLMTURN] supervisor: %d turns, %d total output tokens, agent loop %.1fs",
        len(_ai_turns),
        _total_out,
        _agent_elapsed,
    )

    mechanism: MechanismOutput | None = None
    summary: str = ""
    blurbs: list[dict] = []
    findings_by_disease: dict[str, CandidateFindings] = {}

    # tool_call_id → args, to recover each analyze_* call's disease_name.
    tool_call_args: dict[str, dict] = {}
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            for tc in msg.tool_calls:
                tool_call_args[cast(str, tc["id"])] = tc["args"]

    # First pass: capture the mechanism artifact and the final summary/blurbs.
    for msg in result["messages"]:
        if not isinstance(msg, ToolMessage):
            continue
        if msg.name == "analyze_mechanism":
            mechanism = msg.artifact
        elif msg.name == "finalize_supervisor":
            artifact = msg.artifact or {}
            summary = artifact.get("summary", "") or ""
            blurbs = artifact.get("blurbs", []) or []

    # Merged allowlist enforced by the runtime tools: lowercase name → (canonical_name, source).
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
        # None artifact = sub-agent errored; skip so an uninvestigated disease gets no empty finding.
        if msg.artifact is None:
            logger.warning(
                "Skipping %s for disease=%r (None artifact — sub-agent errored)",
                msg.name,
                disease_raw,
            )
            continue

        findings = findings_by_disease.setdefault(
            canonical, CandidateFindings(disease=canonical, source=source)
        )

        if msg.name == "analyze_literature":
            findings.literature = msg.artifact
        else:  # analyze_clinical_trials
            findings.clinical_trials = msg.artifact

    # Fan-out merge: LLM-driven calls take precedence; only fill slots the LLM didn't populate.
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

    # Set approval relationship from the upstream FDA check, matching on canonical key or the finding's disease.
    approval_labels = get_approval_labels() if get_approval_labels is not None else {}
    finding: CandidateFindings | None
    if approval_labels:
        for canonical, finding in findings_by_disease.items():
            label = approval_labels.get(
                canonical.lower().strip()
            ) or approval_labels.get(finding.disease.lower().strip())
            if label in ("contaminated", "combination_only"):
                finding.approval_relationship = cast(
                    Literal["contaminated", "combination_only"], label
                )

    # Attach blurbs to the matching findings (investigated diseases only).
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

    # Every allowlist disease by canonical name, including mechanism-promoted.
    candidate_diseases = [canonical for (canonical, _) in allowed_lower.values()]

    # top_diseases from rank-ordered blurbs, dropping uninvestigated ones, then capped.
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

    # top_diseases first in rank order, then the rest.
    top_set = set(top_diseases)
    disease_findings = [findings_by_disease[name] for name in top_diseases]
    for canonical, finding in findings_by_disease.items():
        if canonical not in top_set:
            disease_findings.append(finding)

    # Collapse the drug-wide safety signal to one report-level blurb: first non-empty summary,
    # order-preserving union of PMIDs.
    drug_safety_summary = ""
    drug_regulatory_safety_summary = ""
    drug_regulatory_safety_full_labels = ""
    drug_pharmacovigilance_summary = ""
    drug_literature_safety_summary = ""
    drug_safety_pmids: list[str] = []
    for finding in disease_findings:
        es = finding.literature.evidence_summary if finding.literature else None
        if es is None:
            continue
        if not drug_safety_summary and es.safety_summary:
            drug_safety_summary = es.safety_summary
        if not drug_regulatory_safety_summary and es.regulatory_safety_summary:
            drug_regulatory_safety_summary = es.regulatory_safety_summary
        if not drug_regulatory_safety_full_labels and es.regulatory_safety_full_labels:
            drug_regulatory_safety_full_labels = es.regulatory_safety_full_labels
        if not drug_pharmacovigilance_summary and es.pharmacovigilance_summary:
            drug_pharmacovigilance_summary = es.pharmacovigilance_summary
        if not drug_literature_safety_summary and es.literature_safety_summary:
            drug_literature_safety_summary = es.literature_safety_summary
        for pmid in es.safety_pmids:
            if pmid not in drug_safety_pmids:
                drug_safety_pmids.append(pmid)

    return SupervisorOutput(
        drug_name=drug_name,
        candidate_diseases=candidate_diseases,
        mechanism=mechanism,
        disease_findings=disease_findings,
        top_diseases=top_diseases,
        summary=summary,
        date_before=date_before,
        drug_safety_summary=drug_safety_summary,
        drug_regulatory_safety_summary=drug_regulatory_safety_summary,
        drug_regulatory_safety_full_labels=drug_regulatory_safety_full_labels,
        drug_pharmacovigilance_summary=drug_pharmacovigilance_summary,
        drug_literature_safety_summary=drug_literature_safety_summary,
        drug_safety_pmids=drug_safety_pmids,
    )
