"""Supervisor tools: wrap the literature and clinical trials sub-agents as tools (typed output as artifact, short
summary string for the LLM), plus find_candidates, which hits Open Targets directly to surface candidate diseases.
"""

import asyncio
import json
import logging
import random
import re
import time
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any, Literal, TypedDict, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from sqlalchemy.orm import Session, sessionmaker

from indication_scout.agents._trial_formatting import (
    _borda_rank_by_enrollment_and_recency,
    _format_trial_table,
    _phase_distribution,
)
from indication_scout.agents._trial_signals import (
    format_derived_signals,
)
from indication_scout.agents.clinical_trials.clinical_trials_agent import (
    build_clinical_trials_agent,
    run_clinical_trials_agent,
)
from indication_scout.agents.clinical_trials.clinical_trials_output import (
    ClinicalTrialsOutput,
    TrialRelevanceCoverage,
    TrialSignals,
)
from indication_scout.agents.clinical_trials.clinical_trials_tools import (
    _classify_stop_reason,
)
from indication_scout.agents.literature.literature_agent import (
    build_literature_agent,
    run_literature_agent,
)
from indication_scout.agents.literature.literature_output import LiteratureOutput
from indication_scout.agents.mechanism.mechanism_agent import (
    build_mechanism_agent,
    run_mechanism_agent,
)
from indication_scout.agents.mechanism.mechanism_output import MechanismOutput
from indication_scout.agents.supervisor.candidate_dedup import (
    collapse_synonym_entries,
    merge_mechanism_entries,
)
from indication_scout.config import get_settings
from indication_scout.constants import (
    SUPERVISOR_CANDIDATE_JITTER_MAX_SECONDS,
    SUPERVISOR_MIN_PMIDS_NO_TRIALS,
)
from indication_scout.data_sources.open_targets import OpenTargetsClient
from indication_scout.helpers.drug_helpers import (
    DrugIntake,
    normalize_drug_name,
    seed_drug_intake,
)
from indication_scout.models.model_evidence_summary import EvidenceSummary
from indication_scout.services.approval_check import (
    get_approved_indications,
    get_fda_approved_disease_mapping,
)
from indication_scout.services.cost_tracking import candidate_cost_scope
from indication_scout.services.dev_stage import DEV_STAGE_PHRASE, dev_stage_phrase
from indication_scout.services.drug_safety import DrugSafetyService
from indication_scout.services.judge_interpretive import judge_interpretive
from indication_scout.services.llm import parse_last_json_object, query_llm
from indication_scout.services.progress import (
    PHASE_CANDIDATES,
    PHASE_MECHANISM,
    PHASE_SUMMARY,
    PHASE_TRIALS,
    emit_progress,
)
from indication_scout.services.retrieval import RetrievalService

logger = logging.getLogger(__name__)

_EmptyOutputT = TypeVar("_EmptyOutputT")


class _CritiqueState(TypedDict, total=False):
    """Closure-scoped critique_ranking state: whether it ran, and the blurbs it returned."""

    ran: bool
    last_blurbs: list[dict]


# Development-stage phrase + renderer live in services/dev_stage (shared with the report formatter); aliased locally.
_DEV_STAGE_PHRASE = DEV_STAGE_PHRASE
_dev_stage_phrase = dev_stage_phrase


async def _get_shared_drug_intake(
    tasks: dict[str, asyncio.Task[DrugIntake]],
    drug_name: str,
    cache_dir: Path,
    date_before: date | None,
) -> DrugIntake:
    """Return one run-scoped drug-intake result per normalized drug name."""
    normalized = normalize_drug_name(drug_name)
    task = tasks.get(normalized)
    if task is None:
        task = asyncio.create_task(
            seed_drug_intake(normalized, cache_dir, date_before=date_before)
        )
        tasks[normalized] = task
    return await task


# Terse safety flag for ranking blurbs / critic facts, driven by the disease-specific `indication_harm` (harm reported
# for THIS drug in THIS candidate indication), not the drug-level severity. A historical drug-level withdrawal
# (thalidomide, withdrawn 1961 but re-approved) was mis-attributed by the ranking LLM as foreclosing the disease.
_INDICATION_HARM_FLAG = "⚠️ safety signal reported for this indication"
_UNRESOLVED_TRIAL_STAGE = (
    "Trial search unavailable — disease name could not be resolved to a MeSH descriptor"
)
_UNRESOLVED_ACTIVE_PROGRAMS = "Unknown — registry search was not run"
# Summary footer accounting: every disease that reached findings appears in the ranked block or exactly one footer line.
_NOT_RANKED_REASON_UNRESOLVED = "trial search coverage unavailable (disease could not be resolved to a MeSH descriptor)"
_NOT_RANKED_REASON_COMBINATION = "demoted: approved only as a fixed-dose combination"
_NOT_RANKED_REASON_ABSENT = "investigated but absent from the supervisor's ranking"


def _safety_flag(es: EvidenceSummary | None) -> str:
    """Terse safety flag: fires only when a disease-specific harm was reported for this candidate
    indication (EvidenceSummary.indication_harm). "" otherwise (no signal ≠ confirmed safe).
    """
    if es is None or not getattr(es, "indication_harm", False):
        return ""
    return _INDICATION_HARM_FLAG


def _literature_oneliner(es: EvidenceSummary | None) -> str:
    """Deterministic one-line literature summary from the typed EvidenceSummary fields (strength, direction, study
    design), not LLM prose. Feeds judge_interpretive and overwrites the blurb's `literature` field so the design word
    matches the authoritative is_observational token. Returns "None" when no evidence summary is present.
    """
    if es is None:
        return "None"
    # class_level: the relevant RCTs are for OTHER drugs in the class, so it must not read "strong, RCT-backed".
    # synthesize forces strength/direction to "none" for it.
    if getattr(es, "evidence_basis", "none") == "class_level":
        return "class-level signal (no direct evidence for this drug)"
    # approved: the only relevant this-drug evidence studies an APPROVED sub-indication; strength/direction forced to "none".
    if getattr(es, "evidence_basis", "none") == "approved":
        return "evidence is for an already-approved sub-indication (not repurposing)"
    # Animal-only takes precedence: it is the reason the design is otherwise "undetermined"; surface the caveat verbatim.
    if getattr(es, "is_animal_only", None) is True:
        design = "animal/in-vitro only"
    else:
        design = (
            "RCT-backed / controlled"
            if es.is_observational is False
            else (
                "observational"
                if es.is_observational is True
                else "undetermined design"
            )
        )
    direction = es.direction if es.direction and es.direction != "none" else ""
    parts: list[str] = [es.strength or "none"]
    if direction:
        parts.append(direction)
    parts.append(design)
    # Terse severity-aware safety flag only; the full safety_summary is reserved for the rendered report.
    flag = _safety_flag(es)
    if flag:
        parts.append(flag)
    return ", ".join(parts)


# False stage clause the LLM writes in a demotion footer / ranked-summary line when it ignores the authoritative
# dev_stage. Used only there (controlled single-clause contexts); on free-text blurb fields, mid-sentence substitution
# mangled grammar, so the prompt keeps those phase-free. Anchored to whole semicolon/paren-delimited clauses.
_FALSE_STAGE_FRAGMENT = re.compile(
    r"Phase\s*4\s*exploratory\s*only[^;)]*|"
    r"no\s+(?:dedicated|formal)\s+development\s+program[^;)]*|"
    r"exploratory\s+(?:phase\s*4\s+)?only[^;)]*",
    re.IGNORECASE,
)

# dev_stage tiers that assert a real Phase 3 program — only these trigger a footer stage fix.
_PROGRAM_STAGES = {
    "phase3_terminated_for_cause",
    "completed_phase3",
    "active_phase3",
    "phase3_unknown_status",
}


# Fresh-context critic for the ranked blurbs, by reasoning rather than a rule list: (1) sanity-check the ORDER and
# reorder if not defensible; (2) repair blurb fields that contradict the machine-derived per-candidate FACT.
_RANKING_CRITIC_SYSTEM = """\
You are a skeptical reviewer of a drug-repurposing candidate ranking. You are given the candidates \
in their current rank order, each with a verdict tag, a few one-line fields, and a per-candidate \
FACT block (machine-derived, authoritative).

FIRST, audit the ORDER. For each candidate, think about what its evidence actually MEANS for \
whether it is a live repurposing opportunity worth surfacing first — not merely how much or how \
impressive the evidence is. A candidate that has been tested and shown to FAIL, or whose program \
is closed, is a weaker opportunity than one that is still live, however thin. A candidate whose \
only on-record trials were WITHDRAWN before enrolling has no trial evidence at all, and ranks below \
a candidate with any human data. If the current order is not defensible on that basis, reorder the \
candidates so the strongest live opportunities come first. If it is already defensible, keep it.

SECOND, repair any blurb field that CONTRADICTS the candidate's FACT (the FACT is authoritative; \
it states whether a relevant COMPLETED or ACTIVE Phase 3 is on record, or none). Do not let a \
field claim there are no Phase 3 trials when the FACT says one exists. Leave true statements about \
the absence of a commercial/NDA/regulatory PROGRAM alone (a generic drug files no new NDA — that \
is not the same as "no trial"). An unresolved registry query is UNKNOWN, not zero: never claim \
that no trials or active programs exist when the FACT says the query was unresolved. Change nothing \
else.

Output a JSON object ONLY (no prose, no fences):
{"blurbs": [ <every input blurb, in your final rank order, each a full dict with the same keys; \
fields you repaired are rewritten, all others verbatim> ]}
Return every blurb. Preserve every key."""


def _log_disease_banner(title: str, diseases: list[str]) -> None:
    """Emit a boxed WARNING-level banner listing diseases, one per line, to mark candidate-list transitions in run logs."""
    header = f" {title} (n={len(diseases)}) "
    width = max(len(header) + 4, 60)
    bar = "=" * width
    lines = [bar, header.center(width, "="), bar]
    if diseases:
        for d in diseases:
            lines.append(f"  - {d}")
    else:
        lines.append("  (none)")
    lines.append(bar)
    logger.warning("\n%s", "\n".join(lines))


def _evidence_gate_reason(slot: dict) -> str | None:
    """Top-N evidence gate over one findings slot: the exclusion reason, or None when the candidate survives.

    Excludes a candidate with 0 relevant trials AND no usable literature signal (direction "none" OR study_count 0).
    Literature with at least one relevant abstract and 0 trials is a legitimate signal and is kept. The PMID-count
    fallback applies only when synthesize didn't run. The gate keys off direction, not strength, so a contradicting
    body survives as a negative. An unresolved trial query has n_trials None, never 0, so it cannot satisfy the gate.
    """
    lit = slot.get("literature")
    ct = slot.get("clinical_trials")
    n_pmids = len(lit.pmids) if lit else 0
    coverage = ct.search_coverage if ct is not None else None
    n_trials = coverage.relevant_records if coverage is not None else None
    es = lit.evidence_summary if lit and lit.evidence_summary else None
    lit_direction = es.direction if es else None
    lit_study_count = es.study_count if es else None
    no_lit_signal = (
        lit_direction == "none"
        or lit_study_count == 0
        or (
            lit_direction is None
            and lit_study_count is None
            and n_pmids < SUPERVISOR_MIN_PMIDS_NO_TRIALS
        )
    )
    if n_trials == 0 and no_lit_signal:
        return (
            f"0 relevant trials and no usable literature signal "
            f"(direction={lit_direction or 'unavailable'}, "
            f"study_count={lit_study_count if lit_study_count is not None else 'unavailable'})"
        )
    return None


def _trial_query_unresolved(ct: "ClinicalTrialsOutput | None") -> bool:
    """Whether the pair query could not run because disease resolution failed."""
    return bool(
        ct is not None
        and ct.search is not None
        and ct.search.resolution_status == "unresolved"
    )


def _closure_text(
    ct: "ClinicalTrialsOutput | None", literature: EvidenceSummary | None = None
) -> str:
    """Render the overall live/closed verdict as a sentence for the interpretive writer.

    A moderate-or-strong contradictory literature body closes the overall signal even when the registry-only verdict
    is live or unknown, provided the evidence is human clinical data and not graded observational-only.
    `is_animal_only is False` means at least one paper carrying the efficacy verdict studied people; a controlled
    design is not required, since a single-arm Phase 2 showing no activity is legitimate negative evidence (imatinib
    x glioblastoma, ovarian, prostate all read "live" while every trial had failed). A body graded observational-only
    (`is_observational is True`) cannot close.
    """
    if (
        literature is not None
        and literature.evidence_basis == "drug_specific"
        and literature.strength in {"moderate", "strong"}
        and literature.direction == "contradicts"
        and literature.is_animal_only is False
        and literature.is_observational is not True
    ):
        return "CLOSED — human clinical literature contradicts efficacy for this indication"
    if ct is None or ct.closure == "unknown":
        # Must not read "the hypothesis is not established": that was turned into a "Closed signal" for a candidate
        # with no trials.
        return "NOT DECIDED — no closure verdict was reached; do NOT describe the signal as closed"
    reason = ct.closure_reason.strip()
    state = (
        "CLOSED — the signal is not live"
        if ct.closure == "closed"
        else "LIVE — not closed"
    )
    return f"{state}{f' ({reason})' if reason else ''}"


def _terminations_text(sig: TrialSignals | None) -> str:
    """State whether a late-stage trial was terminated for safety or efficacy.

    The development stage's terminated-for-cause tier applies only when no completed Phase 3 exists, so a programme
    that completed one Phase 3 and killed the rest reads as plain "Phase 3 completed". This is passed separately so
    the stage is untouched while the summary can still say it.
    """
    if sig is None:
        return "none reported"
    if getattr(sig, "phase3_terminated_for_cause", False):
        return "a Phase 3-band trial was TERMINATED for a safety or benefit:risk reason"
    return "none reported"


def _trial_evidence_text(coverage: TrialRelevanceCoverage | None) -> str:
    """Describe reviewed relevant trials without promoting raw query matches."""
    if coverage is None:
        return "relevance-reviewed trial count unavailable"
    if coverage.coverage_complete:
        return (
            f"{coverage.relevant_records} relevant trial(s), complete review of "
            f"{coverage.registry_query_matches} registry query match(es)"
        )
    return (
        f"at least {coverage.relevant_records} relevant among "
        f"{coverage.classified_records} reviewed; "
        f"{coverage.registry_query_matches} registry query match(es), "
        f"{coverage.unreviewed_records} not reviewed"
    )


def build_supervisor_tools(
    llm: BaseChatModel,
    svc: RetrievalService,
    db: Session,
    session_factory: "sessionmaker | None" = None,
    date_before: date | None = None,
) -> tuple[
    list,
    Callable[[], dict[str, tuple[str, Literal["competitor", "mechanism", "both"]]]],
    Callable[[], dict[str, dict]],
    Callable[[], dict[str, str]],
]:
    """Build supervisor tools that close over the sub-agents.

    The literature and clinical trials agents are compiled once here and reused across calls.

    `session_factory` is a shared `sessionmaker` used by the concurrent investigate_top_candidates fan-out; each
    analyze_literature call checks out its own Session because a SQLAlchemy Session is not safe for concurrent use.
    When None (single-threaded callers, tests), a factory is derived from the run `db`'s existing engine, never a new one.

    `date_before` is forwarded to the literature and clinical trials sub-agents as a shared temporal cutoff. The
    mechanism sub-agent doesn't accept it (OpenTargets has no date-filtering API).

    Returns (tools, get_merged_allowlist, get_auto_findings, get_approval_labels). get_merged_allowlist() snapshots the
    post-merge disease allowlist (lowercase name → (canonical_name, source)) and is read after the agent loop finishes.
    """
    # Reuse the run db's engine when no shared factory was passed (db.get_bind() creates no new engine/pool).
    if session_factory is None:
        session_factory = sessionmaker(
            autocommit=False, autoflush=False, bind=db.get_bind()
        )

    # The mechanism sub-agent is built once (it runs once in the seed phase). The clinical-trials agent is built PER
    # analyze_clinical_trials call: its tools carry a closure-scoped shown_by_pair set, and investigate_top_candidates
    # fans candidates out concurrently, so a shared instance would mix every pair's trials and race on that state.
    # The build is graph compile only (no I/O), cheap next to the network/LLM work.
    mech_agent = build_mechanism_agent(llm=llm, date_before=date_before)

    # One safety service per run, shared by every per-candidate literature agent. Its in-flight FDA-label task map
    # dedups concurrent label fetches for the same drug; one per candidate would fire identical openFDA requests.
    safety_svc = DrugSafetyService(svc.cache_dir)

    # Closure-scoped allowlist, populated by find_candidates and analyze_mechanism and checked by analyze_literature /
    # analyze_clinical_trials. allowed_diseases: lowercase disease name → (canonical_name, source)
    allowed_diseases: dict[
        str, tuple[str, Literal["competitor", "mechanism", "both"]]
    ] = {}
    # EFO ID → lowercase disease name (key into allowed_diseases), to dedup mechanism candidates against competitor
    # entries by ontology ID even when names differ (e.g. "NSCLC" vs "non-small cell lung cancer").
    allowed_efo_ids: dict[str, str] = {}
    # Raw mechanism candidates buffered by analyze_mechanism; find_candidates consumes them in merge_and_dedup() so the
    # dedup pass sees the full competitor + mechanism union.
    mechanism_candidates_buffer: list = []
    # Drug-level mechanism target list, set by analyze_mechanism and read by merge_and_dedup as MoA context.
    mechanism_targets_for_dedup: list[tuple[str, str]] = []
    # Seed-phase gates. find_candidates and analyze_mechanism run in parallel. analyze_mechanism only buffers raw
    # candidates and sets analyze_mechanism_done. find_candidates seeds the competitor allowlist, awaits that event,
    # then runs merge_and_dedup, which sets find_candidates_done so downstream tools only see the post-dedup allowlist.
    # Both events are set in try/finally so a sub-agent crash can't deadlock downstream tools.
    find_candidates_done = asyncio.Event()
    analyze_mechanism_done = asyncio.Event()

    # Drug-level shared store, populated by sub-agents and surfaced via get_drug_briefing; keyed by normalized drug name.
    drug_facts: dict[str, dict] = {}
    drug_intake_tasks: dict[str, asyncio.Task[DrugIntake]] = {}

    # Fan-out only: investigate_top_candidates calls analyze_* directly, outside the ReAct loop, so their artifacts are
    # stashed here for run_supervisor_agent to read via get_auto_findings(). Keyed by lowercase canonical disease name →
    # {"literature": ..., "clinical_trials": ...}.
    auto_findings: dict[str, dict] = {}

    # FDA approval-relationship labels for kept candidates, keyed by lowercase disease name → "contaminated" |
    # "combination_only" ("approved" is dropped, "none" omitted). run_supervisor_agent reads it via get_approval_labels()
    # so approval_relationship comes from the label, not LLM prose.
    approval_labels: dict[str, str] = {}

    # Per-disease sub-agent artifacts from analyze_literature / analyze_clinical_trials, used by finalize_supervisor's
    # top-N evidence gate. Keyed by lowercase canonical disease name → {"literature": ... | None, "clinical_trials": ... | None}.
    findings_local: dict[str, dict] = {}

    # Ordering gate: finalize_supervisor is rejected until critique_ranking has run. The LLM ignores the prompt-level
    # "MANDATORY" on its own, so this enforces it in code.
    critique_state: _CritiqueState = {"ran": False}

    def _drug_key(drug_name: str) -> str:
        return drug_name.lower().strip()

    def _ensure_drug_entry(drug_name: str) -> dict:
        key = _drug_key(drug_name)
        if key not in drug_facts:
            drug_facts[key] = {
                "drug_name": key,
                "chembl_id": None,
                "drug_profile": None,
                "drug_aliases": [],  # ChEMBL trade/generic names
                "approved_indications": [],  # list of indication strings
                "mechanism_targets": [],  # list of (gene, action_type)
                "mechanism_disease_associations": [],  # list of (gene, disease, score)
                "first_approval": None,  # year first approved anywhere (ChEMBL), or None
            }
        return drug_facts[key]

    async def _get_drug_intake(drug_name: str) -> DrugIntake:
        """Await the run's shared intake task and store its result once available."""
        normalized = normalize_drug_name(drug_name)
        intake = await _get_shared_drug_intake(
            drug_intake_tasks,
            normalized,
            svc.cache_dir,
            date_before,
        )
        entry = _ensure_drug_entry(normalized)
        entry["chembl_id"] = intake.chembl_id
        entry["drug_aliases"] = list(intake.aliases)
        entry["first_approval"] = intake.first_approval
        existing = {ind.lower().strip() for ind in entry["approved_indications"]}
        for indication in intake.approved_indications:
            key = indication.lower().strip()
            if key not in existing:
                entry["approved_indications"].append(indication)
                existing.add(key)
        return intake

    def _render_briefing(drug_name: str) -> str:
        """Render drug_facts[drug_name] as a markdown briefing."""
        entry = drug_facts.get(_drug_key(drug_name))
        if entry is None:
            return f"DRUG INTAKE: {drug_name}\n- (no facts collected yet)"

        lines = [f"DRUG INTAKE: {entry['drug_name']}"]

        if entry["drug_aliases"]:
            lines.append(f"- Trade/generic names: {', '.join(entry['drug_aliases'])}")
        else:
            lines.append("- Trade/generic names: (not yet resolved)")

        if entry["approved_indications"]:
            lines.append("- FDA-approved indications:")
            for ind in entry["approved_indications"]:
                lines.append(f"  - {ind}")
        else:
            lines.append("- FDA-approved indications: (none discovered in this run)")

        if entry["mechanism_targets"]:
            target_strs = [f"{g} ({a})" for g, a in entry["mechanism_targets"]]
            lines.append(f"- Targets: {', '.join(target_strs)}")
        else:
            lines.append("- Targets: (mechanism agent has not run)")

        if entry["mechanism_disease_associations"]:
            lines.append("- Top mechanism-disease associations:")
            # cap at 10 to keep briefing terse
            for g, d, s in entry["mechanism_disease_associations"][:10]:
                # Hide the score when it is the placeholder (not surfaced by MechanismCandidate).
                score_str = f" (score {s:.2f})" if s > 0 else ""
                lines.append(f"  - {g} → {d}{score_str}")

        return "\n".join(lines)

    @tool(response_format="content_and_artifact")
    async def find_candidates(drug_name: str) -> tuple[str, list[str]]:
        """Surface candidate diseases for repurposing this drug.

        Returns the post-dedup candidate list — diseases where competitor drugs (sharing the same molecular targets) are
        being developed, PLUS diseases surfaced by the mechanism sub-agent. Waits for analyze_mechanism to finish, then runs
        merge_and_dedup which performs exact-match dedup (ID, name, OT name-resolve) followed by a hierarchical LLM pass to
        collapse super/subtype overlaps. Each line in the content string is tagged [competitor], [mechanism], or [both].
        """
        try:
            return await _find_candidates_impl(drug_name)
        finally:
            # Always release the seed gate so a failure doesn't deadlock analyze_literature / analyze_clinical_trials,
            # which will see an empty allowlist and reject.
            find_candidates_done.set()

    async def _find_candidates_impl(drug_name: str) -> tuple[str, list[str]]:
        drug_name = normalize_drug_name(drug_name)

        # Drug-level intake via the shared seed_drug_intake helper (also used by the CLI pair-runner): resolve the drug
        # and gather aliases, first_approval and FDA-approved indications. The approved-DROP filter below stays here.
        intake = await _get_drug_intake(drug_name)
        chembl_id = intake.chembl_id
        competitors = await svc.get_drug_competitors(chembl_id, date_before=date_before)
        diseases = list(competitors.keys())

        entry = _ensure_drug_entry(drug_name)

        # Drop competitor diseases already approved for this drug: hardcoded table when date_before is set, live FDA otherwise.
        fda_approved_lower: set[str] = set()
        if diseases:
            if date_before is not None:
                fda_approved = await get_approved_indications(
                    drug_name=drug_name,
                    candidate_diseases=diseases,
                    as_of=date_before,
                )
            else:
                mapping = await get_fda_approved_disease_mapping(
                    drug_name=drug_name,
                    candidate_diseases=diseases,
                    approved_indications=list(intake.approved_indications),
                    cache_dir=svc.cache_dir,
                )
                fda_approved = {
                    disease for disease, label in mapping.items() if label == "approved"
                }
                # Record non-"approved" labels for KEPT candidates so the report renders them from the label, not LLM
                # prose. "none" is omitted. The holdout path has no label data, so labels are captured only in the live path.
                for disease, label in mapping.items():
                    if label in ("contaminated", "combination_only"):
                        approval_labels[disease.lower().strip()] = label
            if fda_approved:
                _log_disease_banner(
                    f"FDA-DROPPED (already approved for {drug_name}, source: "
                    f"{'hardcoded table' if date_before is not None else 'live FDA'})",
                    sorted(fda_approved),
                )
                # Record the approved indications in the shared store: dropped from candidates, but the supervisor needs
                # them for subset/superset reasoning (e.g. CML approval makes a "myeloid leukemia" candidate ambiguous).
                existing = {
                    ind.lower().strip() for ind in entry["approved_indications"]
                }
                # Candidate order, not set order: this list is embedded in every sub-agent prompt, so a stable order keeps
                # prompts byte-identical between runs (the pipeline replay matches requests on body).
                for ind in diseases:
                    if ind in fda_approved and ind.lower().strip() not in existing:
                        entry["approved_indications"].append(ind)
            fda_approved_lower = {d.lower().strip() for d in fda_approved}

        diseases = [d for d in diseases if d.lower().strip() not in fda_approved_lower]

        # Cap applies to competitor entries only; mechanism-promoted entries are appended after the merge (already capped
        # upstream by settings.mechanism_top_candidates).
        candidate_cap = get_settings().supervisor_candidate_cap
        if len(diseases) > candidate_cap:
            logger.warning(
                "[TOOL] find_candidates capping competitor candidates from %d to %d "
                "(SUPERVISOR_CANDIDATE_CAP)",
                len(diseases),
                candidate_cap,
            )
            diseases = diseases[:candidate_cap]

        logger.warning(f"{diseases=}")
        allowed_diseases.clear()
        allowed_efo_ids.clear()
        for d in diseases:
            allowed_diseases[d.lower().strip()] = (d, "competitor")

        # EFO IDs for the competitor allowlist, from the raw OT cache, to dedup mechanism candidates by ontology ID.
        # Names that don't resolve (e.g. renamed by the LLM merge) get no entry and fall back to name match.
        async with OpenTargetsClient(cache_dir=svc.cache_dir) as ot_client:
            raw = await ot_client.get_drug_competitors(
                chembl_id, date_before=date_before
            )
        for disease_lower in allowed_diseases:
            efo_id = raw["disease_efo_ids"].get(disease_lower)
            if efo_id:
                allowed_efo_ids[efo_id] = disease_lower

        _log_disease_banner(
            f"CANDIDATE ALLOWLIST for {drug_name} ({chembl_id}) — competitor source",
            diseases,
        )

        # Wait for analyze_mechanism to fill mechanism_candidates_buffer. Its finally block always sets this gate, so a
        # crash yields an empty mechanism contribution, not a deadlock.
        await analyze_mechanism_done.wait()

        # Run the merge + dedup pipeline; find_candidates_done is set in its finally so downstream readers only see the
        # post-dedup allowlist.
        await merge_and_dedup(drug_name)

        # Snapshot the post-dedup allowlist in insertion order: competitor entries (OT order), then mechanism-only
        # entries; "both" entries keep their competitor slot.
        merged: list[tuple[str, str]] = [
            (canonical, source) for (canonical, source) in allowed_diseases.values()
        ]
        n_competitor = sum(1 for _, s in merged if s == "competitor")
        n_both = sum(1 for _, s in merged if s == "both")
        n_mechanism = sum(1 for _, s in merged if s == "mechanism")

        lines = [
            f"Found {len(merged)} candidate diseases for {drug_name} ({chembl_id}) — "
            f"{n_competitor} competitor, {n_both} both, {n_mechanism} mechanism-only:"
        ]
        for i, (canonical, source) in enumerate(merged, start=1):
            lines.append(f"  {i}. {canonical} [{source}]")
        content = "\n".join(lines)

        merged_names = [canonical for canonical, _ in merged]
        emit_progress(PHASE_CANDIDATES, f"Found {len(merged)} candidate diseases")
        return content, merged_names

    async def merge_and_dedup(drug_name: str) -> None:
        """Merge buffered mechanism candidates into the competitor-seeded allowlist.

        Single chokepoint for candidate-disease deduplication:
          1. Exact ID match: drop a mechanism candidate whose disease_id is already in allowed_efo_ids; upgrade the
             matched competitor entry's source to "both".
          2. Exact name match: same for a lowercased name already in allowed_diseases.
          3. OT name-resolve: resolve unresolved mechanism names to EFO IDs and retry step 1.
          4. Synonym collapse: merge same-disease entries per the hardcoded DISEASE_SYNONYM_CANONICAL table.
          5. LLM merge: the competitor path's merge, re-run over the full list; only groups with a mechanism-route
             name are applied.
          6. Hierarchical LLM pass: identify super/subtype overlaps (UC ⊂ IBD, T2DM ⊂ DM) and pick one survivor each.

        Sets find_candidates_done before returning; the find_candidates wrapper's try/finally also sets it on exceptions.
        """
        try:
            await _merge_and_dedup_impl(drug_name)
        finally:
            find_candidates_done.set()

    async def _merge_and_dedup_impl(drug_name: str) -> None:
        # Steps 1-3: exact-match dedup against the competitor-seeded allowlist.
        promoted: list[str] = []
        if mechanism_candidates_buffer:
            async with OpenTargetsClient(cache_dir=svc.cache_dir) as ot_client:
                for candidate in mechanism_candidates_buffer:
                    key = candidate.disease_name.lower().strip()
                    if not key:
                        continue

                    existing_key: str | None = None
                    if candidate.disease_id and candidate.disease_id in allowed_efo_ids:
                        existing_key = allowed_efo_ids[candidate.disease_id]
                    elif key in allowed_diseases:
                        existing_key = key
                    else:
                        resolved_id = await ot_client.resolve_disease_id(
                            candidate.disease_name
                        )
                        if resolved_id and resolved_id in allowed_efo_ids:
                            existing_key = allowed_efo_ids[resolved_id]

                    if existing_key is not None:
                        existing_name, source = allowed_diseases[existing_key]
                        if source == "competitor":
                            allowed_diseases[existing_key] = (existing_name, "both")
                        if (
                            candidate.disease_id
                            and candidate.disease_id not in allowed_efo_ids
                        ):
                            allowed_efo_ids[candidate.disease_id] = existing_key
                    else:
                        allowed_diseases[key] = (candidate.disease_name, "mechanism")
                        if candidate.disease_id:
                            allowed_efo_ids[candidate.disease_id] = key
                        promoted.append(candidate.disease_name)

        if promoted:
            _log_disease_banner(
                f"MECHANISM-PROMOTED candidates added to allowlist for {drug_name}",
                promoted,
            )

        # Step 4: deterministic synonym collapse of same-disease entries (DISEASE_SYNONYM_CANONICAL); exact synonyms only.
        collapsed = collapse_synonym_entries(allowed_diseases, allowed_efo_ids)
        if collapsed:
            _log_disease_banner(
                f"SYNONYM-COLLAPSED candidates for {drug_name}",
                [f"{dropped} → {survivor}" for dropped, survivor in collapsed],
            )

        # Step 5: LLM merge over the full list. Mechanism names arrive after the competitor path's merge, so the same
        # disease under a second name survives. Only groups with a mechanism entry are applied.
        merged_by_llm = await merge_mechanism_entries(
            drug_name,
            allowed_diseases,
            allowed_efo_ids,
            list(_ensure_drug_entry(drug_name)["approved_indications"]),
        )
        if merged_by_llm:
            _log_disease_banner(
                f"LLM-MERGED mechanism candidates for {drug_name}",
                [f"{dropped} → {survivor}" for dropped, survivor in merged_by_llm],
            )

        # Step 6: hierarchical LLM pass — temporarily disabled: it collapsed actionable subtypes (PCOS, gestational
        # diabetes) into broad parents (metabolic disease) for broadly-acting drugs like metformin. All candidates from
        # the exact-match dedup are kept until a hardcoded equivalence-group approach is decided.
        # if len(allowed_diseases) < 2:
        #     return
        #
        # # Build the (name, source, efo_id) tuple list in insertion order. For each
        # # row we need the EFO ID, which is stored inverted in allowed_efo_ids.
        # name_to_efo: dict[str, str] = {}
        # for efo_id, lc_name in allowed_efo_ids.items():
        #     if lc_name in allowed_diseases:
        #         name_to_efo[lc_name] = efo_id
        #
        # candidate_tuples: list[tuple[str, str, str | None]] = []
        # for lc_name, (canonical, source) in allowed_diseases.items():
        #     candidate_tuples.append((canonical, source, name_to_efo.get(lc_name)))
        #
        # decisions = await run_hierarchical_dedup(
        #     drug_name=drug_name,
        #     mechanism_targets=list(mechanism_targets_for_dedup),
        #     candidates=candidate_tuples,
        # )
        # if not decisions.decisions:
        #     return
        #
        # # Apply decisions: remove dropped entries from allowed_diseases and
        # # allowed_efo_ids. A name appearing in multiple decisions is removed once.
        # canonical_to_lc: dict[str, str] = {
        #     canonical: lc for lc, (canonical, _) in allowed_diseases.items()
        # }
        # removed_canonicals: list[str] = []
        # removed_lc: set[str] = set()
        # for decision in decisions.decisions:
        #     for dropped_name in decision.dropped:
        #         lc = canonical_to_lc.get(dropped_name)
        #         if lc is None or lc in removed_lc:
        #             continue
        #         removed_lc.add(lc)
        #         removed_canonicals.append(dropped_name)
        #         logger.warning(
        #             "[DEDUP] %r → %r (%s)",
        #             dropped_name,
        #             decision.survivor,
        #             decision.reason or "no reason given",
        #         )
        #         allowed_diseases.pop(lc, None)
        #         # Invert lookup: drop any allowed_efo_ids row that points at this lc.
        #         for efo_id in [e for e, v in allowed_efo_ids.items() if v == lc]:
        #             allowed_efo_ids.pop(efo_id, None)
        #
        # if removed_canonicals:
        #     _log_disease_banner(
        #         f"HIERARCHICAL DEDUP removed candidates for {drug_name}",
        #         removed_canonicals,
        #     )
        #     survivors = [canonical for canonical, _ in allowed_diseases.values()]
        #     _log_disease_banner(
        #         f"CANDIDATE ALLOWLIST for {drug_name} after hierarchical dedup",
        #         survivors,
        #     )

    def _reject(
        disease_name: str, tool_label: str, empty_output: _EmptyOutputT
    ) -> tuple[str, _EmptyOutputT]:
        valid = sorted(allowed_diseases.keys())
        msg = (
            f"REJECTED: '{disease_name}' is not in the allowed candidate list. "
            f"You must call {tool_label} only with a disease_name returned VERBATIM by "
            f"find_candidates or added from mechanism associations. "
            f"Do not reword, substitute synonyms, or introduce diseases from training knowledge. "
            f"Valid candidates: {valid}"
        )
        # logger.warning("[TOOL] %s REJECTED disease=%r", tool_label, disease_name)
        return msg, empty_output

    @tool(response_format="content_and_artifact")
    async def analyze_literature(
        drug_name: str, disease_name: str
    ) -> tuple[str, LiteratureOutput]:
        """Run a full literature analysis for a drug-disease pair.

        Investigates published evidence via PubMed, embeds and re-ranks abstracts, and produces a structured evidence
        summary with strength rating (none / weak / moderate / strong).
        """
        # Wait for both seed tools to populate the allowlist; otherwise a parallel call can reject legitimate diseases.
        await find_candidates_done.wait()
        await analyze_mechanism_done.wait()

        drug_name = normalize_drug_name(drug_name)
        # Fresh agent per call so literature_tools' closure-scoped store dict is not shared across diseases.
        if disease_name.lower().strip() not in allowed_diseases:
            return _reject(disease_name, "analyze_literature", LiteratureOutput())

        # logger.warning("[TOOL] analyze_literature(drug=%r, disease=%r)", drug_name, disease_name)

        # Per-call DB session from the shared pool: the fan-out is concurrent and a SQLAlchemy Session is not safe for
        # concurrent use (sharing one parks a connection in an open transaction and hangs Postgres). Closed below.
        # The approved-indication list is read as a literal: find_candidates seeds it before any fan-out. It is threaded
        # into the literature agent so synthesize excludes papers about an approved sub-indication of a broad candidate.
        approved_indications = list(
            _ensure_drug_entry(drug_name)["approved_indications"]
        )
        drug_profile = _ensure_drug_entry(drug_name)["drug_profile"]
        _t0 = time.perf_counter()
        with session_factory() as call_db:
            lit_agent = build_literature_agent(
                llm=llm,
                svc=svc,
                db=call_db,
                safety_svc=safety_svc,
                date_before=date_before,
                approved_indications=approved_indications,
                drug_profile=drug_profile,
            )
            with candidate_cost_scope(disease_name):
                output = await run_literature_agent(lit_agent, drug_name, disease_name)
        # logger.warning(
        #     "[TIMING] literature %s: %.1fs", disease_name, time.perf_counter() - _t0
        # )
        strength = (
            output.evidence_summary.strength if output.evidence_summary else "no data"
        )
        direction = (
            output.evidence_summary.direction if output.evidence_summary else "no data"
        )
        # is_observational is an authoritative design fact (True observational, False RCT-backed, None undetermined);
        # surface it verbatim so the blurb doesn't call RCT-backed evidence observational.
        is_observational = (
            output.evidence_summary.is_observational
            if output.evidence_summary
            else None
        )
        design = (
            "observational"
            if is_observational is True
            else "rct_or_controlled" if is_observational is False else "undetermined"
        )
        # evidence_basis says WHY strength may be none: drug_specific, class_level (other drugs in the class), approved
        # (an approved sub-indication), or none (no relevant abstracts). Ranking weighs class_level/approved differently
        # from a genuine no-abstracts pair.
        basis = (
            output.evidence_summary.evidence_basis
            if output.evidence_summary
            else "none"
        )
        # Terse flag only (the full safety_summary is reserved for the rendered report).
        safety_flag = _safety_flag(output.evidence_summary)
        safety_note = f", {safety_flag}" if safety_flag else ""
        header = (
            f"Literature for {drug_name} × {disease_name}: "
            f"{len(output.pmids)} PMIDs, strength={strength}, direction={direction}, "
            f"study_design={design}, evidence_basis={basis}{safety_note}."
        )
        # Build the supervisor-facing summary deterministically from EvidenceSummary so adverse-signal language arrives verbatim.
        synth_summary = (
            output.evidence_summary.summary if output.evidence_summary else ""
        )
        key_findings = (
            output.evidence_summary.key_findings if output.evidence_summary else []
        )
        parts = [header]
        if synth_summary:
            parts.append(synth_summary)
        if key_findings:
            parts.append("Key findings:\n" + "\n".join(f"- {f}" for f in key_findings))
        summary = "\n\n".join(parts)
        # Write-through for the top-N evidence gate in finalize_supervisor.
        slot = findings_local.setdefault(
            disease_name.lower().strip(), {"literature": None, "clinical_trials": None}
        )
        slot["literature"] = output
        return summary, output

    @tool(response_format="content_and_artifact")
    async def analyze_clinical_trials(
        drug_name: str, disease_name: str
    ) -> tuple[str, ClinicalTrialsOutput]:
        """Run a full clinical trials analysis for a drug-disease pair.

        Checks ClinicalTrials.gov for existing trials, competitive landscape, and terminated trials (safety/efficacy red
        flags).
        """
        # Wait for both seed tools to finish populating the allowlist (see analyze_literature).
        await find_candidates_done.wait()
        await analyze_mechanism_done.wait()

        drug_name = normalize_drug_name(drug_name)
        if disease_name.lower().strip() not in allowed_diseases:
            return _reject(
                disease_name, "analyze_clinical_trials", ClinicalTrialsOutput()
            )

        # logger.warning("[TOOL] analyze_clinical_trials(drug=%r, disease=%r)", drug_name, disease_name)
        _t0 = time.perf_counter()
        # first_approval (seeded in find_candidates) lets the closure judgment tell an old generic from a genuine
        # negative; None when unresolved, passed through as-is.
        seed_entry = drug_facts.get(_drug_key(drug_name))
        first_approval = seed_entry.get("first_approval") if seed_entry else None
        registry_drug = (
            seed_entry["drug_aliases"][0]
            if seed_entry and seed_entry["drug_aliases"]
            else drug_name
        )
        # Fresh agent per call so concurrent investigations don't accumulate each other's trials in shown_by_pair.
        ct_agent = build_clinical_trials_agent(
            llm=llm,
            date_before=date_before,
            assigned_indication=disease_name,
            target_drug=registry_drug,
            cache_dir=svc.cache_dir,
        )
        # Approved indications are seeded by find_candidates before fan-out; threaded into the task so the relevance
        # gate's TEST 1 treats an approved sub-indication's trial as contamination, not roll-up evidence.
        ct_approved = list(_ensure_drug_entry(drug_name)["approved_indications"])
        with candidate_cost_scope(disease_name):
            output = await run_clinical_trials_agent(
                ct_agent,
                registry_drug,
                disease_name,
                first_approval=first_approval,
                approved_indications=ct_approved,
            )
        logger.debug(
            "[TIMING] clinical_trials %s: %.1fs",
            disease_name,
            time.perf_counter() - _t0,
        )

        # Drug-level write-through: when the FDA check matches the candidate to an approved indication, record it in the
        # briefing. Trial data still flows to the summary; the sub-agent always investigates fully.
        approval = output.approval
        if approval is not None and approval.is_approved:
            entry = _ensure_drug_entry(drug_name)
            matched = approval.matched_indication or disease_name
            existing = {ind.lower().strip() for ind in entry["approved_indications"]}
            if matched.lower().strip() not in existing:
                entry["approved_indications"].append(matched)

        completed = output.completed
        terminated = output.terminated
        search_coverage = output.search_coverage
        completed_coverage = output.completed_coverage
        terminated_coverage = output.terminated_coverage
        n_recruiting = (
            search_coverage.relevant_by_status.get("RECRUITING", 0)
            if search_coverage
            else None
        )
        n_active_not_recruiting = (
            search_coverage.relevant_by_status.get("ACTIVE_NOT_RECRUITING", 0)
            if search_coverage
            else None
        )
        n_withdrawn = (
            search_coverage.relevant_by_status.get("WITHDRAWN", 0)
            if search_coverage
            else None
        )
        relevant_ids = set(output.relevant_nct_ids)
        completed_trials = (
            [trial for trial in completed.trials if trial.nct_id in relevant_ids]
            if completed
            else []
        )
        terminated_trials = (
            [trial for trial in terminated.trials if trial.nct_id in relevant_ids]
            if terminated
            else []
        )
        n_safety_efficacy_shown = (
            sum(
                1
                for t in terminated_trials
                if _classify_stop_reason(t.why_stopped) in {"safety", "efficacy"}
            )
            if terminated_coverage
            else None
        )
        header = (
            f"Clinical trials for {drug_name} × {disease_name}: "
            f"all-status {_trial_evidence_text(search_coverage)}. "
            f"Relevant reviewed statuses: {n_recruiting if n_recruiting is not None else 'unavailable'} recruiting, "
            f"{n_active_not_recruiting if n_active_not_recruiting is not None else 'unavailable'} active, "
            f"{n_withdrawn if n_withdrawn is not None else 'unavailable'} withdrawn. "
            f"Completed scope: {_trial_evidence_text(completed_coverage)}. "
            f"Terminated scope: {_trial_evidence_text(terminated_coverage)}; "
            f"{n_safety_efficacy_shown if n_safety_efficacy_shown is not None else 'unavailable'} "
            "reviewed relevant safety/efficacy stops."
        )

        completed_phase_dist = _phase_distribution(completed_trials)
        terminated_phase_dist = _phase_distribution(terminated_trials)

        completed_top = _borda_rank_by_enrollment_and_recency(completed_trials, k=10)
        terminated_top = _borda_rank_by_enrollment_and_recency(terminated_trials, k=10)

        completed_table = _format_trial_table(
            completed_top,
            columns=(
                "nct_id",
                "phase",
                "start_date",
                "completion_date",
                "refs",
                "mesh",
                "title",
            ),
            cap=10,
        )
        terminated_table = _format_trial_table(
            terminated_top,
            columns=(
                "nct_id",
                "phase",
                "stop_reason",
                "start_date",
                "completion_date",
                "refs",
                "mesh",
                "title",
            ),
            cap=10,
            include_why_stopped=True,
            stop_classifier=_classify_stop_reason,
        )

        structured = (
            f"\n\nPhase distribution (completed): {completed_phase_dist}\n"
            f"Phase distribution (terminated): {terminated_phase_dist}\n\n"
            f"Completed trials (top 10 by enrollment + recency):\n"
            f"{completed_table}\n\n"
            f"Terminated trials (top 10 by enrollment + recency):\n"
            f"{terminated_table}"
        )

        signals = output.signals
        signals_block = (
            format_derived_signals(signals)
            if signals is not None
            else "Relevance-filtered trial signals unavailable."
        )
        if output.contaminated_nct_ids:
            signals_block += (
                f"\n  contaminated_excluded: {len(output.contaminated_nct_ids)} trial(s) "
                f"({', '.join(output.contaminated_nct_ids)})"
            )

        # Closure is a typed field from the post-loop synthesis; the supervisor trusts it. The prose summary is
        # human-report context only.
        summary = f"{header}{structured}\n\n{signals_block}"
        closure_line = f"closure={output.closure}"
        if output.closure_reason:
            closure_line += f" — {output.closure_reason}"
        summary = (
            f"{summary}\n\nSub-agent closure verdict (trust, do not re-judge closure):\n"
            f"{closure_line}"
        )
        if output.summary:
            summary = f"{summary}\n\nSub-agent trial-section prose (context):\n{output.summary}"
        # Write-through for the top-N evidence gate in finalize_supervisor.
        slot = findings_local.setdefault(
            disease_name.lower().strip(), {"literature": None, "clinical_trials": None}
        )
        slot["clinical_trials"] = output
        return summary, output

    @tool(response_format="content_and_artifact")
    async def analyze_mechanism(drug_name: str) -> tuple[str, MechanismOutput]:
        """Run the mechanism sub-agent for a drug.

        The mechanism agent returns target-level MoA data and the agent's narrative summary. Raw mechanism candidates are
        buffered for the centralized merge_and_dedup pass that find_candidates runs once both seed tools have completed.
        """
        try:
            return await _analyze_mechanism_impl(drug_name)
        finally:
            # Always release the seed gate so a failure here doesn't deadlock analyze_literature / analyze_clinical_trials.
            analyze_mechanism_done.set()

    async def _analyze_mechanism_impl(drug_name: str) -> tuple[str, MechanismOutput]:
        drug_name = normalize_drug_name(drug_name)
        intake = await _get_drug_intake(drug_name)
        _t0 = time.perf_counter()
        output = await run_mechanism_agent(
            mech_agent,
            drug_name,
            approved_indications=list(intake.approved_indications),
            date_before=date_before,
        )
        logger.debug("[TOOL] analyze_mechanism(drug=%r)", drug_name)
        logger.debug("[TIMING] analyze_mechanism: %.1fs", time.perf_counter() - _t0)

        # Buffer raw mechanism candidates for find_candidates to consume in merge_and_dedup() after both seed tools finish.
        mechanism_candidates_buffer.clear()
        raw_received: list[str] = []
        for candidate in output.candidates:
            if not (candidate.disease_name or "").strip():
                continue
            mechanism_candidates_buffer.append(candidate)
            raw_received.append(candidate.disease_name)

        if raw_received:
            _log_disease_banner(
                f"MECHANISM-RAW candidates received for {drug_name}",
                raw_received,
            )

        # Drug-level write-through: populate mechanism_targets and mechanism_disease_associations in the shared store,
        # per-MoA so the briefing can show "ABL1 (INHIBITOR), KIT (INHIBITOR)".
        entry = _ensure_drug_entry(drug_name)
        target_pairs: list[tuple[str, str]] = []
        seen_target_pairs: set[tuple[str, str]] = set()
        for moa in output.mechanisms_of_action:
            for sym in moa.target_symbols:
                pair = (sym, moa.action_type or "UNKNOWN")
                if pair not in seen_target_pairs:
                    seen_target_pairs.add(pair)
                    target_pairs.append(pair)
        entry["mechanism_targets"] = target_pairs

        # Mechanism candidates carry the high-score target→disease associations but not the raw scores, so record each
        # pair without a score.
        assocs: list[tuple[str, str, float]] = []
        seen_assoc_pairs: set[tuple[str, str]] = set()
        for cand in output.candidates:
            pair_key = (cand.target_symbol, cand.disease_name)
            if pair_key in seen_assoc_pairs:
                continue
            seen_assoc_pairs.add(pair_key)
            # Score not surfaced on MechanismCandidate; 0.0 is a placeholder the briefing renderer hides.
            assocs.append((cand.target_symbol, cand.disease_name, 0.0))
        entry["mechanism_disease_associations"] = assocs

        # Record the mechanism target list for merge_and_dedup() to pass to the hierarchical LLM pass.
        mechanism_targets_for_dedup.clear()
        mechanism_targets_for_dedup.extend(target_pairs)

        n_targets = len(output.drug_targets)
        header = (
            f"Mechanism analysis for {drug_name}: {n_targets} targets, "
            f"{len(output.candidates)} mechanism candidates "
            f"(merge into allowlist deferred to find_candidates)."
        )
        sub_agent_summary = output.summary or ""
        summary = (
            f"{header}\n\n{sub_agent_summary}".strip() if sub_agent_summary else header
        )
        emit_progress(
            PHASE_MECHANISM,
            f"Analyzed mechanism: {n_targets} targets, "
            f"{len(output.candidates)} associations",
        )
        return summary, output

    @tool
    def get_drug_briefing(drug_name: str) -> str:
        """Return the accumulated drug-level briefing for this drug.

        Read-only view of facts collected by find_candidates, analyze_mechanism, and analyze_clinical_trials during this
        run: ChEMBL aliases, FDA-approved indications discovered, mechanism targets, and mechanism disease associations.
        Call this before finalize_supervisor to check whether any candidate is related to an approved indication
        (subset/superset/sibling).
        """
        drug_name = normalize_drug_name(drug_name)
        return _render_briefing(drug_name)

    async def _record_missing_approval_labels(drug_name: str) -> None:
        """Fill `approval_labels` for allowlist entries that have no label yet.

        find_candidates records labels for competitor diseases only, and the mechanism agent doesn't report its labels
        back, so a contaminated mechanism-route candidate would be invisible to the skip below. Labels are cached per
        (drug, disease), so this is a cache read after the first run.
        """
        missing = [
            canonical
            for key, (canonical, _) in allowed_diseases.items()
            if key not in approval_labels
        ]
        if not missing:
            return
        entry = _ensure_drug_entry(drug_name)
        try:
            mapping = await get_fda_approved_disease_mapping(
                drug_name=drug_name,
                candidate_diseases=missing,
                approved_indications=list(entry["approved_indications"]),
                cache_dir=svc.cache_dir,
            )
        except Exception as e:
            logger.warning(
                "_record_missing_approval_labels: approval check failed for %r: %s; investigating every candidate",
                drug_name,
                e,
            )
            return
        for disease, label in mapping.items():
            if label in ("contaminated", "combination_only"):
                approval_labels[disease.lower().strip()] = label

    # Fan-out tool: bulk-investigate the top-N candidates (settings.supervisor_investigation_cap, env-tunable via
    # SUPERVISOR_INVESTIGATION_CAP). The supervisor LLM systematically skipped "obvious" candidates like T2DM for
    # semaglutide regardless of prompt instructions (scripts/probe_supervisor_t2dm.py), so the skip is removed in code.
    settings = get_settings()
    investigation_cap = settings.supervisor_investigation_cap
    investigation_concurrency = settings.supervisor_investigation_concurrency

    @tool(response_format="content_and_artifact")
    async def investigate_top_candidates(
        drug_name: str,
    ) -> tuple[str, list[dict]]:
        """Auto-investigate the top-N candidates from the merged allowlist.

        Runs analyze_literature AND analyze_clinical_trials in parallel for the top supervisor_investigation_cap candidates
        by mechanism+competitor strength. Removes the LLM's ability to skip "obvious" candidates that evaluations
        specifically need to recover.

        Call this ONCE, after find_candidates and analyze_mechanism complete. After this returns, you may still call
        analyze_literature / analyze_clinical_trials for candidates beyond the top N if you want.
        """
        # Wait for both seed tools to finish populating the allowlist.
        await find_candidates_done.wait()
        await analyze_mechanism_done.wait()

        drug_name = normalize_drug_name(drug_name)

        # Skip candidates the approval check labelled "contaminated": a broad parent whose approved child covers the
        # retrievable evidence (diabetes mellitus for metformin). They stay in the candidate list but don't consume an
        # investigation slot. Holdout runs have no label data, so nothing is skipped there.
        if date_before is None:
            await _record_missing_approval_labels(drug_name)
        eligible = [
            (key, value)
            for key, value in allowed_diseases.items()
            if approval_labels.get(key) != "contaminated"
        ]
        skipped = [
            canonical
            for key, (canonical, _) in allowed_diseases.items()
            if approval_labels.get(key) == "contaminated"
        ]
        if skipped:
            _log_disease_banner(
                f"NOT INVESTIGATED (approved-parent contamination) for {drug_name}",
                skipped,
            )

        # Top-N from the merged allowlist in insertion order (competitor ranking, then mechanism-promoted entries).
        # supervisor_candidate_cap is not used here; it only trims the final ranked list.
        top_n = eligible[:investigation_cap]
        if not top_n:
            return "No candidates in allowlist; nothing to investigate.", []

        canonical_diseases = [canonical for _, (canonical, _) in top_n]
        drug_entry = _ensure_drug_entry(drug_name)
        chembl_id = drug_entry["chembl_id"]
        if chembl_id is None:
            raise RuntimeError(
                f"Missing ChEMBL ID for {drug_name} after candidate discovery"
            )
        if drug_entry["drug_profile"] is None:
            drug_entry["drug_profile"] = await svc.build_drug_profile(chembl_id)
        candidate_semaphore = asyncio.Semaphore(investigation_concurrency)
        _log_disease_banner(
            f"INVESTIGATING top candidates for {drug_name} (lit + trials in parallel)",
            canonical_diseases,
        )

        # Fan out analyze_literature + analyze_clinical_trials in parallel. Pass a ToolCall-shaped dict so .ainvoke()
        # returns a ToolMessage with .artifact populated; a plain dict returns only the content string.
        async def _invest(disease: str) -> tuple[str, dict]:
            disease_slug = disease.lower().replace(" ", "_")
            _t0 = time.perf_counter()

            async def _timed_lit() -> Any:
                _lt0 = time.perf_counter()
                msg = await analyze_literature.ainvoke(
                    {
                        "name": "analyze_literature",
                        "args": {"drug_name": drug_name, "disease_name": disease},
                        "id": f"auto_lit_{disease_slug}",
                        "type": "tool_call",
                    }
                )
                # logger.warning(
                #     "[TIMING] investigate %s: lit leg took %.1fs",
                #     disease,
                #     time.perf_counter() - _lt0,
                # )
                return msg

            async def _timed_ct() -> Any:
                _ct0 = time.perf_counter()
                msg = await analyze_clinical_trials.ainvoke(
                    {
                        "name": "analyze_clinical_trials",
                        "args": {"drug_name": drug_name, "disease_name": disease},
                        "id": f"auto_ct_{disease_slug}",
                        "type": "tool_call",
                    }
                )
                # logger.warning(
                #     "[TIMING] investigate %s: trials leg took %.1fs",
                #     disease,
                #     time.perf_counter() - _ct0,
                # )
                return msg

            lit_msg, ct_msg = await asyncio.gather(_timed_lit(), _timed_ct())
            # logger.warning(
            #     "[TIMING] investigate %s: lit+trials took %.1fs",
            #     disease,
            #     time.perf_counter() - _t0,
            # )

            lit_artifact = lit_msg.artifact
            ct_artifact = ct_msg.artifact
            # Stash artifacts in the closure for run_supervisor_agent; the ReAct loop doesn't see these direct invocations.
            auto_findings[disease.lower().strip()] = {
                "literature": lit_artifact,
                "clinical_trials": ct_artifact,
            }
            strength = (
                lit_artifact.evidence_summary.strength
                if lit_artifact and lit_artifact.evidence_summary
                else "no data"
            )
            direction = (
                lit_artifact.evidence_summary.direction
                if lit_artifact and lit_artifact.evidence_summary
                else "no data"
            )
            n_pmids = len(lit_artifact.pmids) if lit_artifact else 0
            # Animal-only literature flag (True = every relevant drug-specific study is a non-human model; False = at
            # least one human study; None = nothing to grade), so the critic can demote animal-only support.
            is_animal_only = (
                lit_artifact.evidence_summary.is_animal_only
                if lit_artifact and lit_artifact.evidence_summary
                else None
            )
            # Terse severity-aware safety flag ("" = no signal, NOT confirmed-safe).
            safety_flag = _safety_flag(
                lit_artifact.evidence_summary if lit_artifact else None
            )
            search_coverage = ct_artifact.search_coverage if ct_artifact else None
            completed_coverage = ct_artifact.completed_coverage if ct_artifact else None
            terminated_coverage = (
                ct_artifact.terminated_coverage if ct_artifact else None
            )
            n_total = search_coverage.relevant_records if search_coverage else None
            n_completed = (
                completed_coverage.relevant_records if completed_coverage else None
            )
            n_terminated = (
                terminated_coverage.relevant_records if terminated_coverage else None
            )
            n_withdrawn = (
                search_coverage.relevant_by_status.get("WITHDRAWN", 0)
                if search_coverage
                else None
            )
            ct_signals = ct_artifact.signals if ct_artifact else None
            trial_query_unresolved = _trial_query_unresolved(ct_artifact)
            relevant_highest_phase = (
                ct_signals.highest_completed_phase if ct_signals else None
            )
            relevant_phase3_terminated = (
                ct_signals.phase3_terminated_for_cause if ct_signals else False
            )
            return disease, {
                "disease": disease,
                "literature_strength": strength,
                "literature_direction": direction,
                "literature_pmids": n_pmids,
                "literature_is_animal_only": is_animal_only,
                "literature_safety_flag": safety_flag,
                "trials_total": n_total,
                "trials_completed": n_completed,
                "trials_terminated": n_terminated,
                "trials_withdrawn": n_withdrawn,
                "trial_evidence": (
                    _UNRESOLVED_TRIAL_STAGE
                    if trial_query_unresolved
                    else _trial_evidence_text(search_coverage)
                ),
                "trial_query_unresolved": trial_query_unresolved,
                "relevant_highest_phase": relevant_highest_phase,
                "relevant_phase3_terminated_for_cause": relevant_phase3_terminated,
                # Authoritative development-stage tier, seeding the ranking with the same fact the blurb/footer/ranked line
                # render, so a low highest_COMPLETED_phase that ignores active pivotal trials doesn't bias it.
                "dev_stage": ct_signals.dev_stage if ct_signals else None,
                "dev_stage_phrase": _dev_stage_phrase(ct_signals),
            }

        _fan_t0 = time.perf_counter()
        logger.warning(
            "[INVEST] fanning out %d candidates: %s",
            len(canonical_diseases),
            canonical_diseases,
        )
        n_total_candidates = len(canonical_diseases)
        emit_progress(
            PHASE_TRIALS,
            f"Investigating trials + literature for {n_total_candidates} candidates",
        )
        _done_count = 0

        async def _invest_tracked(disease: str) -> tuple[str, dict]:
            nonlocal _done_count
            # Jitter before acquiring the semaphore so concurrently-started candidates (and regression drug processes)
            # don't hit their heavy calls at the same instant. See SUPERVISOR_CANDIDATE_JITTER_MAX_SECONDS.
            await asyncio.sleep(
                random.uniform(0, SUPERVISOR_CANDIDATE_JITTER_MAX_SECONDS)
            )
            async with candidate_semaphore:
                result = await _invest(disease)
            _done_count += 1
            emit_progress(
                PHASE_TRIALS,
                f"Investigated {_done_count}/{n_total_candidates} candidates",
            )
            return result

        results = await asyncio.gather(
            *(_invest_tracked(d) for d in canonical_diseases)
        )
        logger.debug(
            "[TIMING] investigate_top_candidates: %d candidates in parallel took %.1fs",
            len(canonical_diseases),
            time.perf_counter() - _fan_t0,
        )
        artifacts = [a for _, a in results]

        # One-line-per-disease compact summary the LLM can rank against.
        lines = [
            f"Auto-investigated {len(artifacts)} top candidates " f"for {drug_name}:"
        ]
        for a in artifacts:
            phase = a.get("relevant_highest_phase") or "none"
            term_note = (
                "; relevant Phase 3 terminated for cause"
                if a.get("relevant_phase3_terminated_for_cause")
                else ""
            )
            direction = a.get("literature_direction") or "none"
            dir_note = f"/{direction}" if direction not in ("none", "no data") else ""
            # Render the authoritative dev_stage so the LLM ranks on the same fact the report shows.
            if a.get("trial_query_unresolved"):
                stage_note = (
                    f"; authoritative stage {_UNRESOLVED_TRIAL_STAGE}; "
                    f"active programs {_UNRESOLVED_ACTIVE_PROGRAMS}"
                )
            else:
                stage_note = (
                    f"; dev_stage {a['dev_stage']} ({a['dev_stage_phrase']})"
                    if a.get("dev_stage_phrase")
                    else ""
                )
            # Withdrawn-before-enrolling note, only when the pair's ONLY on-record trials are withdrawn (no live or
            # completed trial, so it must not read as a registered candidate).
            n_withdrawn = a.get("trials_withdrawn")
            withdrawn_note = (
                "; ALL relevance-reviewed trial(s) WITHDRAWN before enrolling (never dosed a patient)"
                if n_withdrawn and a.get("trials_total") == n_withdrawn
                else (f"; {n_withdrawn} withdrawn" if n_withdrawn else "")
            )
            # Animal-only note: only True fires; False (human data) and None (nothing to grade) stay quiet.
            animal_note = (
                "; literature is ANIMAL/in-vitro only (no human data)"
                if a.get("literature_is_animal_only") is True
                else ""
            )
            # Terse severity-aware safety flag ("" = no signal, NOT confirmed-safe).
            _sflag = a.get("literature_safety_flag") or ""
            safety_note = f"; {_sflag}" if _sflag else ""
            lines.append(
                f"  - {a['disease']}: literature {a['literature_strength']}{dir_note}, "
                f"{a['literature_pmids']} PMIDs; trials {a['trial_evidence']}; "
                f"{a['trials_completed'] if a['trials_completed'] is not None else 'unavailable'} relevant completed, "
                f"{a['trials_terminated'] if a['trials_terminated'] is not None else 'unavailable'} relevant terminated; "
                f"relevant highest phase {phase}{term_note}{stage_note}{withdrawn_note}{animal_note}{safety_note}"
            )
        # Name the exact candidates the ranking must cover, using finalize's own evidence gate and unresolved-trial rule;
        # left to decide, the model cut the list short at an arbitrary length and misapplied the gate.
        must_rank = [
            a["disease"]
            for a in artifacts
            if not a["trial_query_unresolved"]
            and _evidence_gate_reason(auto_findings[a["disease"].lower().strip()])
            is None
        ]
        if must_rank:
            lines.append(
                f"\nYour ranking (summary and blurbs) must include every one of these "
                f"{len(must_rank)} candidates: {', '.join(must_rank)}."
            )
        return "\n".join(lines), artifacts

    async def _run_fact_critic(items: list[dict]) -> list[dict]:
        """Run the critic over `items`; return the audited blurbs (possibly REORDERED).

        The critic (1) sanity-checks the rank ORDER (a tested-and-failed or closed candidate should not lead a live one)
        and reorders if not defensible; (2) repairs a blurb field that contradicts the authoritative per-disease Phase-3
        FACT while preserving true "no regulatory program" claims. Returns the same disease SET, order may differ. On ANY
        parse failure or disease-set mismatch, returns the ORIGINAL items. Used by critique_ranking and finalize_supervisor.
        """
        lines: list[str] = []
        for i, item in enumerate(items, start=1):
            disease = (item.get("disease") or "").strip() or "(unnamed)"
            verdict = (item.get("verdict") or "").strip()
            literature = (item.get("literature") or "").strip()
            blocker = (item.get("blocker") or "").strip()
            slot = findings_local.get(disease.lower().strip()) or {}
            ct = slot.get("clinical_trials")
            sig = ct.signals if ct else None
            stage_phrase = _dev_stage_phrase(sig)
            if _trial_query_unresolved(ct):
                fact = (
                    f"authoritative trial query state = unresolved; stage = "
                    f"{_UNRESOLVED_TRIAL_STAGE}; active programs = "
                    f"{_UNRESOLVED_ACTIVE_PROGRAMS}"
                )
            elif sig is not None and stage_phrase is not None:
                fact = (
                    f"authoritative dev_stage = {sig.dev_stage} ({stage_phrase}) — "
                    "do not contradict this stage"
                )
            else:
                fact = "no trial signal available"
            coverage = ct.search_coverage if ct else None
            n_total = coverage.relevant_records if coverage else None
            n_withdrawn = (
                coverage.relevant_by_status.get("WITHDRAWN", 0) if coverage else None
            )
            if n_withdrawn and n_total == n_withdrawn:
                fact += (
                    f"; all {n_withdrawn} relevance-reviewed trial(s) WITHDRAWN before enrolling "
                    "(never dosed a patient — not a live registered trial)"
                )
            # Animal-only literature clause: only True adds one. Reads the lit artifact's evidence_summary, the same source
            # as the _invest dict's literature_is_animal_only.
            lit = slot.get("literature")
            if (
                lit
                and lit.evidence_summary
                and lit.evidence_summary.is_animal_only is True
            ):
                fact += (
                    "; supporting literature is ANIMAL/in-vitro only (no human data — "
                    "not clinical evidence)"
                )
            # Terse safety clause. "" means no safety-relevant abstract was found, not that the drug is safe, so silence is
            # correct when there is no signal.
            _sflag = _safety_flag(lit.evidence_summary if lit else None)
            if _sflag:
                fact += f"; literature: {_sflag} (see report for detail)"
            lines.append(
                f"{i}. {disease} | FACT: {fact} | verdict: {verdict or '—'} | "
                f"literature: {literature or '—'} | blocker: {blocker or '—'}"
            )
        prompt = (
            "Current ranking (top to bottom), each with its authoritative FACT:\n"
            + "\n".join(lines)
            + "\n\nFull blurbs to audit and repair (return all of them):\n"
            + json.dumps(items)
        )
        _t0 = time.perf_counter()
        critique = await query_llm(prompt, system=_RANKING_CRITIC_SYSTEM)
        logger.debug("[TIMING] fact_critic LLM call: %.1fs", time.perf_counter() - _t0)
        logger.info(
            "[TOOL] fact_critic IN (%d candidates):\n%s", len(items), "\n".join(lines)
        )
        repaired = items
        # The critic sometimes reasons in prose before the object, so scan for the last balanced object rather than
        # requiring the whole response to be one (which discarded valid repair sets).
        data = parse_last_json_object(critique)
        if data is None:
            logger.warning(
                "[TOOL] fact_critic: no JSON object in critic output — keeping originals. "
                "Response was: %s",
                critique,
            )
            return repaired
        cand = data.get("blurbs")
        if (
            isinstance(cand, list)
            and len(cand) == len(items)
            and all(isinstance(b, dict) for b in cand)
            and {(b.get("disease") or "").strip().lower() for b in cand}
            == {(b.get("disease") or "").strip().lower() for b in items}
        ):
            repaired = cand
        else:
            logger.warning("[TOOL] fact_critic: blurb set mismatch — keeping originals")

        # Closure is an authoritative upstream fact, but the critic can leave a closed candidate above live ones. Apply
        # a stable partition after its review (live before closed), preserving the critic's order within each group.
        def _is_closed(item: dict) -> bool:
            disease = (item.get("disease") or "").strip().lower()
            slot = findings_local.get(disease) or {}
            ct = slot.get("clinical_trials")
            lit = slot.get("literature")
            es = lit.evidence_summary if lit else None
            return _closure_text(ct, es).startswith("CLOSED")

        return [item for item in repaired if not _is_closed(item)] + [
            item for item in repaired if _is_closed(item)
        ]

    @tool
    async def critique_ranking(blurbs: list[dict] | None = None) -> str:
        """Have a fresh reviewer audit your ranking before finalizing.

        Call this AFTER you have drafted your ranked blurbs but BEFORE finalize_supervisor. Pass the same `blurbs` list (in
        your intended rank order) you plan to finalize with. A separate reviewer sanity-checks the ORDER (a
        tested-and-failed or closed candidate should not lead over a live one) and may REORDER, and repairs any field that
        contradicts the candidate's authoritative Phase-3 FACT. FINALIZE WITH THE BLURBS IT RETURNS, IN THE ORDER IT RETURNS
        THEM — they are the reviewed ranking.

        Arguments:
        - blurbs: your draft ranked blurbs, same shape as finalize_supervisor.
        """
        critique_state["ran"] = True
        emit_progress(PHASE_SUMMARY, "Ranking candidates and writing the summary")
        items = blurbs or []
        if not items:
            return "No blurbs provided — nothing to critique."
        repaired = await _run_fact_critic(items)
        critique_state["last_blurbs"] = repaired
        result = "Blurbs fact-checked. Finalize with THESE:\n" + json.dumps(repaired)
        logger.info("[TOOL] critique_ranking OUT:\n%s", result)
        return result

    @tool(response_format="content_and_artifact")
    async def finalize_supervisor(
        summary: str, blurbs: list[dict] | None = None
    ) -> tuple[str, dict]:
        """Signal that the repurposing analysis is complete.

        Call this as the very last step. This terminates the agent loop.

        Arguments:
        - summary: your ranked structured fact list of investigated candidates
          (see WRITING THE SUMMARY in the system prompt).
        - blurbs: a list of structured per-candidate entries, one for each candidate
          named in the must-include list that ends the investigate_top_candidates result, in rank order. Each entry is a
          dict with these keys:
            - disease: <verbatim candidate name from find_candidates or
              analyze_mechanism>
            - stage: where the pair sits in development (single line)
            - literature: one-line summary of the literature evidence base —
              strength + shape, e.g. "Strong, 5 RCTs / meta-analyses",
              "Weak, case reports only", "None"
            - blocker: what is currently holding the program back, or empty
              if nothing is
            - active_programs: short summary of what is still moving
            - key_risk: the single biggest risk to the hypothesis
            - verdict: short interpretive tag (e.g. "Live but bottlenecked")
            - watch: next concrete data readout or trial worth watching
              (NCT id and/or expected timing); empty if none on record
            - prose: exactly 2 sentences of interpretive synthesis
          Each blurb must synthesize ONLY the literature and clinical_trials
          sub-agent summaries you saw for that disease this run — do not
          include mechanism content. Pass an empty list if no candidates were
          investigated. Diseases not in the allowlist are dropped. An entry
          with empty disease, or empty in BOTH prose AND every structured
          field, is dropped.
        """
        # Fact-check gate: refuse to finalize until critique_ranking has run this run. Returns a non-terminal instruction
        # so the loop calls critique_ranking and retries (the empty artifact keeps the content_and_artifact contract).
        if not critique_state["ran"]:
            logger.warning(
                "[TOOL] finalize_supervisor rejected — critique_ranking not called yet"
            )
            return (
                "Cannot finalize yet: you must call critique_ranking exactly once with "
                "your draft blurbs (in rank order) BEFORE finalize_supervisor. Call "
                "critique_ranking now, then finalize_supervisor again with the repaired "
                "blurbs it returns.",
                {},
            )
        logger.info(
            "[TOOL] finalize_supervisor called with %d blurbs", len(blurbs or [])
        )
        # The former finalize-time fact-critic re-run was removed: the interpretive fields are now authored only by the
        # isolated judge_interpretive call in the enrich pass, fed the resolved stage. critique_ranking still runs above.
        #
        # AUTHORITATIVE ORDER: the critic's reordered blurbs (critique_state["last_blurbs"]) are the reviewed ranking.
        # The finalize `blurbs` arg is the LLM's re-passed copy and may silently revert the reorder, so use the critic's
        # list when present and fall back to the passed blurbs only if the critic produced none (e.g. called with []).
        critic_blurbs = critique_state.get("last_blurbs")
        source_blurbs = critic_blurbs if critic_blurbs else (blurbs or [])
        if critic_blurbs:
            logger.info(
                "[TOOL] finalize_supervisor using %d critic-reviewed blurbs (ignoring the LLM's "
                "re-passed order)",
                len(critic_blurbs),
            )
        validated: list[dict] = []
        # Diseases the evidence gate actually dropped, as (disease, reason). The "Evidence gate exclusions:" footer is
        # rebuilt from this, not the LLM's prose, which named a kept candidate (semaglutide × hypoglycemia) on an invented
        # criterion.
        gate_excluded: list[tuple[str, str]] = []
        # Lowercase keys already accepted into `validated`. The critic has returned the same disease twice (metformin x
        # cardiovascular disorder); keep the first entry.
        seen_validated: set[str] = set()
        structured_keys = (
            "stage",
            "literature",
            "blocker",
            "active_programs",
            "key_risk",
            "verdict",
            "watch",
        )
        for item in source_blurbs:
            disease = (item.get("disease") or "").strip()
            if not disease:
                continue
            fields = {k: (item.get(k) or "").strip() for k in structured_keys}
            prose = (item.get("prose") or "").strip()
            if not prose and not any(fields.values()):
                continue
            disease_key = disease.lower().strip()
            if disease_key not in allowed_diseases:
                logger.warning(
                    "[TOOL] finalize_supervisor dropping blurb for disease=%r "
                    "(not in allowlist)",
                    disease,
                )
                continue
            if disease_key in seen_validated:
                logger.warning(
                    "[TOOL] finalize_supervisor dropping duplicate blurb for disease=%r",
                    disease,
                )
                continue
            # Top-N evidence gate: drop candidates with 0 trials AND no usable literature signal (strength="none" OR
            # study_count==0). Literature with at least one relevant abstract and 0 trials is kept. The PMID-count
            # fallback applies only when synthesize didn't run. Filtered candidates still appear in disease_findings.
            slot = findings_local.get(disease_key) or {}
            lit = slot.get("literature")
            ct = slot.get("clinical_trials")
            coverage = ct.search_coverage if ct is not None else None
            gate_reason = _evidence_gate_reason(slot)
            if gate_reason is not None:
                logger.warning(
                    "[TOOL] finalize_supervisor dropping blurb for disease=%r (evidence gate: %s)",
                    disease,
                    gate_reason,
                )
                gate_excluded.append((disease, gate_reason))
                seen_validated.add(disease_key)
                continue
            # Deterministic STAGE override: the development stage is a fact the LLM must not author. Render the
            # clinical-trials sub-agent's authoritative dev_stage phrase verbatim over the LLM's `stage`, so the blurb
            # cannot contradict the trial section. An unresolved query has its own availability phrase; other
            # missing-signal cases keep the LLM's stage.
            sig = ct.signals if ct else None
            stage_phrase = _dev_stage_phrase(sig)
            trial_query_unresolved = _trial_query_unresolved(ct)
            if trial_query_unresolved:
                stage_phrase = _UNRESOLVED_TRIAL_STAGE
                if fields["stage"] != stage_phrase:
                    logger.warning(
                        "[TOOL] finalize_supervisor replaced stage for unresolved trial "
                        "query disease=%r; was: %r",
                        disease,
                        fields["stage"],
                    )
                fields["stage"] = stage_phrase
                fields["active_programs"] = _UNRESOLVED_ACTIVE_PROGRAMS
                fields["blocker"] = ""
                fields["key_risk"] = ""
                fields["verdict"] = ""
                fields["watch"] = ""
                prose = ""
            elif (
                sig is not None
                and stage_phrase is not None
                and stage_phrase != fields["stage"]
            ):
                logger.warning(
                    "[TOOL] finalize_supervisor set stage from dev_stage=%s for disease=%r; "
                    "was: %r",
                    sig.dev_stage,
                    disease,
                    fields["stage"],
                )
                fields["stage"] = stage_phrase
            # active_programs ("what is still moving") is also authoritative from the stage judgment (services/dev_stage);
            # the blurb LLM mis-filled it (e.g. a COMPLETED trial as active). Render it verbatim.
            if not trial_query_unresolved and sig is not None and sig.active_programs:
                if sig.active_programs != fields["active_programs"]:
                    logger.warning(
                        "[TOOL] finalize_supervisor set active_programs from signal for "
                        "disease=%r; was: %r",
                        disease,
                        fields["active_programs"],
                    )
                fields["active_programs"] = sig.active_programs
            # literature: overwrite with the deterministic one-liner from the typed EvidenceSummary (strength + direction +
            # design), the single source for the design word.
            es = lit.evidence_summary if lit else None
            if es is not None:
                fields["literature"] = _literature_oneliner(es)
            # Stash the RESOLVED facts for the interpretive enrich pass. The interpretive fields (blocker/key_risk/
            # verdict/prose) are authored only by the isolated judge_interpretive call fed these facts. Unresolved queries
            # get explicit unavailable-search facts; other entries without a stage phrase keep the LLM blurb text.
            approved_ind = (
                ct.approval.matched_indication
                if (ct is not None and ct.approval is not None)
                else None
            )
            closure_text = _closure_text(ct, es)
            interp_facts = (
                {
                    "stage": stage_phrase,
                    "active_programs": fields["active_programs"],
                    "literature": fields["literature"],
                    # Authoritative upstream FDA label relationship (NOT the LLM blurb value).
                    "relationship": approval_labels.get(
                        disease.lower().strip(), "none"
                    ),
                    "approved_indication": approved_ind,
                    "trial_evidence": _trial_evidence_text(coverage),
                    # Closure and late-stage terminations are decided upstream but weren't reaching the writer, which then
                    # judged closure itself and read the stage alone (a completed Phase 3 outranks the terminated-for-cause
                    # tier, hiding the stops).
                    "closure": closure_text,
                    "terminations": _terminations_text(sig),
                }
                if stage_phrase is not None
                else None
            )
            entry = {
                "disease": disease,
                "prose": prose,
                "_interp_facts": interp_facts,
                # The same closure fact the writer sees; the "Closed signals:" footer is rebuilt from it so it cannot
                # disagree with a card's "Closed signal" assessment.
                "_closed": closure_text.startswith("CLOSED"),
                **fields,
            }
            validated.append(entry)
            seen_validated.add(disease_key)

        # Enrich pass: judge_interpretive is the single author of blocker/key_risk/verdict/prose, fed the resolved
        # stage/active_programs/literature/approval so it cannot contradict them. Entries without resolved facts
        # (_interp_facts is None) keep their LLM blurb text.
        _to_interp = [e for e in validated if e.get("_interp_facts")]
        if _to_interp:
            # drug name is for cache-readability/logging only (not part of the cache identity); a run is for one drug, so
            # take the single drug_facts key.
            _drug = next(iter(drug_facts), "")
            _results = await asyncio.gather(
                *(
                    judge_interpretive(
                        **e["_interp_facts"],
                        cache_dir=svc.cache_dir,
                        drug=_drug,
                        indication=e["disease"],
                    )
                    for e in _to_interp
                )
            )
            for e, j in zip(_to_interp, _results, strict=True):
                if j is not None:
                    e["blocker"] = j.blocker
                    e["key_risk"] = j.key_risk
                    e["verdict"] = j.verdict
                    e["prose"] = j.prose
        closed_diseases = [e["disease"] for e in validated if e.get("_closed")]
        for e in validated:
            e.pop("_interp_facts", None)
            e.pop("_closed", None)

        # Drop LLM summary ranked lines whose disease didn't pass the evidence gate (not in validated); non-ranked lines
        # pass through and survivors are renumbered. Uses the same regex and longest-substring match as the report
        # formatter's _splice_blurbs_into_summary.
        validated_diseases = {e["disease"].lower().strip() for e in validated}

        # Authoritative dev_stage phrase per disease from every investigated candidate (demoted ones appear only in the
        # footer and need correcting there too), used to overwrite a false stage clause in a demotion footer line.
        dev_stage_by_disease: dict[str, str] = {}
        unresolved_trial_queries: set[str] = set()
        for _dkey, _slot in findings_local.items():
            _ct = (_slot or {}).get("clinical_trials")
            if _trial_query_unresolved(_ct):
                unresolved_trial_queries.add(_dkey)
            _phrase = _dev_stage_phrase(_ct.signals if _ct else None)
            if _phrase is not None:
                dev_stage_by_disease[_dkey] = _phrase

        # Reuses the module-level _FALSE_STAGE_FRAGMENT / _PROGRAM_STAGES so footer, ranked line and blurb share one rule.
        footer_line = re.compile(r"^\s*-\s+(?P<disease>.+?)\s+—\s+.+$")

        def _repair_stage_clause(disease_text: str, line: str, context: str) -> str:
            """Replace a false 'no program / Phase 4 only' stage clause in `line` with the authoritative dev_stage phrase,
            when the matched disease's dev_stage asserts a real Phase 3 program. Used for both ranked lines and footers.
            """
            head_lower = disease_text.lower()
            # Match the disease by longest containing key (same strategy as ranked lines).
            matched_key = next(
                (
                    d
                    for d in sorted(dev_stage_by_disease, key=len, reverse=True)
                    if d and d in head_lower
                ),
                None,
            )
            if matched_key is None:
                return line
            _ct = (findings_local.get(matched_key) or {}).get("clinical_trials")
            _sig = _ct.signals if _ct else None
            if _sig is None or _sig.dev_stage not in _PROGRAM_STAGES:
                return line
            if not _FALSE_STAGE_FRAGMENT.search(line):
                return line
            repaired = _FALSE_STAGE_FRAGMENT.sub(
                dev_stage_by_disease[matched_key], line, count=1
            )
            logger.warning(
                "[TOOL] finalize_supervisor repaired false stage in %s for "
                "disease=%r (dev_stage=%s); was: %r",
                context,
                disease_text,
                _sig.dev_stage,
                line,
            )
            return repaired

        def _repair_footer_stage(line: str) -> str:
            """Correct a false stage clause in a demotion footer line ('- <disease> — ...')."""
            fm = footer_line.match(line)
            if fm is None:
                return line
            disease_text = fm.group("disease")
            disease_lower = disease_text.lower()
            unresolved_key = next(
                (
                    key
                    for key in sorted(unresolved_trial_queries, key=len, reverse=True)
                    if key and key in disease_lower
                ),
                None,
            )
            if unresolved_key is not None:
                return (
                    f"- {disease_text} — {_UNRESOLVED_TRIAL_STAGE}; "
                    f"active programs {_UNRESOLVED_ACTIVE_PROGRAMS}"
                )
            return _repair_stage_clause(disease_text, line, "demotion footer")

        # A ranked summary line: "N. <disease> — <tail>" or tail-less "N. <disease>". The tail is optional because the
        # formatter discards it, and a tail-less line must still be recognized so it isn't leaked into the footer.
        rank_line = re.compile(
            r"^\s*(?P<rank>\d+)\.\s+(?P<head>.+?)(?:\s+—\s+(?P<tail>.+))?$"
        )
        # Footer lines rebuilt below from finalize's own decisions; the LLM's versions are discarded.
        rebuilt_footer_line = re.compile(
            r"^\s*(?:Evidence\s+gate\s+exclusions|Closed\s+signals|Not\s+ranked)\s*:",
            re.IGNORECASE,
        )
        # Normalize the heading to "Candidates assessed for <drug>:"; the legacy "Ranked repurposing <noun>" wording is
        # still matched so a model that reverts to it is rewritten.
        heading_line = re.compile(
            r"^\s*(?:Ranked\s+repurposing\s+"
            r"(?:candidates|opportunities|indications|signals)"
            r"|Candidates\s+assessed)"
            r"(?P<rest>\s+for\s+.+?:?\s*)$",
            re.IGNORECASE,
        )

        # RANK ORDER = `validated` (the critic-reviewed blurbs), never the LLM's line order. The LLM's ranked lines supply
        # only per-disease tail text; the ranked block is rebuilt by iterating `validated`, so numbering agrees with
        # top_diseases/blurbs. Heading and non-ranked footer lines keep their position; the block goes where the LLM's
        # first ranked line sat.
        def _match_validated(head_lower: str) -> str | None:
            """The validated-disease key contained in `head_lower` (longest match), or None."""
            return next(
                (
                    d
                    for d in sorted(validated_diseases, key=len, reverse=True)
                    if d and d in head_lower
                ),
                None,
            )

        # First pass: collect each LLM ranked line's tail by matched validated disease, and record where the ranked block
        # begins so the rebuilt block goes back in the same place.
        tail_by_disease: dict[str, str] = {}
        ranked_block_index: int | None = None
        passthrough: list[str | None] = []  # None marks the ranked-block insertion slot
        for line in summary.splitlines():
            heading_match = heading_line.match(line)
            if heading_match is not None:
                rest = heading_match.group("rest").rstrip()
                if not rest.endswith(":"):
                    rest = rest + ":"
                passthrough.append(f"Candidates assessed{rest}")
                continue
            m = rank_line.match(line)
            if m is None:
                # The LLM's own "Evidence gate exclusions:" / "Closed signals:" / "Not ranked:" lines are discarded and
                # rebuilt below from finalize's decisions. Other lines (demotion footer entries) pass through with a false
                # stage clause corrected.
                if rebuilt_footer_line.match(line):
                    continue
                passthrough.append(_repair_footer_stage(line))
                continue
            matched = _match_validated(m.group("head").lower())
            if matched is None:
                logger.warning(
                    "[TOOL] finalize_supervisor dropping summary line for "
                    "disease=%r (failed evidence gate)",
                    m.group("head"),
                )
                continue
            tail_by_disease[matched] = m.group("tail") or ""
            if ranked_block_index is None:
                ranked_block_index = len(passthrough)
                passthrough.append(None)  # reserve the ranked-block slot

        # Build the ranked block in `validated` order, renumbered 1..N. A validated disease with no LLM line still appears
        # with an empty tail (the formatter splices the blurb regardless).
        ranked_lines: list[str] = []
        for entry in validated:
            disease_key = entry["disease"].lower().strip()
            tail = tail_by_disease.get(disease_key, "")
            head = entry["disease"]
            line = f"{len(ranked_lines) + 1}. {head}" + (f" — {tail}" if tail else "")
            line = _repair_stage_clause(head, line, "ranked summary line")
            ranked_lines.append(line)

        # If the LLM produced no ranked lines at all, the block still must appear — put it right after the heading.
        if ranked_block_index is None and ranked_lines:
            insert_at = 1 if passthrough else 0
            passthrough.insert(insert_at, None)
            ranked_block_index = insert_at

        filtered_lines: list[str] = []
        for passthrough_line in passthrough:
            if passthrough_line is None:
                filtered_lines.extend(ranked_lines)
            else:
                filtered_lines.append(passthrough_line)

        # Derived footers, each omitted when empty (the LLM's version could assert an exclusion or closure that never
        # happened). Order: closed signals, not ranked, gate exclusions.
        if closed_diseases:
            filtered_lines.append("Closed signals: " + ", ".join(closed_diseases))

        # Every disease that reached the findings stage and is neither ranked nor gate-excluded gets a "Not ranked:" line
        # with its reason, so an assessed candidate cannot vanish silently.
        investigated_keys = list(findings_local.keys()) + [
            k for k in auto_findings if k not in findings_local
        ]
        gate_excluded_keys = {d.lower().strip() for d, _ in gate_excluded}
        not_ranked: list[tuple[str, str]] = []
        for dkey in investigated_keys:
            if dkey in validated_diseases or dkey in gate_excluded_keys:
                continue
            canonical, _src = allowed_diseases.get(dkey, (dkey, "competitor"))
            # The gate above only sees blurbs the model submitted; a disease it left out gets the same zero-evidence check
            # here, so "Absent from ranking" is reserved for candidates that had evidence.
            unranked_gate_reason = _evidence_gate_reason(
                findings_local.get(dkey) or auto_findings.get(dkey) or {}
            )
            if unranked_gate_reason is not None:
                logger.warning(
                    "[TOOL] finalize_supervisor unranked disease=%r fails evidence gate: %s",
                    canonical,
                    unranked_gate_reason,
                )
                gate_excluded.append((canonical, unranked_gate_reason))
                continue
            if dkey in unresolved_trial_queries:
                reason = _NOT_RANKED_REASON_UNRESOLVED
            elif approval_labels.get(dkey) == "combination_only":
                reason = _NOT_RANKED_REASON_COMBINATION
            else:
                reason = _NOT_RANKED_REASON_ABSENT
            logger.warning(
                "[TOOL] finalize_supervisor footer 'Not ranked' disease=%r (%s)",
                canonical,
                reason,
            )
            not_ranked.append((canonical, reason))
        if not_ranked:
            filtered_lines.append(
                "Not ranked: "
                + "; ".join(f"{disease} — {reason}" for disease, reason in not_ranked)
            )

        if gate_excluded:
            filtered_lines.append(
                "Evidence gate exclusions: "
                + "; ".join(
                    f"{disease} — {reason}" for disease, reason in gate_excluded
                )
            )
        filtered_summary = "\n".join(filtered_lines)

        artifact = {"summary": filtered_summary, "blurbs": validated}
        return "Supervisor analysis complete.", artifact

    def get_merged_allowlist() -> (
        dict[str, tuple[str, Literal["competitor", "mechanism", "both"]]]
    ):
        """Snapshot the post-merge competitor + mechanism disease allowlist.

        Returns a copy keyed by lowercase disease name → (canonical_name, source), source being "competitor", "mechanism"
        or "both".
        """
        return dict(allowed_diseases)

    def get_auto_findings() -> dict[str, dict]:
        """Snapshot artifacts produced by investigate_top_candidates (fan-out only).

        Returns {lowercase_canonical_disease: {"literature": LiteratureOutput,
        "clinical_trials": ClinicalTrialsOutput}}. Empty when fan-out is off.
        """
        return dict(auto_findings)

    def get_approval_labels() -> dict[str, str]:
        """Snapshot upstream FDA approval-relationship labels for kept candidates.

        Returns {lowercase_disease: "contaminated" | "combination_only"}. Diseases with no relationship ("none") or that
        were dropped ("approved") are absent. Empty in holdout runs (no label data on the date_before path).
        """
        return dict(approval_labels)

    # supervisor_fanout has no default in Settings, so a missing SUPERVISOR_FANOUT fails loudly at startup. False
    # investigates per candidate via the ReAct loop; True fans out via investigate_top_candidates. Holdout runs use the
    # same flag as production.
    fanout = get_settings().supervisor_fanout
    tools = [
        find_candidates,
        analyze_mechanism,
        analyze_literature,
        analyze_clinical_trials,
        get_drug_briefing,
        critique_ranking,
        finalize_supervisor,
    ]
    if fanout:
        # Insert investigate_top_candidates before finalize so the LLM can call it after the seed-phase tools (enabled
        # by supervisor_fanout).
        tools.insert(-1, investigate_top_candidates)
        # Force the parallel path: the LLM ignores the prompt-level fan-out directive, so remove the per-candidate tools.
        tools = [
            t for t in tools if t not in (analyze_literature, analyze_clinical_trials)
        ]
    return tools, get_merged_allowlist, get_auto_findings, get_approval_labels
