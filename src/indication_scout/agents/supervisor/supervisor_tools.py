"""Supervisor tools — wraps sub-agents as tools.

Each sub-agent (literature, clinical trials) becomes one tool: it runs the full sub-agent and
returns its typed output as an artifact plus a short summary string for the LLM. Also a
find_candidates tool that hits Open Targets directly to surface disease candidates for a drug.
"""

import asyncio
import json
import logging
import re
import time
from datetime import date
from typing import Any, Literal

from langchain_core.tools import tool
from sqlalchemy.orm import Session, sessionmaker

from indication_scout.agents.supervisor.candidate_dedup import (
    run_hierarchical_dedup,
)
from indication_scout.config import get_settings
from indication_scout.constants import SUPERVISOR_MIN_PMIDS_NO_TRIALS
from indication_scout.data_sources.chembl import (
    ChEMBLClient,
    get_all_drug_names,
    resolve_drug_name,
)
from indication_scout.data_sources.fda import FDAClient
from indication_scout.data_sources.open_targets import OpenTargetsClient
from indication_scout.helpers.drug_helpers import normalize_drug_name
from indication_scout.services.approval_check import (
    get_approved_indications,
    get_fda_approved_disease_mapping,
    list_approved_indications_at,
    list_approved_indications_from_labels,
)
from indication_scout.services.dev_stage import DEV_STAGE_PHRASE, dev_stage_phrase
from indication_scout.services.judge_interpretive import judge_interpretive
from indication_scout.services.llm import query_llm, strip_markdown_fences
from indication_scout.services.progress import (
    PHASE_CANDIDATES,
    PHASE_MECHANISM,
    PHASE_SUMMARY,
    PHASE_TRIALS,
    emit_progress,
)

logger = logging.getLogger(__name__)

# Authoritative development-stage phrase + renderer live in services/dev_stage (the single
# home shared with the report formatter). Aliased to the local names used throughout.
_DEV_STAGE_PHRASE = DEV_STAGE_PHRASE
_dev_stage_phrase = dev_stage_phrase


def _literature_oneliner(es) -> str:
    """Deterministic one-line literature summary from the typed EvidenceSummary fields (strength, direction,
    study design) — NOT free LLM prose. Fed to judge_interpretive and used to overwrite the blurb's `literature`
    field, so the design word always matches the authoritative is_observational token. Returns "None" when no
    evidence summary is present.
    """
    if es is None:
        return "None"
    # class_level = the disease-relevant RCTs are for OTHER drugs in the class. Must NOT read "strong,
    # RCT-backed" — there is no drug-specific body. strength/direction are forced to "none" for class_level by
    # synthesize, so the ranking path also sees no drug strength.
    if getattr(es, "evidence_basis", "none") == "class_level":
        return "class-level signal (no direct evidence for this drug)"
    # approved = the only relevant this-drug evidence studies an APPROVED sub-indication (not repurposing).
    # strength/direction forced to "none" here too, so no repurposing strength.
    if getattr(es, "evidence_basis", "none") == "approved":
        return "evidence is for an already-approved sub-indication (not repurposing)"
    design = (
        "RCT-backed / controlled"
        if es.is_observational is False
        else "observational" if es.is_observational is True else "undetermined design"
    )
    direction = es.direction if es.direction and es.direction != "none" else ""
    parts = [es.strength or "none"]
    if direction:
        parts.append(direction)
    parts.append(design)
    return ", ".join(parts)


# False stage clause the LLM writes in a DEMOTION FOOTER / ranked-summary line when it ignores the authoritative
# dev_stage. Used ONLY there — those lines are controlled single-clause contexts where replacement is safe. NOT
# used on free-text blurb fields (verdict/blocker/prose), where mid-sentence substitution mangled grammar; those
# are kept phase-free by the prompt. Anchored to whole semicolon/paren-delimited clauses so a match spans a full
# clause.
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


# Fresh-context critic for the supervisor's ranked blurbs. Two jobs, both by REASONING (not a rule list): (1) sanity-check
# the ORDER — think about what each candidate's evidence actually means for whether it is a live repurposing opportunity, and
# if the order is not defensible, reorder it; (2) repair blurb fields that contradict the machine-derived per-candidate FACT.
_RANKING_CRITIC_SYSTEM = """\
You are a skeptical reviewer of a drug-repurposing candidate ranking. You are given the candidates \
in their current rank order, each with a verdict tag, a few one-line fields, and a per-candidate \
FACT block (machine-derived, authoritative).

FIRST, audit the ORDER. For each candidate, think about what its evidence actually MEANS for \
whether it is a live repurposing opportunity worth surfacing first — not merely how much or how \
impressive the evidence is. A candidate that has been tested and shown to FAIL, or whose program \
is closed, is a weaker opportunity than one that is still live, however thin. If the current order \
is not defensible on that basis, reorder the candidates so the strongest live opportunities come \
first. If it is already defensible, keep it.

SECOND, repair any blurb field that CONTRADICTS the candidate's FACT (the FACT is authoritative; \
it states whether a relevant COMPLETED or ACTIVE Phase 3 is on record, or none). Do not let a \
field claim there are no Phase 3 trials when the FACT says one exists. Leave true statements about \
the absence of a commercial/NDA/regulatory PROGRAM alone (a generic drug files no new NDA — that \
is not the same as "no trial"). Change nothing else.

Output a JSON object ONLY (no prose, no fences):
{"blurbs": [ <every input blurb, in your final rank order, each a full dict with the same keys; \
fields you repaired are rewritten, all others verbatim> ]}
Return every blurb. Preserve every key."""


def _log_disease_banner(title: str, diseases: list[str]) -> None:
    """Emit a boxed WARNING-level banner listing diseases, one per line.

    Makes pivotal candidate-list transitions easy to spot in run logs: FDA-dropped, final allowlist, mechanism-promoted,
    and top-N investigation set.
    """
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


from indication_scout.agents._trial_formatting import (
    _borda_rank_by_enrollment_and_recency,
    _format_trial_table,
    _phase_distribution,
)
from indication_scout.agents._trial_signals import (
    derive_trial_signals,
    format_derived_signals,
)
from indication_scout.agents.clinical_trials.clinical_trials_agent import (
    build_clinical_trials_agent,
    run_clinical_trials_agent,
)
from indication_scout.agents.clinical_trials.clinical_trials_output import (
    ClinicalTrialsOutput,
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
from indication_scout.services.retrieval import RetrievalService


def build_supervisor_tools(
    llm,
    svc: RetrievalService,
    db: Session,
    session_factory: "sessionmaker | None" = None,
    date_before: date | None = None,
) -> tuple[list, "callable", "callable"]:
    """Build supervisor tools that close over the sub-agents.

    The literature and clinical trials agents are compiled once here and reused across calls.

    `session_factory` is a shared `sessionmaker` (one engine/pool) used by the concurrent investigate_top_candidates fan-out:
    each analyze_literature call checks out its own Session from it, because a SQLAlchemy Session is not safe for concurrent
    use. When None (single-threaded callers, tests), a factory bound to the run `db`'s existing engine is derived — reusing
    that engine, never creating a new one (which would leak a pool).

    `date_before` is forwarded to the literature and clinical trials sub-agents so all PubMed and ClinicalTrials.gov queries
    share the same temporal cutoff. The mechanism sub-agent doesn't accept it (OpenTargets has no date-filtering API).

    Returns (tools, get_merged_allowlist) where get_merged_allowlist() snapshots the post-merge competitor + mechanism
    disease allowlist (lowercase name → (canonical_name, source)), intended to be read after the agent loop has finished.
    """
    # Reuse the run db's engine when no shared factory was passed (single-threaded callers, tests). Binding to db.get_bind()
    # means no new engine/pool is created.
    if session_factory is None:
        session_factory = sessionmaker(
            autocommit=False, autoflush=False, bind=db.get_bind()
        )

    # Build the mechanism sub-agent once at supervisor construction (it runs once in the seed phase, not in the concurrent
    # per-candidate fan-out). The clinical-trials agent is built PER analyze_clinical_trials call instead — its tools carry a
    # closure-scoped shown_by_pair set (the per-trial relevance gate), and investigate_top_candidates fans candidates out
    # concurrently (asyncio.gather), so a single shared instance would accumulate every pair's trials across candidates and
    # race on that mutable state. A fresh build per call isolates the closure state (graph compile only, no I/O — cheap
    # relative to the network/LLM work).
    mech_agent = build_mechanism_agent(llm=llm, date_before=date_before)

    # Closure-scoped allowlist — populated by find_candidates and analyze_mechanism, checked by analyze_literature /
    # analyze_clinical_trials.
    # allowed_diseases: lowercase disease name → (canonical_name, source)
    allowed_diseases: dict[
        str, tuple[str, Literal["competitor", "mechanism", "both"]]
    ] = {}
    # EFO ID → lowercase disease name (key into allowed_diseases). Lets the merge step dedup mechanism candidates against
    # competitor entries by ontology ID even when names differ (e.g. "NSCLC" vs "non-small cell lung cancer").
    allowed_efo_ids: dict[str, str] = {}
    # Raw mechanism candidates as they arrive. analyze_mechanism appends here; find_candidates consumes them in
    # merge_and_dedup() after both seed tools finish. Holding the full list lets the hierarchical LLM pass see the complete
    # competitor + mechanism union in one shot.
    mechanism_candidates_buffer: list = []
    # Drug-level mechanism target list, set by analyze_mechanism and read by merge_and_dedup so the hierarchical LLM pass can
    # reason about the drug's MoA when picking survivors.
    mechanism_targets_for_dedup: list[tuple[str, str]] = []
    # Seed-phase gates. find_candidates and analyze_mechanism run in parallel. analyze_mechanism only buffers raw candidates
    # and sets analyze_mechanism_done; it doesn't touch the allowlist. find_candidates seeds the competitor allowlist, awaits
    # analyze_mechanism_done, then runs merge_and_dedup (centralized exact-match + hierarchical-LLM dedup over the union).
    # find_candidates_done is set inside merge_and_dedup so downstream tools (analyze_literature, analyze_clinical_trials,
    # investigate_top_candidates) only observe the post-dedup allowlist. Both events are set in try/finally so a sub-agent
    # crash doesn't deadlock downstream tools.
    find_candidates_done = asyncio.Event()
    analyze_mechanism_done = asyncio.Event()

    # Drug-level shared store. Populated by sub-agents as they run; surfaced to the supervisor via get_drug_briefing. Keyed
    # by normalized drug name. See supervisor_ideas.md for rationale.
    drug_facts: dict[str, dict] = {}

    # Fan-out only: artifacts produced by investigate_top_candidates. The tool invokes analyze_literature/
    # analyze_clinical_trials directly (not through the LangGraph ReAct loop), so their tool messages don't reach
    # result["messages"]. We stash them here and run_supervisor_agent reads them via get_auto_findings() after the run
    # completes. Keyed by lowercase canonical disease name → {"literature": ..., "clinical_trials": ...}.
    auto_findings: dict[str, dict] = {}

    # Upstream FDA approval-relationship labels for kept candidates, keyed by lowercase disease name → "contaminated" |
    # "combination_only". "approved" diseases are dropped (never recorded here); "none" diseases are omitted (no
    # relationship). run_supervisor_agent reads this via get_approval_labels() to set CandidateFindings.approval_relationship
    # from the label-grounded source, NOT from LLM prose.
    approval_labels: dict[str, str] = {}

    # Per-disease sub-agent artifacts written by analyze_literature and analyze_clinical_trials as the supervisor runs. Used
    # by finalize_supervisor to enforce the top-N evidence gate (drop blurbs for candidates failing the (0 trials AND <N
    # PMIDs) check). Keyed by lowercase canonical disease name → {"literature": ... | None, "clinical_trials": ... | None}.
    findings_local: dict[str, dict] = {}

    # Ordering gate: finalize_supervisor is rejected until critique_ranking has run this run, so the ranking is always
    # audited before it is committed. The LLM ignores the prompt-level "MANDATORY" instruction on its own, so this enforces
    # it in code.
    critique_state: dict[str, bool] = {"ran": False}

    def _drug_key(drug_name: str) -> str:
        return drug_name.lower().strip()

    def _ensure_drug_entry(drug_name: str) -> dict:
        key = _drug_key(drug_name)
        if key not in drug_facts:
            drug_facts[key] = {
                "drug_name": key,
                "drug_aliases": [],  # ChEMBL trade/generic names
                "approved_indications": [],  # list of indication strings
                "mechanism_targets": [],  # list of (gene, action_type)
                "mechanism_disease_associations": [],  # list of (gene, disease, score)
                "first_approval": None,  # year first approved anywhere (ChEMBL), or None
            }
        return drug_facts[key]

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
                # Hide score when it's the placeholder (not surfaced by
                # MechanismCandidate). Show otherwise.
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
            # Always release the seed gate so a failure here doesn't deadlock analyze_literature / analyze_clinical_trials.
            # They will see an empty allowlist and reject downstream.
            find_candidates_done.set()

    async def _find_candidates_impl(drug_name: str) -> tuple[str, list[str]]:
        drug_name = normalize_drug_name(drug_name)
        chembl_id = await resolve_drug_name(drug_name, svc.cache_dir)
        competitors = await svc.get_drug_competitors(chembl_id, date_before=date_before)
        diseases = list(competitors.keys())

        # Drug-level intake: populate the shared store with aliases and any
        # FDA-approved indications discovered during the candidate filter.
        entry = _ensure_drug_entry(drug_name)
        try:
            entry["drug_aliases"] = await get_all_drug_names(chembl_id, svc.cache_dir)
        except Exception as e:
            logger.warning(
                "find_candidates: get_all_drug_names failed for %s: %s", chembl_id, e
            )

        # Fetch first_approval (year first approved anywhere) so the clinical_trials sub-agent's closure judgment can tell an
        # old/generic drug (no new NDA expected) from a genuine negative. Stays None on failure — never defaulted to a year
        # (CLAUDE.md no-fallback).
        try:
            async with ChEMBLClient(cache_dir=svc.cache_dir) as chembl_client:
                molecule = await chembl_client.get_molecule(chembl_id)
            entry["first_approval"] = molecule.first_approval
        except Exception as e:
            logger.warning(
                "find_candidates: get_molecule(first_approval) failed for %s: %s",
                chembl_id,
                e,
            )

        # Seed approved_indications from the drug's own FDA label, independent of any candidate list. Without this, an
        # approved indication absent from OpenTargets competitor diseases (e.g. semaglutide × MASH) never reaches the
        # briefing and the supervisor can't reason about subset/superset relationships against it.
        #
        # When date_before is set, swap the live openFDA path for the hardcoded approvals table — the live path leaks today's
        # approvals past the cutoff. Drugs not in the table return [] and approval reasoning is silently disabled for that
        # cutoff run.
        seed_aliases = entry["drug_aliases"] or [drug_name]
        try:
            if date_before is not None:
                seeded = list_approved_indications_at(drug_name, date_before)
            else:
                async with FDAClient(cache_dir=svc.cache_dir) as fda_client:
                    label_texts = await fda_client.get_all_label_indications(
                        seed_aliases
                    )
                seeded = await list_approved_indications_from_labels(
                    label_texts=label_texts,
                    cache_dir=svc.cache_dir,
                )
            if seeded:
                existing = {
                    ind.lower().strip() for ind in entry["approved_indications"]
                }
                for ind in seeded:
                    if ind.lower().strip() not in existing:
                        entry["approved_indications"].append(ind)
                # logger.warning(
                #     "[TOOL] find_candidates seeded %d approved indication(s) from %s: %s",
                #     len(seeded),
                #     "hardcoded table" if date_before is not None else "label",
                #     seeded,
                # )
        except Exception as e:
            logger.warning(
                "find_candidates: approved-indication seed failed for %s: %s",
                drug_name,
                e,
            )

        # Drop competitor diseases already approved for this drug. Same swap as above: hardcoded table when date_before is
        # set, live FDA otherwise.
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
                    cache_dir=svc.cache_dir,
                )
                fda_approved = {
                    disease for disease, label in mapping.items() if label == "approved"
                }
                # Record the non-"approved" relationship labels for KEPT candidates so the report can render them from the
                # label-grounded source (not LLM prose). "none" carries no relationship → omitted. Holdout (date_before) path
                # has no label data, so labels are only captured here in the live path.
                for disease, label in mapping.items():
                    if label in ("contaminated", "combination_only"):
                        approval_labels[disease.lower().strip()] = label
            if fda_approved:
                _log_disease_banner(
                    f"FDA-DROPPED (already approved for {drug_name}, source: "
                    f"{'hardcoded table' if date_before is not None else 'live FDA'})",
                    sorted(fda_approved),
                )
                # Record the approved indications in the shared store. Discovered as a side effect of candidate filtering —
                # even though dropped from the candidate list, the supervisor needs them to reason about subset/superset
                # relationships (e.g. CML approval makes "myeloid leukemia" candidate ambiguous).
                existing = {
                    ind.lower().strip() for ind in entry["approved_indications"]
                }
                for ind in fda_approved:
                    if ind.lower().strip() not in existing:
                        entry["approved_indications"].append(ind)
            fda_approved_lower = {d.lower().strip() for d in fda_approved}

        diseases = [d for d in diseases if d.lower().strip() not in fda_approved_lower]

        # Cap applies to competitor entries only. Mechanism-promoted entries are appended on top after the merge below —
        # they're already small (capped upstream by settings.mechanism_top_candidates) and dropping them here would defeat
        # the purpose of the merge.
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

        # Pull EFO IDs for the competitor allowlist from the raw OT cache. Used to dedup mechanism candidates against
        # competitor entries by ontology ID. Names that don't resolve to an EFO (e.g. renamed by the LLM merge step) get no
        # entry — dedup falls back to name match.
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

        # Wait for analyze_mechanism to populate mechanism_candidates_buffer. The mechanism sub-agent's finally block always
        # sets this gate, so a crash just yields an empty mechanism contribution rather than deadlocking.
        await analyze_mechanism_done.wait()

        # Run the centralized merge + dedup pipeline. find_candidates_done is set inside merge_and_dedup's finally so
        # downstream readers (analyze_literature, analyze_clinical_trials, investigate_top_candidates) only see the
        # post-dedup allowlist.
        await merge_and_dedup(drug_name)

        # Snapshot the post-dedup allowlist in dict insertion order: competitor entries first (OT ranked order), then
        # mechanism-only entries (analyze_mechanism's order); hierarchical dedup may have removed entries from either.
        # "both" entries stay in their competitor slot.
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

        Pipeline (single chokepoint for all candidate-disease deduplication):
          1. Exact ID match — drop mechanism candidate if disease_id already in allowed_efo_ids; upgrade matched competitor
             entry's source to "both".
          2. Exact name match — drop if lowercased name already in allowed_diseases; upgrade to "both".
          3. OT name-resolve — resolve unresolved mechanism candidate names to EFO IDs; retry step 1 against allowed_efo_ids.
          4. Hierarchical LLM pass — over the full merged list, identify super/subtype overlaps the exact-match passes can't
             catch (UC ⊂ IBD, T2DM ⊂ DM); pick one survivor each.

        Sets find_candidates_done before returning so downstream readers (analyze_literature, analyze_clinical_trials,
        investigate_top_candidates) observe the post-dedup allowlist. The find_candidates wrapper's try/finally still sets
        the gate on any exception path.
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

        # Step 4: hierarchical LLM pass — temporarily disabled while we revisit the case that motivated it. The dedup was
        # collapsing actionable subtype candidates (e.g. PCOS, gestational diabetes) into broad parents (metabolic disease)
        # for broadly-acting drugs like metformin. Keep all candidates from the exact-match dedup until we decide on a
        # hardcoded equivalence-group approach.
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

    def _reject(disease_name: str, tool_label: str, empty_output):
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
        # Wait for both seed tools to finish populating the allowlist. Without this, parallel tool calls can hit
        # analyze_literature before find_candidates / analyze_mechanism have merged their candidates, causing legitimate
        # diseases to be rejected.
        await find_candidates_done.wait()
        await analyze_mechanism_done.wait()

        drug_name = normalize_drug_name(drug_name)
        # Build a fresh agent per call so the closure-scoped store dict in literature_tools is not shared across disease
        # invocations.
        if disease_name.lower().strip() not in allowed_diseases:
            return _reject(disease_name, "analyze_literature", LiteratureOutput())

        # logger.warning("[TOOL] analyze_literature(drug=%r, disease=%r)", drug_name, disease_name)

        logger.warning(
            "[TOOL] analyze_literature(drug=%r, disease=%r)", drug_name, disease_name
        )

        # Per-call DB session from the shared pool. investigate_top_candidates fans candidates out concurrently
        # (asyncio.gather), and a SQLAlchemy Session is not safe for concurrent use — sharing one across the fan-out parks a
        # connection in an open transaction nothing advances (flat CPU / idle Postgres hang). Each call checks out its own
        # Session and closes it here.
        # Read the drug's approved-indication list as a LITERAL here (not a deferred closure): find_candidates has already
        # seeded it before any fan-out, so it is complete. Threaded into the literature agent so the combined synthesize call
        # excludes papers about an approved sub-indication of this (possibly broad) candidate.
        approved_indications = list(
            _ensure_drug_entry(drug_name)["approved_indications"]
        )
        _t0 = time.perf_counter()
        with session_factory() as call_db:
            lit_agent = build_literature_agent(
                llm=llm,
                svc=svc,
                db=call_db,
                date_before=date_before,
                approved_indications=approved_indications,
            )
            output = await run_literature_agent(lit_agent, drug_name, disease_name)
        logger.warning(
            "[TIMING] literature %s: %.1fs", disease_name, time.perf_counter() - _t0
        )
        strength = (
            output.evidence_summary.strength if output.evidence_summary else "no data"
        )
        direction = (
            output.evidence_summary.direction if output.evidence_summary else "no data"
        )
        # is_observational is an authoritative design fact from the literature agent: True = purely observational, False =
        # RCT-backed, None = undetermined. Surface it verbatim so the blurb does not infer "observational" from prose (it
        # must NOT call RCT-backed evidence observational).
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
        # evidence_basis tells the supervisor WHY a candidate may have strength=none: drug_specific (real repurposing
        # evidence), class_level (other drugs in the class), approved (an approved sub-indication), or none (no relevant
        # abstracts retrieved). The ranking judgment weighs class_level/approved differently from a genuine no-abstracts-yet
        # pair, so surface it.
        basis = (
            output.evidence_summary.evidence_basis
            if output.evidence_summary
            else "none"
        )
        header = (
            f"Literature for {drug_name} × {disease_name}: "
            f"{len(output.pmids)} PMIDs, strength={strength}, direction={direction}, "
            f"study_design={design}, evidence_basis={basis}."
        )
        # Build supervisor-facing summary deterministically from EvidenceSummary so adverse-signal language reaches the
        # supervisor verbatim with no LLM rewrite in between.
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
        # first_approval (seeded in find_candidates) lets the sub-agent's closure judgment tell an old/generic drug from a
        # genuine negative. None when not resolved — passed through as-is.
        seed_entry = drug_facts.get(_drug_key(drug_name))
        first_approval = seed_entry.get("first_approval") if seed_entry else None
        # Fresh agent per call — isolates the tools' closure-scoped shown_by_pair so concurrent candidate investigations
        # don't accumulate each other's trials into this pair's contaminated set (see the build-site note above).
        ct_agent = build_clinical_trials_agent(
            llm=llm, date_before=date_before, assigned_indication=disease_name
        )
        # Approved indications fully seeded by find_candidates before fan-out (label-grounded; see
        # PLAN_approval_aware_relevance.md §A). Threaded into the task so the relevance gate's TEST 1 treats an approved
        # sub-indication's trial as contamination, not roll-up evidence.
        ct_approved = list(_ensure_drug_entry(drug_name)["approved_indications"])
        output = await run_clinical_trials_agent(
            ct_agent,
            drug_name,
            disease_name,
            first_approval=first_approval,
            approved_indications=ct_approved,
        )
        logger.warning(
            "[TIMING] clinical_trials %s: %.1fs",
            disease_name,
            time.perf_counter() - _t0,
        )

        # Drug-level write-through: when the FDA check matches the candidate against an approved indication, capture it in
        # the supervisor's briefing so subsequent reasoning sees the approval status. Trial data still flows through to the
        # summary below — the sub-agent always investigates fully now (no short-circuits).
        approval = output.approval
        if approval is not None and approval.is_approved:
            entry = _ensure_drug_entry(drug_name)
            matched = approval.matched_indication or disease_name
            existing = {ind.lower().strip() for ind in entry["approved_indications"]}
            if matched.lower().strip() not in existing:
                entry["approved_indications"].append(matched)

        # Normal path: counts come from the new exact-count tools (countTotal API). Each scope owns its own count; no
        # cross-scope summing.
        search = output.search
        completed = output.completed
        terminated = output.terminated

        n_total = search.total_count if search else 0
        n_recruiting = search.by_status.get("RECRUITING", 0) if search else 0
        n_active_not_recruiting = (
            search.by_status.get("ACTIVE_NOT_RECRUITING", 0) if search else 0
        )
        n_withdrawn = search.by_status.get("WITHDRAWN", 0) if search else 0
        n_completed = completed.total_count if completed else 0
        n_terminated = terminated.total_count if terminated else 0
        # safety/efficacy classification is computed from the top-50 shown terminated trials; if total_count > len(trials)
        # this is a floor.
        n_safety_efficacy_shown = (
            sum(
                1
                for t in terminated.trials
                if _classify_stop_reason(t.why_stopped) in {"safety", "efficacy"}
            )
            if terminated
            else 0
        )
        header = (
            f"Clinical trials for {drug_name} × {disease_name}: "
            f"{n_total} total ({n_recruiting} recruiting, "
            f"{n_active_not_recruiting} active, {n_withdrawn} withdrawn). "
            f"{n_completed} completed. "
            f"{n_terminated} terminated "
            f"({n_safety_efficacy_shown} safety/efficacy in shown set)."
        )

        completed_trials = completed.trials if completed else []
        terminated_trials = terminated.trials if terminated else []

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

        # Authoritative reasoning basis: the sub-agent's relevance-filtered signals + its closure verdict (carried in the
        # prose). The supervisor reasons over THESE, not the raw counts above and not by re-deriving phase/closure from
        # prose. When the sub-agent didn't classify relevance (signals None), fall back to all-trial facts so the phase is
        # still surfaced rather than silently dropped.
        signals = output.signals or derive_trial_signals(output)
        signals_block = format_derived_signals(signals)
        if output.contaminated_nct_ids:
            signals_block += (
                f"\n  contaminated_excluded: {len(output.contaminated_nct_ids)} trial(s) "
                f"({', '.join(output.contaminated_nct_ids)})"
            )

        # Closure is a TYPED field from the post-loop synthesis call (no longer parsed from prose). The supervisor TRUSTS it
        # and must not re-judge closure. The prose summary is HUMAN-REPORT context only.
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
        _t0 = time.perf_counter()
        output = await run_mechanism_agent(
            mech_agent, drug_name, date_before=date_before
        )
        logger.warning("[TOOL] analyze_mechanism(drug=%r)", drug_name)
        logger.warning("[TIMING] analyze_mechanism: %.1fs", time.perf_counter() - _t0)

        # Buffer raw mechanism candidates for find_candidates to consume in merge_and_dedup() after both seed tools finish.
        # Centralizing the merge lets the full competitor + mechanism union pass through a single hierarchical-dedup pass.
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

        # Drug-level write-through: populate mechanism_targets and mechanism_disease_associations in the shared store.
        # Captured per-MoA so the briefing can show "ABL1 (INHIBITOR), KIT (INHIBITOR)".
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

        # Mechanism candidates already carry the high-score target→disease associations the agent surfaced. We don't have
        # the raw scores on the candidate model — record the pair without a score for now.
        assocs: list[tuple[str, str, float]] = []
        seen_assoc_pairs: set[tuple[str, str]] = set()
        for cand in output.candidates:
            pair_key = (cand.target_symbol, cand.disease_name)
            if pair_key in seen_assoc_pairs:
                continue
            seen_assoc_pairs.add(pair_key)
            # Score not surfaced on MechanismCandidate; use 0.0 as a placeholder. The supervisor only needs to know "this
            # gene is associated with this disease per OT mechanism evidence" — the briefing renderer hides the score if it's
            # the placeholder.
            assocs.append((cand.target_symbol, cand.disease_name, 0.0))
        entry["mechanism_disease_associations"] = assocs

        # Record the mechanism target list so merge_and_dedup() can pass it to the hierarchical LLM pass as context for
        # survivor selection.
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

    # Fan-out tool: bulk-investigate the top-N candidates with no LLM discretion. The probe (scripts/probe_supervisor_t2dm.py)
    # showed the supervisor LLM systematically skips "obvious" candidates like T2DM for semaglutide regardless of prompt
    # instructions, so we remove the LLM's ability to skip by auto-investigating the top-N
    # (settings.supervisor_investigation_cap). Env-tunable via .env.constants (SUPERVISOR_INVESTIGATION_CAP) so a validation
    # run can widen coverage.
    investigation_cap = get_settings().supervisor_investigation_cap

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

        # Top-N from the merged allowlist. Insertion order preserves find_candidates's competitor ranking, with
        # mechanism-promoted entries appended in analyze_mechanism's order. supervisor_candidate_cap is NOT used here — it
        # only trims the final ranked list, not how many diseases get investigated.
        top_n = list(allowed_diseases.items())[:investigation_cap]
        if not top_n:
            return "No candidates in allowlist; nothing to investigate.", []

        canonical_diseases = [canonical for _, (canonical, _) in top_n]
        _log_disease_banner(
            f"INVESTIGATING top candidates for {drug_name} (lit + trials in parallel)",
            canonical_diseases,
        )

        # Fan out: analyze_literature + analyze_clinical_trials in parallel. Pass a ToolCall-shaped dict (not a plain args
        # dict) so .ainvoke() returns a ToolMessage with .artifact populated. A plain dict input returns just the content
        # string and loses the typed artifact.
        async def _invest(disease: str) -> tuple[str, dict]:
            disease_slug = disease.lower().replace(" ", "_")
            logger.warning("[INVEST] starting %s", disease)
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
                logger.warning(
                    "[TIMING] investigate %s: lit leg took %.1fs",
                    disease,
                    time.perf_counter() - _lt0,
                )
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
                logger.warning(
                    "[TIMING] investigate %s: trials leg took %.1fs",
                    disease,
                    time.perf_counter() - _ct0,
                )
                return msg

            lit_msg, ct_msg = await asyncio.gather(_timed_lit(), _timed_ct())
            logger.warning(
                "[TIMING] investigate %s: lit+trials took %.1fs",
                disease,
                time.perf_counter() - _t0,
            )

            lit_artifact = lit_msg.artifact
            ct_artifact = ct_msg.artifact
            # Stash artifacts in the closure so run_supervisor_agent can merge them into the SupervisorOutput. The LangGraph
            # ReAct loop doesn't see these tool messages because they were invoked directly, not through the agent.
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
            n_total = (
                ct_artifact.search.total_count
                if ct_artifact and ct_artifact.search
                else 0
            )
            n_completed = (
                ct_artifact.completed.total_count
                if ct_artifact and ct_artifact.completed
                else 0
            )
            n_terminated = (
                ct_artifact.terminated.total_count
                if ct_artifact and ct_artifact.terminated
                else 0
            )
            # Relevance-filtered signals (sub-agent judgment) — the phase facts the supervisor ranks on, not the raw counts.
            # None when the sub-agent didn't classify; fall back to all-trial facts so the phase is still surfaced.
            ct_signals = (
                (ct_artifact.signals or derive_trial_signals(ct_artifact))
                if ct_artifact
                else None
            )
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
                "trials_total": n_total,
                "trials_completed": n_completed,
                "trials_terminated": n_terminated,
                "relevant_highest_phase": relevant_highest_phase,
                "relevant_phase3_terminated_for_cause": relevant_phase3_terminated,
                # Authoritative development-stage tier — seed the LLM's ranking with the same fact the downstream
                # blurb/footer/ranked-line render, so it isn't biased toward "Phase 4 only / no program" by a low
                # highest_COMPLETED_phase that ignores active/recruiting pivotal trials (the T1DM active-Phase-3 case).
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
        logger.warning(
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
            # Authoritative dev_stage seeds the ranking; render it so the LLM ranks on the same fact the report will show,
            # not on highest_completed_phase alone.
            stage_note = (
                f"; dev_stage {a['dev_stage']} ({a['dev_stage_phrase']})"
                if a.get("dev_stage_phrase")
                else ""
            )
            lines.append(
                f"  - {a['disease']}: literature {a['literature_strength']}{dir_note}, "
                f"{a['literature_pmids']} PMIDs; trials {a['trials_total']} total, "
                f"{a['trials_completed']} completed, {a['trials_terminated']} terminated; "
                f"relevant highest phase {phase}{term_note}{stage_note}"
            )
        return "\n".join(lines), artifacts

    async def _run_fact_critic(items: list[dict]) -> list[dict]:
        """Run the critic over `items`; return the audited blurbs (possibly REORDERED).

        The critic does two things by reasoning: (1) sanity-checks the rank ORDER — a tested-and-failed / closed candidate
        is a weaker opportunity than a live one and should not lead — and reorders if the order is not defensible; (2)
        repairs a blurb field that contradicts the authoritative per-disease Phase-3 FACT while preserving true "no
        regulatory program" claims. Returns the critic's blurb list (same disease SET, order may differ). On ANY parse
        failure or disease-set mismatch, returns the ORIGINAL items (data-loss guard). Used by both critique_ranking and
        finalize_supervisor.
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
            if stage_phrase is not None:
                fact = (
                    f"authoritative dev_stage = {sig.dev_stage} ({stage_phrase}) — "
                    "do not contradict this stage"
                )
            else:
                fact = "no trial signal available"
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
        logger.warning(
            "[TIMING] fact_critic LLM call: %.1fs", time.perf_counter() - _t0
        )
        logger.info(
            "[TOOL] fact_critic IN (%d candidates):\n%s", len(items), "\n".join(lines)
        )
        repaired = items
        try:
            data = json.loads(strip_markdown_fences(critique.strip()))
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
                logger.warning(
                    "[TOOL] fact_critic: blurb set mismatch — keeping originals"
                )
        except (json.JSONDecodeError, AttributeError, TypeError) as e:
            logger.warning(
                "[TOOL] fact_critic: unparseable critic output (%s) — keeping originals",
                e,
            )
        return repaired

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
        - blurbs: a list of structured per-candidate entries for the TOP 3
          ranked candidates in your summary, in rank order. Each entry is a
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
        # Fact-check gate: refuse to finalize until the blurbs have been fact-checked by critique_ranking this run. Returns a
        # non-terminal instruction so the agent loop calls critique_ranking and then retries finalize. (Reject path returns
        # an empty artifact dict so the content_and_artifact contract holds.)
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
        # NOTE: the former finalize-time fact-critic re-run (which repaired false "no Phase 3 / no development program"
        # claims in the LLM-written verdict/blocker/prose) has been removed. Those interpretive fields are now authored ONLY
        # by the isolated judge_interpretive call (in the enrich pass below), fed the resolved stage —  it cannot produce
        # that false claim, so there is nothing to repair. critique_ranking (fact check) is unaffected and still runs as the
        # pre-finalize gate above.
        validated: list[dict] = []
        structured_keys = (
            "stage",
            "literature",
            "blocker",
            "active_programs",
            "key_risk",
            "verdict",
            "watch",
        )
        for item in blurbs or []:
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
            # Top-N evidence gate: drop candidates where 0 trials AND synthesize indicates no usable literature signal —
            # strength="none" OR study_count==0 (OR both). Strong/moderate/weak literature with at least one relevant
            # abstract and 0 trials is a legitimate repurposing signal and is kept. PMID-count fallback applies only when
            # synthesize didn't run. Investigated-but-filtered candidates still appear in disease_findings (per-disease
            # section); they're only removed from top-N + blurbs.
            slot = findings_local.get(disease_key) or {}
            lit = slot.get("literature")
            ct = slot.get("clinical_trials")
            n_pmids = len(lit.pmids) if lit else 0
            n_trials = (
                ct.search.total_count
                if (ct is not None and ct.search is not None)
                else 0
            )
            lit_strength = (
                lit.evidence_summary.strength if lit and lit.evidence_summary else None
            )
            lit_direction = (
                lit.evidence_summary.direction if lit and lit.evidence_summary else None
            )
            lit_study_count = (
                lit.evidence_summary.study_count
                if lit and lit.evidence_summary
                else None
            )
            # Zero-evidence gate keys off DIRECTION, not strength: a contradicting body is real evidence and must survive the
            # gate, ranked as a negative.
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
                logger.warning(
                    "[TOOL] finalize_supervisor dropping blurb for disease=%r "
                    "(evidence gate: %d trials, %d PMIDs, strength=%s, "
                    "direction=%s, study_count=%s)",
                    disease,
                    n_trials,
                    n_pmids,
                    lit_strength,
                    lit_direction,
                    lit_study_count,
                )
                continue
            # Deterministic STAGE override: the development-stage tier is a fact the LLM must NOT author. The clinical-trials
            # sub-agent computed an authoritative dev_stage from the relevance-filtered signals; render its phrase verbatim,
            # overwriting whatever the LLM wrote for `stage`. This replaces the earlier per-signal regex repairs (false-"no
            # Phase 3" / false-"no development program") with one source of truth, so there is no path left for the blurb
            # stage to contradict the trial section. Only when a dev_stage phrase is available (sub-agent classified
            # relevance) — otherwise the LLM's stage is left as-is rather than asserting a stage the signal doesn't show.
            sig = ct.signals if ct else None
            stage_phrase = _dev_stage_phrase(sig)
            if stage_phrase is not None and stage_phrase != fields["stage"]:
                logger.warning(
                    "[TOOL] finalize_supervisor set stage from dev_stage=%s for disease=%r; "
                    "was: %r",
                    sig.dev_stage,
                    disease,
                    fields["stage"],
                )
                fields["stage"] = stage_phrase
            # active_programs ("what is still moving") is also an authoritative fact from the isolated stage judgment
            # (services/dev_stage), not a free-text interpretation — the blurb LLM kept mis-filling it (e.g. listing a
            # COMPLETED trial as an active program). Render it verbatim from the signal, overwriting whatever the LLM wrote.
            if sig is not None and sig.active_programs:
                if sig.active_programs != fields["active_programs"]:
                    logger.warning(
                        "[TOOL] finalize_supervisor set active_programs from signal for "
                        "disease=%r; was: %r",
                        disease,
                        fields["active_programs"],
                    )
                fields["active_programs"] = sig.active_programs
            # literature: overwrite with the deterministic one-liner from the typed EvidenceSummary (strength + direction +
            # design). One source of truth for the design word — replaces the earlier deterministic 'observational' text
            # repair.
            es = lit.evidence_summary if lit else None
            if es is not None:
                fields["literature"] = _literature_oneliner(es)
            # Stash the RESOLVED facts for the interpretive enrich pass after the loop. The interpretive fields
            # (blocker/key_risk/verdict/prose) are authored ONLY by the isolated judge_interpretive call fed these facts —
            # never re-derived in the blurb pass (which contradicted the stage). `_interp_facts` is None when no stage phrase
            # is available (sub-agent didn't classify) — then the LLM blurb text stays.
            approved_ind = (
                ct.approval.matched_indication
                if (ct is not None and ct.approval is not None)
                else None
            )
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
                    # Registry trial count so the judge doesn't call a multi-trial candidate untested/abandoned when its
                    # literature came back empty.
                    "trials_on_record": n_trials,
                }
                if stage_phrase is not None
                else None
            )
            entry = {
                "disease": disease,
                "prose": prose,
                "_interp_facts": interp_facts,
                **fields,
            }
            validated.append(entry)

        # Enrich pass: synthesize the interpretive fields from the resolved facts, concurrently. judge_interpretive is the
        # SINGLE author of blocker/key_risk/verdict/prose — fed the already-resolved stage/active_programs/literature/
        # approval so it cannot contradict them (proven in
        # scratch/{interpretive_fields,staged_blurb,approval_input}_harness.py). Entries without resolved facts
        # (_interp_facts is None) keep their LLM blurb text.
        _to_interp = [e for e in validated if e.get("_interp_facts")]
        if _to_interp:
            # drug name is for cache-readability/logging only (not part of the cache identity, which is the fact-tuple). A
            # run is for one drug → take the single drug_facts key.
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
            for e, j in zip(_to_interp, _results):
                if j is not None:
                    e["blocker"] = j.blocker
                    e["key_risk"] = j.key_risk
                    e["verdict"] = j.verdict
                    e["prose"] = j.prose
        for e in validated:
            e.pop("_interp_facts", None)

        # Filter the LLM-written summary to drop ranked lines whose disease didn't pass the evidence gate (not in validated).
        # Non-ranked lines (e.g. trailing "Closed signals:") pass through unchanged. Surviving lines are renumbered to stay
        # contiguous. Uses the same regex and longest-substring match strategy as the report formatter's
        # _splice_blurbs_into_summary so disease matching is consistent.
        validated_diseases = {e["disease"].lower().strip() for e in validated}

        # Authoritative dev_stage phrase per disease, from the relevance-filtered signals of every investigated candidate
        # (NOT just the ranked/validated ones — demoted candidates appear only in the footer and must be corrected there
        # too). Used to overwrite a false stage clause the LLM wrote in a demotion footer line.
        dev_stage_by_disease: dict[str, str] = {}
        for _dkey, _slot in findings_local.items():
            _ct = (_slot or {}).get("clinical_trials")
            _phrase = _dev_stage_phrase(_ct.signals if _ct else None)
            if _phrase is not None:
                dev_stage_by_disease[_dkey] = _phrase

        # Reuses the module-level _FALSE_STAGE_FRAGMENT / _PROGRAM_STAGES (shared with the blurb sibling-field scrub) so the
        # footer, ranked line, and blurb fields enforce one rule.
        footer_line = re.compile(r"^\s*-\s+(?P<disease>.+?)\s+—\s+.+$")

        def _repair_stage_clause(disease_text: str, line: str, context: str) -> str:
            """Replace a false 'no program / Phase 4 only' stage clause in `line` with the authoritative dev_stage phrase,
            when the matched disease's dev_stage asserts a real Phase 3 program. Used for BOTH the ranked summary line and
            the demotion footer line — the same class of LLM-authored stage leak appears in both."""
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
            return _repair_stage_clause(fm.group("disease"), line, "demotion footer")

        rank_line = re.compile(
            r"^\s*(?P<rank>\d+)\.\s+(?P<head>.+?)\s+—\s+(?P<tail>.+)$"
        )
        # Normalize the heading line so the report uses "signals" instead of "candidates" / "opportunities" / "indications"
        # regardless of what the LLM wrote.
        heading_line = re.compile(
            r"^\s*Ranked\s+repurposing\s+"
            r"(?:candidates|opportunities|indications|signals)"
            r"(?P<rest>\s+for\s+.+?:?\s*)$",
            re.IGNORECASE,
        )
        filtered_lines: list[str] = []
        next_rank = 1
        for line in summary.splitlines():
            heading_match = heading_line.match(line)
            if heading_match is not None:
                rest = heading_match.group("rest").rstrip()
                if not rest.endswith(":"):
                    rest = rest + ":"
                filtered_lines.append(f"Ranked repurposing signals{rest}")
                continue
            m = rank_line.match(line)
            if m is None:
                # Non-ranked line (e.g. a demotion footer entry). Correct a false stage clause against the authoritative
                # dev_stage before passing it through.
                filtered_lines.append(_repair_footer_stage(line))
                continue
            head_lower = m.group("head").lower()
            keep = any(
                d and d in head_lower
                for d in sorted(validated_diseases, key=len, reverse=True)
            )
            if not keep:
                logger.warning(
                    "[TOOL] finalize_supervisor dropping summary line for "
                    "disease=%r (failed evidence gate)",
                    m.group("head"),
                )
                continue
            # Repair a false development-status clause in the ranked line's tail against the authoritative dev_stage (same
            # leak class as the demotion footer: the LLM authors the dev-status phrase from the prompt's tier menu and can
            # contradict the signal).
            ranked_line = f"{next_rank}. {m.group('head')} — {m.group('tail')}"
            ranked_line = _repair_stage_clause(
                m.group("head"), ranked_line, "ranked summary line"
            )
            filtered_lines.append(ranked_line)
            next_rank += 1
        filtered_summary = "\n".join(filtered_lines)

        artifact = {"summary": filtered_summary, "blurbs": validated}
        return "Supervisor analysis complete.", artifact

    def get_merged_allowlist() -> (
        dict[str, tuple[str, Literal["competitor", "mechanism", "both"]]]
    ):
        """Snapshot the post-merge competitor + mechanism disease allowlist.

        Returns a copy keyed by lowercase disease name → (canonical_name, source). Sources are "competitor", "mechanism", or
        "both" depending on which sub-agent surfaced the disease.
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

    # supervisor_fanout has no default in Settings, so a missing SUPERVISOR_FANOUT fails loudly at startup (ValidationError)
    # — it can never silently default to False. A holdout (date_before) run uses this same flag as a production run: when
    # False the supervisor investigates per candidate via the ReAct loop, when True it fans out via
    # investigate_top_candidates. Both mirror production; holdout is no longer special-cased here.
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
        # Insert investigate_top_candidates before finalize so the LLM can see it after seed-phase tools but before
        # terminating. Enabled when supervisor_fanout is set (parallel fan-out for speed; see
        # config.Settings.supervisor_fanout).
        tools.insert(-1, investigate_top_candidates)
        # Force the parallel path: with the per-candidate tools removed, the LLM cannot investigate serially (it ignores the
        # prompt-level fan-out directive on its own), so it must call investigate_top_candidates once.
        tools = [
            t for t in tools if t not in (analyze_literature, analyze_clinical_trials)
        ]
    return tools, get_merged_allowlist, get_auto_findings, get_approval_labels
