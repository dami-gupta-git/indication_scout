"""Analysis runner — invokes the supervisor agent for a drug, outside any UI.

The single shared entry point for running the pipeline: both the CLI (`scout find`) and the
FastAPI layer call `run_analysis`. No duplicate agent-building wiring. Uses the existing blocking
`run_supervisor_agent` (`ainvoke`).
"""

import asyncio
import logging
import time
from datetime import date
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from sqlalchemy.orm import Session

from indication_scout.agents.supervisor.supervisor_agent import (
    build_supervisor_agent,
    run_supervisor_agent,
)
from indication_scout.agents.supervisor.supervisor_output import SupervisorOutput
from indication_scout.config import get_settings
from indication_scout.constants import DEFAULT_CACHE_DIR
from indication_scout.db.session import make_session_factory
from indication_scout.helpers.drug_helpers import normalize_drug_name
from indication_scout.report.format_report import format_report
from indication_scout.services.retrieval import RetrievalService

logger = logging.getLogger(__name__)


def build_agent(
    db: Session,
    session_factory=None,
    date_before: date | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
):
    settings = get_settings()
    llm = ChatAnthropic(
        model=settings.llm_model,
        temperature=0,
        max_tokens=settings.llm_max_tokens,
        anthropic_api_key=settings.anthropic_api_key,
    )
    svc = RetrievalService(cache_dir)
    return build_supervisor_agent(
        llm, svc=svc, db=db, session_factory=session_factory, date_before=date_before
    )


async def run_pair_analysis(
    drug_name: str, disease_name: str, *, date_before: date | None = None
) -> tuple[SupervisorOutput, str]:
    """Run the pipeline on a single fixed (drug, disease) pair; return its output and Markdown report.

    CLI-only (`scout investigate`). Skips candidate discovery, competitor/mechanism-based disease surfacing, and the
    supervisor ranking loop: the disease is taken as given. Runs the mechanism agent (drug MoA context), plus the
    literature and clinical-trials sub-agents for the pair. Replicates find_candidates's drug-level seeding via
    seed_drug_intake so the sub-agents see the same first_approval / approved_indications a normal run would.

    The result is assembled into a SupervisorOutput with a single CandidateFindings (source="" — the pair was neither
    competitor- nor mechanism-sourced) and no ranked summary/blurb, then rendered with the shared format_report, so the
    report shape matches `scout find`.
    """
    from indication_scout.agents.clinical_trials.clinical_trials_agent import (
        build_clinical_trials_agent,
        run_clinical_trials_agent,
    )
    from indication_scout.agents.literature.literature_agent import (
        build_literature_agent,
        run_literature_agent,
    )
    from indication_scout.agents.mechanism.mechanism_agent import (
        build_mechanism_agent,
        run_mechanism_agent,
    )
    from indication_scout.agents.supervisor.supervisor_output import (
        CandidateBlurb,
        CandidateFindings,
    )
    from indication_scout.agents.supervisor.supervisor_tools import _literature_oneliner
    from indication_scout.helpers.drug_helpers import seed_drug_intake
    from indication_scout.services.approval_check import (
        get_fda_approved_disease_mapping,
    )
    from indication_scout.services.dev_stage import dev_stage_phrase
    from indication_scout.services.embeddings import embed_async
    from indication_scout.services.judge_interpretive import judge_interpretive

    drug = normalize_drug_name(drug_name)
    # Fail-fast the drug exists (raises DataSourceError) and gather drug-level facts the sub-agents read.
    intake = await seed_drug_intake(drug, DEFAULT_CACHE_DIR, date_before=date_before)

    settings = get_settings()
    llm = ChatAnthropic(
        model=settings.llm_model,
        temperature=0,
        max_tokens=settings.llm_max_tokens,
        anthropic_api_key=settings.anthropic_api_key,
    )
    svc = RetrievalService(DEFAULT_CACHE_DIR)
    # One shared sessionmaker (single engine/pool). Only the literature agent uses a DB session, and it must get its own
    # per-call session — a SQLAlchemy Session is not safe across the concurrent gather below.
    session_factory = make_session_factory()

    # Warm the embedding model off the critical path (see run_analysis for rationale). Fire-and-forget; non-fatal.
    async def _warm_embeddings() -> None:
        try:
            await embed_async(["warmup"])
        except Exception as e:  # noqa: BLE001 — warmup must never break the run
            logger.warning("Embedding model warmup failed (non-fatal): %s", e)

    _warmup_task = asyncio.create_task(_warm_embeddings())

    async def _run_literature() -> object:
        # Own session for this coroutine — never shared with the concurrent trials/mechanism work.
        with session_factory() as call_db:
            lit_agent = build_literature_agent(
                llm=llm,
                svc=svc,
                db=call_db,
                date_before=date_before,
                approved_indications=list(intake.approved_indications),
            )
            return await run_literature_agent(lit_agent, drug, disease_name)

    async def _run_clinical_trials() -> object:
        ct_agent = build_clinical_trials_agent(
            llm=llm, date_before=date_before, assigned_indication=disease_name
        )
        return await run_clinical_trials_agent(
            ct_agent,
            drug,
            disease_name,
            first_approval=intake.first_approval,
            approved_indications=list(intake.approved_indications),
        )

    async def _run_mechanism() -> object:
        mech_agent = build_mechanism_agent(llm=llm, date_before=date_before)
        return await run_mechanism_agent(mech_agent, drug, date_before=date_before)

    try:
        logger.info(
            "Starting pair analysis %s x %s (date_before=%s)",
            drug,
            disease_name,
            date_before,
        )
        mechanism, literature, clinical_trials = await asyncio.gather(
            _run_mechanism(), _run_literature(), _run_clinical_trials()
        )

        # Label-grounded FDA relationship for the pair, from the same source find_candidates uses (NOT LLM prose). Unlike the
        # supervisor path we do NOT drop an "approved" pair — the user asked for it explicitly — but we still surface the
        # relationship so the blurb judge reasons on-label instead of as if novel (e.g. metformin × type 2 diabetes). Maps to
        # CandidateFindings.approval_relationship, which has no "approved" member: an "approved" pair carries no relationship
        # tag ("none") but its matched approved_indication is threaded to the judge below.
        fda_mapping = await get_fda_approved_disease_mapping(
            drug_name=drug,
            candidate_diseases=[disease_name],
            cache_dir=DEFAULT_CACHE_DIR,
        )
        fda_label = fda_mapping.get(disease_name, "none")
        approval_relationship = (
            fda_label if fda_label in ("contaminated", "combination_only") else "none"
        )

        # Build the per-candidate blurb the report card renders. Replicates finalize_supervisor's enrich pass for a single
        # pair: the deterministic fields (stage / active_programs / literature) come straight from the trial signals and the
        # typed EvidenceSummary; the interpretive fields (blocker / key_risk / verdict / prose) are authored ONLY by
        # judge_interpretive, fed those resolved facts so it cannot contradict them. No ranking/critic loop is involved.
        blurb: CandidateBlurb | None = None
        sig = clinical_trials.signals if clinical_trials else None
        stage_phrase = dev_stage_phrase(sig)
        # Only synthesize when the trials sub-agent classified relevance (a dev_stage phrase exists) — mirrors
        # finalize_supervisor, which leaves the blurb empty otherwise rather than asserting an unsupported stage.
        if stage_phrase is not None:
            es = literature.evidence_summary if literature else None
            lit_oneliner = _literature_oneliner(es) if es is not None else ""
            active_programs = sig.active_programs if sig and sig.active_programs else ""
            # When the pair itself is FDA-approved, the candidate disease IS the approved indication — tell the judge so it
            # reasons on-label. Otherwise fall back to the trials agent's matched indication (a related approved sub-indication).
            approved_ind = (
                disease_name
                if fda_label == "approved"
                else (
                    clinical_trials.approval.matched_indication
                    if (
                        clinical_trials is not None
                        and clinical_trials.approval is not None
                    )
                    else None
                )
            )
            trials_on_record = (
                clinical_trials.search.total_count
                if (clinical_trials is not None and clinical_trials.search is not None)
                else 0
            )
            judgment = await judge_interpretive(
                stage=stage_phrase,
                active_programs=active_programs,
                literature=lit_oneliner,
                relationship=approval_relationship,
                approved_indication=approved_ind,
                trials_on_record=trials_on_record,
                cache_dir=DEFAULT_CACHE_DIR,
                drug=drug,
                indication=disease_name,
            )
            blurb = CandidateBlurb(
                stage=stage_phrase,
                literature=lit_oneliner,
                active_programs=active_programs,
                blocker=judgment.blocker if judgment else "",
                key_risk=judgment.key_risk if judgment else "",
                verdict=judgment.verdict if judgment else "",
                prose=judgment.prose if judgment else "",
            )

        findings = CandidateFindings(
            disease=disease_name,
            source="",
            approval_relationship=approval_relationship,
            literature=literature,
            clinical_trials=clinical_trials,
            blurb=blurb,
        )
        output = SupervisorOutput(
            drug_name=drug,
            candidate_diseases=[disease_name],
            mechanism=mechanism,
            disease_findings=[findings],
            top_diseases=[disease_name],
            summary="",
        )
        return output, format_report(output)
    finally:
        if not _warmup_task.done():
            _warmup_task.cancel()
        # Dispose the engine/pool this call created so a batch loop over pairs doesn't leak one per pair.
        session_factory.kw["bind"].dispose()


async def run_analysis(
    drug_name: str, *, date_before: date | None = None
) -> tuple[SupervisorOutput, str]:
    """Run the supervisor agent for `drug_name`; return its output and the Markdown report.

    Normalizes the drug name at the entry point (cache keys, tools, sub-agents, logs all see the
    same lowercased form), owns the DB session lifecycle (opened per run, always closed), and
    threads `date_before` for holdout runs. Blocking `ainvoke` path.
    """
    from indication_scout.data_sources.base_client import (
        api_timing_snapshot,
        reset_api_timing,
    )
    from indication_scout.data_sources.chembl import resolve_drug_name
    from indication_scout.services.embeddings import (
        embed_timing_snapshot,
        reset_embed_timing,
    )

    drug = normalize_drug_name(drug_name)
    # Fail fast: one quick Open Targets search confirms the drug exists before any agents run.
    # Raises DataSourceError if not found; the result is cached for the in-agent resolves.
    await resolve_drug_name(drug, DEFAULT_CACHE_DIR)
    # One shared sessionmaker (single engine/pool) for the whole run. The run-level
    # `db` serves the sequential seed tools; the concurrent investigate_top_candidates
    # fan-out checks out its own per-call sessions from this same factory (see
    # build_supervisor_tools). Do NOT create a factory per call — that leaks an engine
    # and pool each time.
    session_factory = make_session_factory()
    db = session_factory()
    _t0 = time.perf_counter()
    reset_api_timing()
    reset_embed_timing()

    # Warm the embedding model off the critical path. It lazy-loads (~10s+) on
    # first embed(), and the literature stage hits it from many parallel callers
    # that serialize on the model lock — paying the cold load there stalls all of
    # them. Loading now overlaps the OT/mechanism stages, which don't embed, so
    # the model is ready by the time literature needs it. Fire-and-forget; errors
    # are non-fatal (the real embed call will surface any genuine failure).
    import asyncio

    from indication_scout.services.embeddings import embed_async

    async def _warm_embeddings() -> None:
        try:
            await embed_async(["warmup"])
        except Exception as e:  # noqa: BLE001 — warmup must never break the run
            logger.warning("Embedding model warmup failed (non-fatal): %s", e)

    _warmup_task = asyncio.create_task(_warm_embeddings())

    try:
        agent, get_merged_allowlist, get_auto_findings, get_approval_labels = (
            build_agent(db, session_factory, date_before=date_before)
        )
        output = await run_supervisor_agent(
            agent,
            get_merged_allowlist,
            drug,
            get_auto_findings=get_auto_findings,
            get_approval_labels=get_approval_labels,
        )
        total = time.perf_counter() - _t0
        # External-API breakdown: how much of the run was spent awaiting HTTP responses
        # (PubMed, ClinicalTrials.gov, OpenTargets, FDA, ChEMBL), per source.
        api = api_timing_snapshot()
        api_total = sum(s for _, s in api.values())
        per_source = ", ".join(
            f"{src}={secs:.1f}s/{cnt} calls" for src, (cnt, secs) in sorted(api.items())
        )
        embed_calls, embed_total = embed_timing_snapshot()
        # Whatever is left after external API + BioLORD encode is Claude inference plus
        # agent-loop/serialization overhead. These three buckets are sequential enough that
        # the remainder is a useful proxy for "time spent waiting on the LLM".
        other = total - api_total - embed_total
        logger.warning("[TIMING] run_analysis(%s) total: %.1fs", drug, total)
        logger.warning(
            "[TIMING] external API total: %.1fs (%.0f%% of run) — %s",
            api_total,
            100 * api_total / total if total else 0,
            per_source or "no calls",
        )
        logger.warning(
            "[TIMING] embedding (BioLORD) total: %.1fs (%.0f%% of run) — %d encode calls",
            embed_total,
            100 * embed_total / total if total else 0,
            embed_calls,
        )
        logger.warning(
            "[TIMING] LLM + overhead (remainder): %.1fs (%.0f%% of run)",
            other,
            100 * other / total if total else 0,
        )
        return output, format_report(output)
    finally:
        # Ensure the warmup task is settled so it isn't garbage-collected while
        # pending (which logs a noisy "Task was destroyed" warning).
        if not _warmup_task.done():
            _warmup_task.cancel()
        db.close()
