"""Analysis runner — invokes the supervisor agent for a drug, outside any UI.

The single shared entry point for running the pipeline: both the CLI (`scout find`) and the
FastAPI layer call `run_analysis`. No duplicate agent-building wiring. Uses the existing blocking
`run_supervisor_agent` (`ainvoke`).
"""

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
from indication_scout.db.session import get_db
from indication_scout.helpers.drug_helpers import normalize_drug_name
from indication_scout.report.format_report import format_report
from indication_scout.services.retrieval import RetrievalService

logger = logging.getLogger(__name__)


def build_agent(
    db: Session,
    date_before: date | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
):
    settings = get_settings()
    llm = ChatAnthropic(
        model=settings.llm_model,
        temperature=0,
        max_tokens=4096,
        anthropic_api_key=settings.anthropic_api_key,
    )
    svc = RetrievalService(cache_dir)
    return build_supervisor_agent(llm, svc=svc, db=db, date_before=date_before)


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

    drug = normalize_drug_name(drug_name)
    # Fail fast: one quick Open Targets search confirms the drug exists before any agents run.
    # Raises DataSourceError if not found; the result is cached for the in-agent resolves.
    await resolve_drug_name(drug, DEFAULT_CACHE_DIR)
    db = next(get_db())
    _t0 = time.perf_counter()
    reset_api_timing()

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
        agent, get_merged_allowlist, get_auto_findings = build_agent(db, date_before=date_before)
        output = await run_supervisor_agent(
            agent, get_merged_allowlist, drug, get_auto_findings=get_auto_findings
        )
        total = time.perf_counter() - _t0
        # External-API breakdown: how much of the run was spent awaiting HTTP responses
        # (PubMed, ClinicalTrials.gov, OpenTargets, FDA, ChEMBL), per source.
        api = api_timing_snapshot()
        api_total = sum(s for _, s in api.values())
        per_source = ", ".join(
            f"{src}={secs:.1f}s/{cnt} calls" for src, (cnt, secs) in sorted(api.items())
        )
        logger.warning("[TIMING] run_analysis(%s) total: %.1fs", drug, total)
        logger.warning(
            "[TIMING] external API total: %.1fs (%.0f%% of run) — %s",
            api_total,
            100 * api_total / total if total else 0,
            per_source or "no calls",
        )
        return output, format_report(output)
    finally:
        # Ensure the warmup task is settled so it isn't garbage-collected while
        # pending (which logs a noisy "Task was destroyed" warning).
        if not _warmup_task.done():
            _warmup_task.cancel()
        db.close()
