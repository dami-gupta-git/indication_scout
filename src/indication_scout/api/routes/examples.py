"""Landing-page example routes.

Serves a cached SupervisorOutput for each example drug. A cached run is kept on
the persistent volume for EXAMPLE_CACHE_TTL_SECONDS; once it lapses (or is
absent) the next request runs one real analysis, persists it, and serves it.
The analysis is run only once per TTL window — repeat clicks hit the cache.
"""

import asyncio
import json
import logging
import os
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from indication_scout.api.schemas.analyses import AnalysisStatusResponse
from indication_scout.report.format_report import format_report
from indication_scout.constants import (
    EXAMPLE_CACHE_DIR,
    EXAMPLE_CACHE_TTL_SECONDS,
    EXAMPLE_DRUGS,
    EXAMPLE_SEED_DIR,
)
from indication_scout.agents.supervisor.supervisor_output import SupervisorOutput
from indication_scout.services.analysis_runner import run_analysis

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/examples", tags=["examples"])

# Guards against two concurrent requests for the same expired example both
# kicking off a (costly) analysis. One key per drug.
_locks: dict[str, asyncio.Lock] = {drug: asyncio.Lock() for drug in EXAMPLE_DRUGS}


def _cache_path(drug: str):
    return EXAMPLE_CACHE_DIR / f"{drug}.json"


def seed_example_cache() -> None:
    """Copy committed seed snapshots into the volume cache for any example that
    is missing there. Sets each file's mtime to its capture date so the TTL
    counts from capture, not deploy. Existing cache entries are left untouched.
    """
    captured_path = EXAMPLE_SEED_DIR / "captured_at.json"
    if not captured_path.exists():
        logger.warning("No example seed manifest at %s; skipping seed", captured_path)
        return
    captured_at: dict[str, float] = json.loads(captured_path.read_text())
    EXAMPLE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for drug in EXAMPLE_DRUGS:
        dest = _cache_path(drug)
        if dest.exists():
            continue
        src = EXAMPLE_SEED_DIR / f"{drug}.json"
        if not src.exists():
            logger.warning("No seed file for example %s at %s", drug, src)
            continue
        dest.write_text(src.read_text())
        mtime = captured_at.get(drug)
        if mtime is not None:
            os.utime(dest, (mtime, mtime))
        logger.info("Seeded example cache for %s", drug)


def _load_fresh(drug: str) -> SupervisorOutput | None:
    """Return the cached output if present and within TTL, else None."""
    path = _cache_path(drug)
    if not path.exists():
        return None
    age = time.time() - path.stat().st_mtime
    if age > EXAMPLE_CACHE_TTL_SECONDS:
        return None
    try:
        return SupervisorOutput.model_validate_json(path.read_text())
    except Exception:  # noqa: BLE001 — a corrupt cache file should not 500
        logger.exception("Corrupt example cache for %s; will re-run", drug)
        return None


def _save(drug: str, output: SupervisorOutput) -> None:
    EXAMPLE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(drug).write_text(output.model_dump_json())


@router.get("/{drug}")
async def get_example(drug: str) -> AnalysisStatusResponse:
    """Return the cached example output, running it once if stale or missing."""
    drug = drug.strip().lower()
    if drug not in _locks:
        raise HTTPException(status_code=404, detail=f"Not an example drug: {drug}")

    cached = _load_fresh(drug)
    if cached is None:
        async with _locks[drug]:
            # Re-check inside the lock: another request may have just filled it.
            cached = _load_fresh(drug)
            if cached is None:
                logger.info("Example cache miss/stale for %s; running analysis", drug)
                output, _ = await run_analysis(drug)
                _save(drug, output)
                cached = output

    return AnalysisStatusResponse(
        job_id="example",
        drug_name=drug,
        status="done",
        result=cached,
        error=None,
    )


@router.get("/{drug}/report.md", response_class=PlainTextResponse)
async def get_example_report(drug: str) -> str:
    """Return the formatted Markdown report for a cached example, running it once if stale."""
    drug = drug.strip().lower()
    if drug not in _locks:
        raise HTTPException(status_code=404, detail=f"Not an example drug: {drug}")

    cached = _load_fresh(drug)
    if cached is None:
        async with _locks[drug]:
            cached = _load_fresh(drug)
            if cached is None:
                logger.info("Example cache miss/stale for %s; running analysis", drug)
                output, _ = await run_analysis(drug)
                _save(drug, output)
                cached = output

    return format_report(cached)
