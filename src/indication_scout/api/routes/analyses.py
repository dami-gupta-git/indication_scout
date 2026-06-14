"""Analyses routes.

Blocking runner inside an asyncio background task; the frontend polls `GET` for status/result.
No orchestration touch — the runner calls the existing blocking `run_analysis`.
"""

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Response, status
from fastapi.responses import PlainTextResponse

from indication_scout.api.schemas.analyses import (
    AnalysisCreatedResponse,
    AnalysisRequest,
    AnalysisStatusResponse,
)
from indication_scout.constants import DEFAULT_CACHE_DIR, SEED_REPORT_SPINNER_SECONDS
from indication_scout.data_sources.base_client import DataSourceError
from indication_scout.data_sources.chembl import resolve_drug_name
from indication_scout.helpers.drug_helpers import normalize_drug_name
from indication_scout.report.format_report import format_report
from indication_scout.services.analysis_runner import run_analysis
from indication_scout.services.job_store import Job, job_store
from indication_scout.services.seed_reports import load_fresh_seed_report

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/analyses", tags=["analyses"])


async def _execute(job: Job) -> None:
    """Run the analysis for `job`, recording status/result/error. Catches cancellation."""
    job.status = "running"
    try:
        seed = load_fresh_seed_report(job.drug_name)
        if seed is not None:
            # Fresh seed report: skip the agents, hold the spinner briefly, then serve it.
            logger.info(
                "Job %s served from seed report for %s", job.job_id, job.drug_name
            )
            await asyncio.sleep(SEED_REPORT_SPINNER_SECONDS)
            job.result = seed
            job.status = "done"
            return
        output, _ = await run_analysis(job.drug_name)
        job.result = output
        job.status = "done"
    except asyncio.CancelledError:
        job.status = "cancelled"
        logger.info("Job %s cancelled", job.job_id)
        raise
    except Exception as exc:  # noqa: BLE001 — surface any runner failure to the client
        job.error = str(exc)
        job.status = "error"
        logger.exception("Job %s failed", job.job_id)


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def create_analysis(req: AnalysisRequest) -> AnalysisCreatedResponse:
    """Launch a background analysis; return its job id immediately."""
    drug = normalize_drug_name(req.drug_name)
    # Fail fast: one quick Open Targets search confirms the drug exists before we spin up
    # a job. Seed-report drugs skip the check (they don't need OT resolution).
    if load_fresh_seed_report(drug) is None:
        try:
            await resolve_drug_name(drug, DEFAULT_CACHE_DIR)
        except DataSourceError:
            raise HTTPException(
                status_code=422,
                detail=f"No drug found matching '{req.drug_name}'.",
            )
    job = job_store.create(req.drug_name)
    job.task = asyncio.create_task(_execute(job))
    return AnalysisCreatedResponse(job_id=job.job_id, status=job.status)


@router.get("/{job_id}")
async def get_analysis(job_id: str) -> AnalysisStatusResponse:
    """Return current status and, when done, the analysis result."""
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return AnalysisStatusResponse(
        job_id=job.job_id,
        drug_name=job.drug_name,
        status=job.status,
        result=job.result,
        error=job.error,
    )


@router.get("/{job_id}/report.md", response_class=PlainTextResponse)
async def get_analysis_report(job_id: str) -> str:
    """Return the formatted Markdown report for a completed job."""
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "done" or job.result is None:
        raise HTTPException(
            status_code=409, detail=f"Job not done (status={job.status})"
        )
    return format_report(job.result)


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_analysis(job_id: str) -> Response:
    """Cancel a running job. Idempotent: finished jobs are left as-is; absent → 404."""
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.task is not None and not job.task.done():
        job.task.cancel()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
