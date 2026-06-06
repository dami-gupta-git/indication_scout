"""Analyses routes (PLAN_react.md §2.3, T2.2–T2.5).

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
from indication_scout.report.format_report import format_report
from indication_scout.services.analysis_runner import run_analysis
from indication_scout.services.job_store import Job, job_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/analyses", tags=["analyses"])


async def _execute(job: Job) -> None:
    """Run the analysis for `job`, recording status/result/error. Catches cancellation."""
    job.status = "running"
    try:
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
        raise HTTPException(status_code=409, detail=f"Job not done (status={job.status})")
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
