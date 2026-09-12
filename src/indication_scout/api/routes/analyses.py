"""Analyses routes.

The runner remains inside an asyncio background task, while Postgres owns lifecycle state,
progress, errors, and validated results for the polling and report routes.
"""

import asyncio
import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from fastapi import APIRouter, HTTPException, Response, status
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import sessionmaker

from indication_scout.agents.supervisor.supervisor_output import SupervisorOutput
from indication_scout.api.schemas.analyses import (
    AnalysisCreatedResponse,
    AnalysisRequest,
    AnalysisStatusResponse,
)
from indication_scout.api.schemas.progress import ProgressEvent
from indication_scout.constants import DEFAULT_CACHE_DIR, SEED_REPORT_SPINNER_SECONDS
from indication_scout.data_sources.base_client import DataSourceError
from indication_scout.data_sources.chembl import resolve_drug_name
from indication_scout.db.session import make_session_factory
from indication_scout.helpers.drug_helpers import normalize_drug_name
from indication_scout.metrics import analysis_finished, analysis_started
from indication_scout.observability import (
    bind_log_context,
    log_context,
    reset_log_context,
)
from indication_scout.report.format_report import format_report
from indication_scout.services.analysis_runner import run_analysis
from indication_scout.services.job_store import Job, job_store
from indication_scout.services.progress import reset_emitter, set_emitter
from indication_scout.services.run_repository import (
    AnalysisRunRepository,
    RunStateError,
)
from indication_scout.services.seed_reports import load_fresh_seed_report
from indication_scout.sqlalchemy.analysis_runs import AnalysisEvent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/analyses", tags=["analyses"])


@lru_cache(maxsize=1)
def _get_run_session_factory() -> sessionmaker:
    """Create one shared database connection pool for API run persistence."""
    return make_session_factory()


def dispose_run_session_factory() -> None:
    """Dispose the shared run-persistence pool during application shutdown."""
    if _get_run_session_factory.cache_info().currsize == 0:
        return
    _get_run_session_factory().kw["bind"].dispose()
    _get_run_session_factory.cache_clear()


@contextmanager
def _repository_scope() -> Iterator[AnalysisRunRepository]:
    """Provide one short-lived repository session."""
    with _get_run_session_factory()() as db:
        yield AnalysisRunRepository(db)


def _execution_identity() -> tuple[str | None, str | None, str | None]:
    """Return Railway execution identifiers when present."""
    return (
        os.environ.get("RAILWAY_REPLICA_ID"),
        os.environ.get("RAILWAY_DEPLOYMENT_ID"),
        os.environ.get("RAILWAY_GIT_COMMIT_SHA"),
    )


def _persist_progress(job: Job, attempt_id: str, phase: str, message: str) -> None:
    """Persist a progress event and retain the process-local mirror."""
    job.emit(phase, message)
    with _repository_scope() as repository:
        repository.append_event(
            job.job_id,
            "analysis.progress",
            "info",
            attempt_id=attempt_id,
            stage=phase,
            attributes={"message": message},
        )
    with log_context(stage=phase):
        logger.info(
            "Analysis progress",
            extra={"event_name": "analysis.progress"},
        )


def _progress_from_events(events: list[AnalysisEvent]) -> list[ProgressEvent]:
    """Rebuild the API progress feed from durable events."""
    progress = []
    for event in events:
        attributes = event.attributes or {}
        message = attributes.get("message")
        if (
            event.event_name == "analysis.progress"
            and event.stage is not None
            and isinstance(message, str)
        ):
            progress.append(ProgressEvent(phase=event.stage, message=message))
    return progress


async def _execute(job: Job) -> None:
    """Run the analysis for `job`, recording status/result/error. Catches cancellation."""
    worker_id, deployment_id, release = _execution_identity()
    try:
        with _repository_scope() as repository:
            attempt = repository.start_attempt(
                job.job_id,
                worker_id=worker_id,
                deployment_id=deployment_id,
                release=release,
            )
            run = repository.get_run(job.job_id)
    except RunStateError:
        with _repository_scope() as repository:
            run = repository.get_run(job.job_id)
        if run is not None and run.status == "cancelled":
            job.status = "cancelled"
            return
        raise

    if run is None:
        raise RuntimeError(
            f"Analysis run disappeared after attempt start: {job.job_id}"
        )

    job.status = "running"
    started = time.monotonic()
    outcome = "error"
    analysis_started(run.submission_source, run.execution_mode)
    log_token = bind_log_context(
        run_id=job.job_id,
        attempt_id=attempt.attempt_id,
        execution_mode=run.execution_mode,
        submission_source=run.submission_source,
    )
    logger.info("Analysis started", extra={"event_name": "analysis.started"})
    # Bind this attempt's durable progress feed for the duration of the run. Reset in finally
    # so the context variable does not leak into other analyses.
    emitter_token = set_emitter(
        lambda phase, message: _persist_progress(
            job, attempt.attempt_id, phase, message
        )
    )
    try:
        seed = load_fresh_seed_report(job.drug_name)
        if seed is not None:
            # Fresh seed report: skip the agents, hold the spinner briefly, then serve it.
            logger.info(
                "Job %s served from seed report for %s", job.job_id, job.drug_name
            )
            await asyncio.sleep(SEED_REPORT_SPINNER_SECONDS)
            with _repository_scope() as repository:
                repository.complete_validated_attempt(
                    job.job_id, attempt.attempt_id, seed
                )
            job.result = seed
            job.status = "done"
            outcome = "done"
            return
        output, _ = await run_analysis(job.drug_name)
        with _repository_scope() as repository:
            repository.complete_validated_attempt(
                job.job_id, attempt.attempt_id, output
            )
        job.result = output
        job.status = "done"
        outcome = "done"
    except asyncio.CancelledError:
        with _repository_scope() as repository:
            repository.cancel_attempt(job.job_id, attempt.attempt_id)
        job.status = "cancelled"
        outcome = "cancelled"
        logger.info("Analysis cancelled", extra={"event_name": "analysis.cancelled"})
        raise
    except Exception as exc:  # noqa: BLE001 — surface any runner failure to the client
        persisted_error = f"{type(exc).__name__}: {exc}"
        with _repository_scope() as repository:
            repository.fail_attempt(
                job.job_id,
                attempt.attempt_id,
                error_code=type(exc).__name__,
                error_message=persisted_error,
                retryable=False,
                integrity_failed=False,
            )
        job.error = str(exc)
        job.status = "error"
        logger.exception(
            "Analysis failed",
            extra={"event_name": "analysis.failed", "outcome": "error"},
        )
    finally:
        duration = time.monotonic() - started
        analysis_finished(
            run.submission_source,
            run.execution_mode,
            outcome,
            duration,
        )
        logger.info(
            "Analysis attempt finished",
            extra={
                "event_name": "analysis.finished",
                "duration_seconds": duration,
                "outcome": outcome,
            },
        )
        reset_emitter(emitter_token)
        reset_log_context(log_token)


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def create_analysis(req: AnalysisRequest) -> AnalysisCreatedResponse:
    """Launch a background analysis; return its job id immediately."""
    drug = normalize_drug_name(req.drug_name)
    # Fail fast: one quick Open Targets search confirms the drug exists before we spin up
    # a job. Seed-report drugs skip the check (they don't need OT resolution).
    seed = load_fresh_seed_report(drug)
    if seed is None:
        try:
            await resolve_drug_name(drug, DEFAULT_CACHE_DIR)
        except DataSourceError as e:
            raise HTTPException(
                status_code=422,
                detail=f"No drug found matching '{req.drug_name}'.",
            ) from e
    with _repository_scope() as repository:
        run = repository.create_run(
            drug,
            "seed" if seed is not None else "live",
            submission_source="api",
            analysis_kind="find",
            disease_name=None,
            date_before=None,
        )
    job = job_store.create(drug, job_id=run.run_id)
    job.task = asyncio.create_task(_execute(job))
    return AnalysisCreatedResponse(job_id=job.job_id, status=job.status)


@router.get("/{job_id}")
async def get_analysis(job_id: str) -> AnalysisStatusResponse:
    """Return current status and, when done, the analysis result."""
    with _repository_scope() as repository:
        run = repository.get_run(job_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Job not found")
        attempts = repository.list_attempts(job_id)
        events = repository.list_events(job_id)
    result = (
        SupervisorOutput.model_validate(run.result) if run.result is not None else None
    )
    error = attempts[-1].error_message if run.status == "error" and attempts else None
    return AnalysisStatusResponse(
        job_id=run.run_id,
        drug_name=run.drug_name,
        status=run.status,
        result=result,
        error=error,
        progress=_progress_from_events(events),
    )


@router.get("/{job_id}/report.md", response_class=PlainTextResponse)
async def get_analysis_report(job_id: str) -> str:
    """Return the formatted Markdown report for a completed job."""
    with _repository_scope() as repository:
        run = repository.get_run(job_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if run.status != "done" or run.result is None:
        raise HTTPException(
            status_code=409, detail=f"Job not done (status={run.status})"
        )
    return format_report(SupervisorOutput.model_validate(run.result))


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_analysis(job_id: str) -> Response:
    """Cancel a running job. Idempotent: finished jobs are left as-is; absent → 404."""
    cancel_runtime = False
    with _repository_scope() as repository:
        run = repository.get_run(job_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if run.status == "pending":
            try:
                repository.cancel_pending_run(job_id)
                cancel_runtime = True
            except RunStateError:
                current = repository.get_run(job_id)
                if current is not None and current.status == "running":
                    repository.request_cancellation(job_id)
                    cancel_runtime = True
        elif run.status == "running":
            try:
                repository.request_cancellation(job_id)
                cancel_runtime = True
            except RunStateError:
                pass

    job = job_store.get(job_id)
    if (
        cancel_runtime
        and job is not None
        and job.task is not None
        and not job.task.done()
    ):
        job.task.cancel()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
