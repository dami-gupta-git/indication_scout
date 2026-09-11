"""Transactional persistence for analysis runs, attempts, and events."""

import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from indication_scout.agents.supervisor.supervisor_output import SupervisorOutput
from indication_scout.sqlalchemy.analysis_runs import (
    ATTEMPT_STATUSES,
    EVENT_SEVERITIES,
    EXECUTION_MODES,
    AnalysisAttempt,
    AnalysisEvent,
    AnalysisRun,
)

ExecutionMode = Literal["live", "seed"]
EventSeverity = Literal["debug", "info", "warning", "error", "critical"]

_TERMINAL_RUN_STATUSES = {"done", "error", "cancelled"}


class RunNotFoundError(LookupError):
    """Raised when a requested run or attempt does not exist."""


class RunStateError(RuntimeError):
    """Raised when a requested lifecycle transition is invalid."""


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _duration_ms(started_at: datetime, finished_at: datetime) -> int:
    return max(0, int((finished_at - started_at).total_seconds() * 1000))


class AnalysisRunRepository:
    """Own database transactions for the durable analysis lifecycle."""

    def __init__(self, db: Session) -> None:
        self._db = db

    def create_run(
        self,
        drug_name: str,
        execution_mode: ExecutionMode,
        *,
        run_id: str | None = None,
    ) -> AnalysisRun:
        """Create a pending run and its initial event in one transaction."""
        normalized_drug = drug_name.strip()
        if not normalized_drug:
            raise ValueError("drug_name must not be empty")
        if execution_mode not in EXECUTION_MODES:
            raise ValueError(f"Unsupported execution mode: {execution_mode}")

        now = _utcnow()
        run = AnalysisRun(
            run_id=run_id or uuid.uuid4().hex,
            drug_name=normalized_drug,
            status="pending",
            execution_mode=execution_mode,
            result=None,
            integrity_status=None,
            cancellation_requested_at=None,
            created_at=now,
            started_at=None,
            finished_at=None,
            updated_at=now,
        )
        event = self._new_event(
            run_id=run.run_id,
            attempt_id=None,
            event_name="analysis.created",
            severity="info",
            occurred_at=now,
        )
        try:
            self._db.add(run)
            self._db.flush()
            self._db.add(event)
            self._db.commit()
        except Exception:
            self._db.rollback()
            raise
        self._db.refresh(run)
        return run

    def get_run(self, run_id: str) -> AnalysisRun | None:
        """Return a run without changing its lifecycle."""
        return self._db.get(AnalysisRun, run_id)

    def list_attempts(self, run_id: str) -> list[AnalysisAttempt]:
        """Return attempts in execution order."""
        statement = (
            select(AnalysisAttempt)
            .where(AnalysisAttempt.run_id == run_id)
            .order_by(AnalysisAttempt.attempt_number)
        )
        return list(self._db.scalars(statement))

    def list_events(self, run_id: str) -> list[AnalysisEvent]:
        """Return the append-only event timeline in insertion order."""
        statement = (
            select(AnalysisEvent)
            .where(AnalysisEvent.run_id == run_id)
            .order_by(AnalysisEvent.occurred_at, AnalysisEvent.event_id)
        )
        return list(self._db.scalars(statement))

    def start_attempt(
        self,
        run_id: str,
        *,
        worker_id: str | None,
        deployment_id: str | None,
        release: str | None,
        attempt_id: str | None = None,
    ) -> AnalysisAttempt:
        """Start the first execution attempt for a pending run."""
        now = _utcnow()
        try:
            run = self._require_run(run_id, for_update=True)
            self._require_run_status(run, {"pending"})
            latest_number = self._db.scalar(
                select(func.max(AnalysisAttempt.attempt_number)).where(
                    AnalysisAttempt.run_id == run_id
                )
            )
            attempt = AnalysisAttempt(
                attempt_id=attempt_id or uuid.uuid4().hex,
                run_id=run_id,
                attempt_number=(latest_number or 0) + 1,
                status="running",
                worker_id=worker_id,
                deployment_id=deployment_id,
                release=release,
                started_at=now,
                finished_at=None,
                heartbeat_at=now,
                duration_ms=None,
                error_code=None,
                error_message=None,
                retryable=None,
            )
            run.status = "running"
            run.started_at = now
            run.updated_at = now
            self._db.add(attempt)
            self._db.flush()
            self._db.add(
                self._new_event(
                    run_id=run_id,
                    attempt_id=attempt.attempt_id,
                    event_name="analysis.started",
                    severity="info",
                    occurred_at=now,
                )
            )
            self._db.commit()
        except Exception:
            self._db.rollback()
            raise
        self._db.refresh(attempt)
        return attempt

    def append_event(
        self,
        run_id: str,
        event_name: str,
        severity: EventSeverity,
        *,
        attempt_id: str | None = None,
        stage: str | None = None,
        duration_ms: int | None = None,
        dependency: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> AnalysisEvent:
        """Append an event without changing run or attempt state."""
        if not event_name.strip():
            raise ValueError("event_name must not be empty")
        if severity not in EVENT_SEVERITIES:
            raise ValueError(f"Unsupported event severity: {severity}")
        if duration_ms is not None and duration_ms < 0:
            raise ValueError("duration_ms must not be negative")

        try:
            self._require_run(run_id)
            if attempt_id is not None:
                self._require_attempt(run_id, attempt_id)
            event = self._new_event(
                run_id=run_id,
                attempt_id=attempt_id,
                event_name=event_name.strip(),
                severity=severity,
                occurred_at=_utcnow(),
                stage=stage,
                duration_ms=duration_ms,
                dependency=dependency,
                attributes=attributes,
            )
            self._db.add(event)
            self._db.commit()
        except Exception:
            self._db.rollback()
            raise
        self._db.refresh(event)
        return event

    def heartbeat(self, run_id: str, attempt_id: str) -> AnalysisAttempt:
        """Record that a running attempt still owns its work."""
        now = _utcnow()
        try:
            run = self._require_run(run_id, for_update=True)
            self._require_run_status(run, {"running"})
            attempt = self._require_attempt(run_id, attempt_id, for_update=True)
            self._require_attempt_status(attempt, {"running"})
            attempt.heartbeat_at = now
            run.updated_at = now
            self._db.commit()
        except Exception:
            self._db.rollback()
            raise
        self._db.refresh(attempt)
        return attempt

    def request_cancellation(self, run_id: str) -> AnalysisRun:
        """Persist a cancellation request once for a pending or running run."""
        now = _utcnow()
        try:
            run = self._require_run(run_id, for_update=True)
            self._require_run_status(run, {"pending", "running"})
            if run.cancellation_requested_at is None:
                run.cancellation_requested_at = now
                run.updated_at = now
                self._db.add(
                    self._new_event(
                        run_id=run_id,
                        attempt_id=None,
                        event_name="analysis.cancellation_requested",
                        severity="info",
                        occurred_at=now,
                    )
                )
            self._db.commit()
        except Exception:
            self._db.rollback()
            raise
        self._db.refresh(run)
        return run

    def complete_validated_attempt(
        self,
        run_id: str,
        attempt_id: str,
        result: SupervisorOutput,
    ) -> AnalysisRun:
        """Atomically publish a validated result and complete its running attempt."""
        now = _utcnow()
        try:
            run = self._require_run(run_id, for_update=True)
            self._require_run_status(run, {"running"})
            attempt = self._require_attempt(run_id, attempt_id, for_update=True)
            self._require_attempt_status(attempt, {"running"})
            duration_ms = _duration_ms(attempt.started_at, now)

            attempt.status = "done"
            attempt.finished_at = now
            attempt.heartbeat_at = now
            attempt.duration_ms = duration_ms
            run.status = "done"
            run.result = result.model_dump(mode="json")
            run.integrity_status = "passed"
            run.finished_at = now
            run.updated_at = now
            self._db.add(
                self._new_event(
                    run_id=run_id,
                    attempt_id=attempt_id,
                    event_name="analysis.completed",
                    severity="info",
                    occurred_at=now,
                    duration_ms=duration_ms,
                )
            )
            self._db.commit()
        except Exception:
            self._db.rollback()
            raise
        self._db.refresh(run)
        return run

    def fail_attempt(
        self,
        run_id: str,
        attempt_id: str,
        *,
        error_code: str,
        error_message: str,
        retryable: bool,
        integrity_failed: bool,
    ) -> AnalysisRun:
        """Atomically fail a running attempt and its logical run."""
        if not error_code.strip():
            raise ValueError("error_code must not be empty")
        if not error_message.strip():
            raise ValueError("error_message must not be empty")

        now = _utcnow()
        try:
            run = self._require_run(run_id, for_update=True)
            self._require_run_status(run, {"running"})
            attempt = self._require_attempt(run_id, attempt_id, for_update=True)
            self._require_attempt_status(attempt, {"running"})
            duration_ms = _duration_ms(attempt.started_at, now)

            attempt.status = "error"
            attempt.finished_at = now
            attempt.heartbeat_at = now
            attempt.duration_ms = duration_ms
            attempt.error_code = error_code.strip()
            attempt.error_message = error_message.strip()
            attempt.retryable = retryable
            run.status = "error"
            run.integrity_status = "failed" if integrity_failed else None
            run.finished_at = now
            run.updated_at = now
            self._db.add(
                self._new_event(
                    run_id=run_id,
                    attempt_id=attempt_id,
                    event_name="analysis.failed",
                    severity="error",
                    occurred_at=now,
                    duration_ms=duration_ms,
                    attributes={
                        "error_code": error_code.strip(),
                        "retryable": retryable,
                    },
                )
            )
            self._db.commit()
        except Exception:
            self._db.rollback()
            raise
        self._db.refresh(run)
        return run

    def cancel_attempt(self, run_id: str, attempt_id: str) -> AnalysisRun:
        """Atomically cancel a running attempt and its logical run."""
        now = _utcnow()
        try:
            run = self._require_run(run_id, for_update=True)
            self._require_run_status(run, {"running"})
            attempt = self._require_attempt(run_id, attempt_id, for_update=True)
            self._require_attempt_status(attempt, {"running"})
            duration_ms = _duration_ms(attempt.started_at, now)

            attempt.status = "cancelled"
            attempt.finished_at = now
            attempt.heartbeat_at = now
            attempt.duration_ms = duration_ms
            run.status = "cancelled"
            run.finished_at = now
            run.updated_at = now
            self._db.add(
                self._new_event(
                    run_id=run_id,
                    attempt_id=attempt_id,
                    event_name="analysis.cancelled",
                    severity="info",
                    occurred_at=now,
                    duration_ms=duration_ms,
                )
            )
            self._db.commit()
        except Exception:
            self._db.rollback()
            raise
        self._db.refresh(run)
        return run

    def _require_run(self, run_id: str, *, for_update: bool = False) -> AnalysisRun:
        statement = select(AnalysisRun).where(AnalysisRun.run_id == run_id)
        if for_update:
            statement = statement.with_for_update()
        run = self._db.scalar(statement)
        if run is None:
            raise RunNotFoundError(f"Analysis run not found: {run_id}")
        return run

    def _require_attempt(
        self,
        run_id: str,
        attempt_id: str,
        *,
        for_update: bool = False,
    ) -> AnalysisAttempt:
        statement = select(AnalysisAttempt).where(
            AnalysisAttempt.attempt_id == attempt_id,
            AnalysisAttempt.run_id == run_id,
        )
        if for_update:
            statement = statement.with_for_update()
        attempt = self._db.scalar(statement)
        if attempt is None:
            raise RunNotFoundError(
                f"Analysis attempt not found for run {run_id}: {attempt_id}"
            )
        return attempt

    @staticmethod
    def _require_run_status(run: AnalysisRun, allowed: set[str]) -> None:
        if run.status in _TERMINAL_RUN_STATUSES or run.status not in allowed:
            expected = ", ".join(sorted(allowed))
            raise RunStateError(
                f"Run {run.run_id} is {run.status}; expected one of: {expected}"
            )

    @staticmethod
    def _require_attempt_status(attempt: AnalysisAttempt, allowed: set[str]) -> None:
        if attempt.status not in ATTEMPT_STATUSES or attempt.status not in allowed:
            expected = ", ".join(sorted(allowed))
            raise RunStateError(
                f"Attempt {attempt.attempt_id} is {attempt.status}; "
                f"expected one of: {expected}"
            )

    @staticmethod
    def _new_event(
        *,
        run_id: str,
        attempt_id: str | None,
        event_name: str,
        severity: EventSeverity,
        occurred_at: datetime,
        stage: str | None = None,
        duration_ms: int | None = None,
        dependency: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> AnalysisEvent:
        return AnalysisEvent(
            run_id=run_id,
            attempt_id=attempt_id,
            occurred_at=occurred_at,
            event_name=event_name,
            stage=stage,
            severity=severity,
            duration_ms=duration_ms,
            dependency=dependency,
            attributes=attributes,
        )
