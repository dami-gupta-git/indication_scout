"""SQLAlchemy models for durable analysis-run lifecycle records."""

from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from indication_scout.db.base import Base

RUN_STATUSES = ("pending", "running", "done", "error", "cancelled")
EXECUTION_MODES = ("live", "seed")
ATTEMPT_STATUSES = ("running", "done", "error", "cancelled", "interrupted")
INTEGRITY_STATUSES = ("passed", "failed")
EVENT_SEVERITIES = ("debug", "info", "warning", "error", "critical")


class AnalysisRun(Base):
    """One logical analysis request exposed through the API."""

    __tablename__ = "analysis_runs"
    __table_args__ = (
        CheckConstraint(f"status IN {RUN_STATUSES}", name="ck_analysis_runs_status"),
        CheckConstraint(
            f"execution_mode IN {EXECUTION_MODES}",
            name="ck_analysis_runs_execution_mode",
        ),
        CheckConstraint(
            f"integrity_status IN {INTEGRITY_STATUSES}",
            name="ck_analysis_runs_integrity_status",
        ),
        Index("ix_analysis_runs_status", "status"),
        Index("ix_analysis_runs_created_at", "created_at"),
    )

    run_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    drug_name: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    execution_mode: Mapped[str] = mapped_column(String(20), nullable=False)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    integrity_status: Mapped[str | None] = mapped_column(String(20), nullable=True)
    cancellation_requested_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class AnalysisAttempt(Base):
    """One execution attempt belonging to a logical analysis run."""

    __tablename__ = "analysis_attempts"
    __table_args__ = (
        CheckConstraint(
            f"status IN {ATTEMPT_STATUSES}",
            name="ck_analysis_attempts_status",
        ),
        UniqueConstraint(
            "run_id", "attempt_number", name="uq_analysis_attempts_run_number"
        ),
        Index("ix_analysis_attempts_run_id", "run_id"),
        Index("ix_analysis_attempts_heartbeat_at", "heartbeat_at"),
    )

    attempt_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        String(32),
        ForeignKey("analysis_runs.run_id", ondelete="CASCADE"),
        nullable=False,
    )
    attempt_number: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    worker_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    deployment_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    release: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    heartbeat_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    duration_ms: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(100), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    retryable: Mapped[bool | None] = mapped_column(Boolean, nullable=True)


class AnalysisEvent(Base):
    """One append-only lifecycle or operational event for an analysis run."""

    __tablename__ = "analysis_events"
    __table_args__ = (
        CheckConstraint(
            f"severity IN {EVENT_SEVERITIES}",
            name="ck_analysis_events_severity",
        ),
        Index("ix_analysis_events_run_occurred", "run_id", "occurred_at"),
        Index("ix_analysis_events_attempt_id", "attempt_id"),
    )

    event_id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True
    )
    run_id: Mapped[str] = mapped_column(
        String(32),
        ForeignKey("analysis_runs.run_id", ondelete="CASCADE"),
        nullable=False,
    )
    attempt_id: Mapped[str | None] = mapped_column(
        String(32),
        ForeignKey("analysis_attempts.attempt_id", ondelete="CASCADE"),
        nullable=True,
    )
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    event_name: Mapped[str] = mapped_column(String(100), nullable=False)
    stage: Mapped[str | None] = mapped_column(String(100), nullable=True)
    severity: Mapped[str] = mapped_column(String(20), nullable=False)
    duration_ms: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    dependency: Mapped[str | None] = mapped_column(String(100), nullable=True)
    attributes: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
