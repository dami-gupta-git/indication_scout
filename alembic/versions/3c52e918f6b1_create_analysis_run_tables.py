"""create analysis run tables

Revision ID: 3c52e918f6b1
Revises: f0ccb024a181
Create Date: 2026-09-10

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "3c52e918f6b1"
down_revision: str | Sequence[str] | None = "f0ccb024a181"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create durable analysis run, attempt, and event tables."""
    op.create_table(
        "analysis_runs",
        sa.Column("run_id", sa.String(length=32), nullable=False),
        sa.Column("drug_name", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("execution_mode", sa.String(length=20), nullable=False),
        sa.Column("result", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("integrity_status", sa.String(length=20), nullable=True),
        sa.Column(
            "cancellation_requested_at", sa.DateTime(timezone=True), nullable=True
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'done', 'error', 'cancelled')",
            name="ck_analysis_runs_status",
        ),
        sa.CheckConstraint(
            "execution_mode IN ('live', 'seed')",
            name="ck_analysis_runs_execution_mode",
        ),
        sa.CheckConstraint(
            "integrity_status IN ('passed', 'failed')",
            name="ck_analysis_runs_integrity_status",
        ),
        sa.PrimaryKeyConstraint("run_id"),
    )
    op.create_index(
        "ix_analysis_runs_status", "analysis_runs", ["status"], unique=False
    )
    op.create_index(
        "ix_analysis_runs_created_at",
        "analysis_runs",
        ["created_at"],
        unique=False,
    )

    op.create_table(
        "analysis_attempts",
        sa.Column("attempt_id", sa.String(length=32), nullable=False),
        sa.Column("run_id", sa.String(length=32), nullable=False),
        sa.Column("attempt_number", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("worker_id", sa.Text(), nullable=True),
        sa.Column("deployment_id", sa.Text(), nullable=True),
        sa.Column("release", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("duration_ms", sa.BigInteger(), nullable=True),
        sa.Column("error_code", sa.String(length=100), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("retryable", sa.Boolean(), nullable=True),
        sa.CheckConstraint(
            "status IN ('running', 'done', 'error', 'cancelled', 'interrupted')",
            name="ck_analysis_attempts_status",
        ),
        sa.ForeignKeyConstraint(
            ["run_id"], ["analysis_runs.run_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("attempt_id"),
        sa.UniqueConstraint(
            "run_id", "attempt_number", name="uq_analysis_attempts_run_number"
        ),
    )
    op.create_index(
        "ix_analysis_attempts_run_id",
        "analysis_attempts",
        ["run_id"],
        unique=False,
    )
    op.create_index(
        "ix_analysis_attempts_heartbeat_at",
        "analysis_attempts",
        ["heartbeat_at"],
        unique=False,
    )

    op.create_table(
        "analysis_events",
        sa.Column("event_id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.String(length=32), nullable=False),
        sa.Column("attempt_id", sa.String(length=32), nullable=True),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("event_name", sa.String(length=100), nullable=False),
        sa.Column("stage", sa.String(length=100), nullable=True),
        sa.Column("severity", sa.String(length=20), nullable=False),
        sa.Column("duration_ms", sa.BigInteger(), nullable=True),
        sa.Column("dependency", sa.String(length=100), nullable=True),
        sa.Column("attributes", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.CheckConstraint(
            "severity IN ('debug', 'info', 'warning', 'error', 'critical')",
            name="ck_analysis_events_severity",
        ),
        sa.ForeignKeyConstraint(
            ["attempt_id"],
            ["analysis_attempts.attempt_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["run_id"], ["analysis_runs.run_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("event_id"),
    )
    op.create_index(
        "ix_analysis_events_attempt_id",
        "analysis_events",
        ["attempt_id"],
        unique=False,
    )
    op.create_index(
        "ix_analysis_events_run_occurred",
        "analysis_events",
        ["run_id", "occurred_at"],
        unique=False,
    )


def downgrade() -> None:
    """Drop durable analysis run, attempt, and event tables."""
    op.drop_index("ix_analysis_events_run_occurred", table_name="analysis_events")
    op.drop_index("ix_analysis_events_attempt_id", table_name="analysis_events")
    op.drop_table("analysis_events")
    op.drop_index("ix_analysis_attempts_heartbeat_at", table_name="analysis_attempts")
    op.drop_index("ix_analysis_attempts_run_id", table_name="analysis_attempts")
    op.drop_table("analysis_attempts")
    op.drop_index("ix_analysis_runs_created_at", table_name="analysis_runs")
    op.drop_index("ix_analysis_runs_status", table_name="analysis_runs")
    op.drop_table("analysis_runs")
