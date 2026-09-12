"""add analysis cost measurement

Revision ID: b73f04d17c82
Revises: 8d1f2a6c9b04
Create Date: 2026-09-12
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "b73f04d17c82"
down_revision: str | Sequence[str] | None = "8d1f2a6c9b04"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    for name in (
        "llm_input_tokens",
        "llm_output_tokens",
        "llm_cache_read_tokens",
        "llm_cache_write_tokens",
    ):
        op.add_column(
            "analysis_attempts", sa.Column(name, sa.BigInteger(), nullable=True)
        )
    op.add_column(
        "analysis_attempts",
        sa.Column("llm_cost_usd", sa.Numeric(16, 8), nullable=True),
    )
    op.add_column(
        "analysis_attempts",
        sa.Column("llm_overhead_cost_usd", sa.Numeric(16, 8), nullable=True),
    )
    op.add_column(
        "analysis_attempts",
        sa.Column("llm_pricing_complete", sa.Boolean(), nullable=True),
    )
    op.create_table(
        "analysis_candidate_costs",
        sa.Column(
            "candidate_cost_id", sa.BigInteger(), autoincrement=True, nullable=False
        ),
        sa.Column("run_id", sa.String(length=32), nullable=False),
        sa.Column("attempt_id", sa.String(length=32), nullable=False),
        sa.Column("candidate_name", sa.Text(), nullable=False),
        sa.Column("input_tokens", sa.BigInteger(), nullable=False),
        sa.Column("output_tokens", sa.BigInteger(), nullable=False),
        sa.Column("cache_read_tokens", sa.BigInteger(), nullable=False),
        sa.Column("cache_write_tokens", sa.BigInteger(), nullable=False),
        sa.Column("cost_usd", sa.Numeric(16, 8), nullable=True),
        sa.Column("pricing_complete", sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(
            ["attempt_id"], ["analysis_attempts.attempt_id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["run_id"], ["analysis_runs.run_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("candidate_cost_id"),
        sa.UniqueConstraint(
            "attempt_id", "candidate_name", name="uq_candidate_cost_attempt_name"
        ),
    )
    op.create_index("ix_candidate_costs_run_id", "analysis_candidate_costs", ["run_id"])


def downgrade() -> None:
    op.drop_index("ix_candidate_costs_run_id", table_name="analysis_candidate_costs")
    op.drop_table("analysis_candidate_costs")
    for name in (
        "llm_pricing_complete",
        "llm_overhead_cost_usd",
        "llm_cost_usd",
        "llm_cache_write_tokens",
        "llm_cache_read_tokens",
        "llm_output_tokens",
        "llm_input_tokens",
    ):
        op.drop_column("analysis_attempts", name)
