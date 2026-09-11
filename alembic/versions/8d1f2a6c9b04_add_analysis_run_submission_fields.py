"""add analysis run submission fields

Revision ID: 8d1f2a6c9b04
Revises: 3c52e918f6b1
Create Date: 2026-09-11

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8d1f2a6c9b04"
down_revision: str | Sequence[str] | None = "3c52e918f6b1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add source, analysis kind, disease, and temporal-cutoff fields."""
    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM analysis_runs) THEN
                RAISE EXCEPTION
                    'analysis_runs contains rows that require explicit source and kind classification';
            END IF;
        END
        $$
        """)
    op.add_column(
        "analysis_runs",
        sa.Column("disease_name", sa.Text(), nullable=True),
    )
    op.add_column(
        "analysis_runs",
        sa.Column("submission_source", sa.String(length=20), nullable=False),
    )
    op.add_column(
        "analysis_runs",
        sa.Column("analysis_kind", sa.String(length=20), nullable=False),
    )
    op.add_column(
        "analysis_runs",
        sa.Column("date_before", sa.Date(), nullable=True),
    )
    op.create_check_constraint(
        "ck_analysis_runs_submission_source",
        "analysis_runs",
        "submission_source IN ('api', 'cli')",
    )
    op.create_check_constraint(
        "ck_analysis_runs_analysis_kind",
        "analysis_runs",
        "analysis_kind IN ('find', 'investigate')",
    )
    op.create_check_constraint(
        "ck_analysis_runs_disease_for_kind",
        "analysis_runs",
        "(analysis_kind = 'find' AND disease_name IS NULL) OR "
        "(analysis_kind = 'investigate' AND disease_name IS NOT NULL)",
    )


def downgrade() -> None:
    """Remove analysis-run submission fields."""
    op.drop_constraint(
        "ck_analysis_runs_disease_for_kind",
        "analysis_runs",
        type_="check",
    )
    op.drop_constraint(
        "ck_analysis_runs_analysis_kind",
        "analysis_runs",
        type_="check",
    )
    op.drop_constraint(
        "ck_analysis_runs_submission_source",
        "analysis_runs",
        type_="check",
    )
    op.drop_column("analysis_runs", "date_before")
    op.drop_column("analysis_runs", "analysis_kind")
    op.drop_column("analysis_runs", "submission_source")
    op.drop_column("analysis_runs", "disease_name")
