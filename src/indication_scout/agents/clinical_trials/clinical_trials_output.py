from pydantic import BaseModel, Field

from indication_scout.models.model_clinical_trials import (
    ApprovalCheck,
    CompletedTrialsResult,
    IndicationLandscape,
    SearchTrialsResult,
    TerminatedTrialsResult,
)


class ClinicalTrialsOutput(BaseModel):
    """Final assembled output from a single clinical trials agent run."""

    search: SearchTrialsResult | None = Field(
        default=None,
        description=(
            "All-status trial query for the pair: total + per-status counts "
            "(recruiting / active / withdrawn) + top 50 trials by enrollment."
        ),
    )

    completed: CompletedTrialsResult | None = Field(
        default=None,
        description=(
            "COMPLETED trial query for the pair: total + Phase 3 count + "
            "top 50 trials by enrollment."
        ),
    )

    terminated: TerminatedTrialsResult | None = Field(
        default=None,
        description=(
            "TERMINATED trial query for the pair: total + top 50 trials by "
            "enrollment. Stop-category counts are computed at the tool layer."
        ),
    )

    landscape: IndicationLandscape | None = Field(
        default=None,
        description="Competitive landscape for the indication.",
    )

    approval: ApprovalCheck | None = Field(
        default=None,
        description=(
            "FDA-label approval status for the drug × indication pair. Populated "
            "when the agent calls check_fda_approval."
        ),
    )

    summary: str = Field(
        default="",
        description="LLM narrative summary from the final agent message.",
    )
