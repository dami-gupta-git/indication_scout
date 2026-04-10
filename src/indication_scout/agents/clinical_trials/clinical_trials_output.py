from pydantic import BaseModel, Field

from indication_scout.models.model_clinical_trials import (
    WhitespaceResult,
    Trial,
    IndicationLandscape,
    TerminatedTrial,
)


class ClinicalTrialsOutput(BaseModel):
    """Final assembled output from a single clinical trials agent run."""

    whitespace: WhitespaceResult | None = Field(
        default=None,
        description=(
            "Whitespace analysis: trial counts and competing drugs for the indication."
        ),
    )

    landscape: IndicationLandscape | None = Field(
        default=None,
        description=("Competitive landscape for the indication."),
    )

    trials: list[Trial] = Field(
        default_factory=list,
        description=("Trials matching the drug-indication pair."),
    )

    terminated: list[TerminatedTrial] = Field(
        default_factory=list,
        description=("Terminated, withdrawn, or suspended trials."),
    )

    summary: str = Field(
        default="",
        description="LLM narrative summary from the final agent message.",
    )
