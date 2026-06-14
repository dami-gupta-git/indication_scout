from pydantic import BaseModel, Field

from indication_scout.models.model_clinical_trials import (
    ApprovalCheck,
    CompletedTrialsResult,
    IndicationLandscape,
    SearchTrialsResult,
    TerminatedTrialsResult,
)


class TrialSignals(BaseModel):
    """Machine-computed trial facts for a single drug × indication pair (no verdict).

    Computed from RELEVANT trials only by derive_trial_signals (agents/_trial_signals).
    Lives here so ClinicalTrialsOutput can store it without an import cycle.
    """

    highest_completed_phase: str | None = None
    has_completed_phase3: bool = False
    completed_phase3_nct_ids: list[str] = []
    # A true Phase 3 terminated for a non-operational reason — the positive closure
    # signal. The sub-agent decides whether this actually closes the candidate.
    phase3_terminated_for_cause: bool = False
    terminated_phase3_nct_ids: list[str] = []


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

    relevant_nct_ids: list[str] = Field(
        default_factory=list,
        description=(
            "NCT ids the agent judged RELEVANT to this drug × indication pair "
            "(MeSH conditions cover this indication or a clinically overlapping form)."
        ),
    )

    contaminated_nct_ids: list[str] = Field(
        default_factory=list,
        description=(
            "NCT ids the agent judged CONTAMINATION — a distinct disease pulled in by "
            "the recall-first search, or a different drug's trial. Excluded from signals."
        ),
    )

    relevance_reasoning: str = Field(
        default="",
        description="1-2 sentence justification for the relevance split.",
    )

    signals: TrialSignals | None = Field(
        default=None,
        description=(
            "Deterministic trial facts (highest phase, completed Phase 3, relevant "
            "Phase 3 terminated for cause) computed from RELEVANT trials only."
        ),
    )


class FinalizeClinicalTrialsArtifact(BaseModel):
    """Artifact returned by the finalize_analysis tool.

    Carries the human-report prose summary plus the agent's structured relevance split,
    harvested into ClinicalTrialsOutput by run_clinical_trials_agent.
    """

    summary: str = ""
    relevant_ncts: list[str] = []
    contaminated_ncts: list[str] = []
    relevance_reasoning: str = ""
