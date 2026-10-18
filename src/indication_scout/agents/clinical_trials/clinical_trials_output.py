from typing import Literal

from pydantic import BaseModel, Field

from indication_scout.models.model_clinical_trials import (
    ApprovalCheck,
    CompletedTrialsResult,
    IndicationLandscape,
    SearchTrialsResult,
    TerminatedTrialsResult,
)


class TrialSignals(BaseModel):
    """Machine-computed trial facts for one drug × indication pair (no verdict), from relevant trials only.

    Computed by derive_trial_signals; lives here to avoid an import cycle.
    """

    # Highest phase label among any completed (or P3-terminated-for-cause) trial, including pure Phase 4.
    # Display fact only; use has_completed_phase3 / dev_stage for tier or closure decisions.
    highest_completed_phase: str | None = None
    has_completed_phase3: bool = False
    completed_phase3_nct_ids: list[str] = []
    # Completed pure Phase 3 trials ("Phase 3" / "Phase 3/Phase 4"), excluding "Phase 2/Phase 3". If
    # has_completed_phase3 is True but this is empty, the dev-stage judge must not call it a completed pivotal Phase 3.
    has_completed_pure_phase3: bool = False
    completed_pure_phase3_nct_ids: list[str] = []
    # An active/recruiting Phase 3 (>= Phase 2/Phase 3) on the all-status search set; blocks a false
    # "Phase 4 only" claim. Contamination drop is best-effort.
    has_active_phase3: bool = False
    active_phase3_nct_ids: list[str] = []
    # A Phase 3 terminated for a non-operational reason: the positive closure signal; the sub-agent decides closure.
    phase3_terminated_for_cause: bool = False
    terminated_phase3_nct_ids: list[str] = []
    # Development-stage tier and the "what is still moving" line, both from judge_dev_stage over the relevant
    # trials; the supervisor renders them verbatim. dev_stage is one of: phase3_terminated_for_cause,
    # completed_phase3, active_phase3, phase3_unknown_status, completed_phase2, exploratory_phase4_only,
    # early_phase, untested. active_programs is free text or "None active".
    dev_stage: str = "untested"
    active_programs: str = "None active"


class TrialRelevanceCoverage(BaseModel):
    """Coverage and relevance counts for one ClinicalTrials.gov query scope."""

    registry_query_matches: int
    retrieved_records: int
    classified_records: int
    relevant_records: int
    contaminated_records: int
    unreviewed_records: int
    coverage_complete: bool
    relevant_by_status: dict[str, int]


class ClinicalTrialsOutput(BaseModel):
    """Final assembled output from a single clinical trials agent run."""

    search: SearchTrialsResult | None = Field(
        default=None,
        description=(
            "All-status trial query for the pair: total + per-status counts "
            "(recruiting / active / withdrawn) + the top 50 trials by enrollment "
            "and every ongoing trial."
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

    search_coverage: TrialRelevanceCoverage | None = Field(
        default=None,
        description="Relevance coverage for the all-status registry query.",
    )

    completed_coverage: TrialRelevanceCoverage | None = Field(
        default=None,
        description="Relevance coverage for the completed-trial registry query.",
    )

    terminated_coverage: TrialRelevanceCoverage | None = Field(
        default=None,
        description="Relevance coverage for the terminated-trial registry query.",
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
        description=(
            "Trial-section prose, authored post-loop by services.clinical_trials_summary fed "
            "the resolved dev_stage (so it cannot contradict the tier). Empty when the "
            "synthesis call returned None."
        ),
    )

    closure: Literal["live", "closed", "unknown"] = Field(
        default="unknown",
        description=(
            "Typed live-vs-closed verdict from the same synthesis call. The supervisor "
            "consumes this directly and does NOT re-judge closure."
        ),
    )

    closure_reason: str = Field(
        default="",
        description="One-sentence justification for the closure verdict.",
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
    """Artifact returned by finalize_analysis: the agent's relevance split, harvested by run_clinical_trials_agent.

    Trial-section prose is written post-loop by services.clinical_trials_summary.
    """

    relevant_ncts: list[str] = []
    contaminated_ncts: list[str] = []
    relevance_reasoning: str = ""
