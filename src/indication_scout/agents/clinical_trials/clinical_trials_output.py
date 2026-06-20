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
    """Machine-computed trial facts for a single drug × indication pair (no verdict).

    Computed from RELEVANT trials only by derive_trial_signals (agents/_trial_signals).
    Lives here so ClinicalTrialsOutput can store it without an import cycle.
    """

    # Highest phase label among ANY completed (or P3-terminated-for-cause) trial — including
    # pure Phase 4, which ranks above Phase 3. This is a DISPLAY fact ("highest completed, any
    # kind"), NOT a pivotal-evidence signal: it can read "Phase 4" while has_completed_phase3 is
    # False. Do NOT make tier/closure decisions on it — use has_completed_phase3 / dev_stage.
    highest_completed_phase: str | None = None
    has_completed_phase3: bool = False
    completed_phase3_nct_ids: list[str] = []
    # Subset of completed_phase3 that are PURE Phase 3 ("Phase 3" / "Phase 3/Phase 4"), excluding
    # combined "Phase 2/Phase 3". A completed Phase 2/3 trial is NOT a completed pivotal Phase 3
    # readout on its own — when has_completed_phase3 is True but this is empty, the "Phase 3
    # completed" stage came only from a Phase 2/3 trial and the dev-stage judge must not call it a
    # completed pivotal Phase 3 (prefer active_phase3 framing if active pure Phase 3 exists).
    has_completed_pure_phase3: bool = False
    completed_pure_phase3_nct_ids: list[str] = []
    # An active/recruiting Phase 3 (>= Phase 2/Phase 3 floor) on the all-status search set.
    # The recruiting-pipeline analog of has_completed_phase3 — blocks a false "no
    # development program / Phase 4 only" claim. Best-effort contamination drop only.
    has_active_phase3: bool = False
    active_phase3_nct_ids: list[str] = []
    # A true Phase 3 terminated for a non-operational reason — the positive closure
    # signal. The sub-agent decides whether this actually closes the candidate.
    phase3_terminated_for_cause: bool = False
    terminated_phase3_nct_ids: list[str] = []
    # Single authoritative development-stage tier + the "what is still moving" line, both from
    # the isolated LLM judgment (services/dev_stage.judge_dev_stage) over the relevant trials.
    # The supervisor renders these verbatim in the blurb stage / active_programs — it does NOT
    # re-author them. dev_stage is one of: phase3_terminated_for_cause, completed_phase3,
    # active_phase3, phase3_unknown_status, completed_phase2, exploratory_phase4_only,
    # early_phase, untested. active_programs is a free-text line or "None active".
    dev_stage: str = "untested"
    active_programs: str = "None active"


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
    """Artifact returned by the finalize_analysis tool.

    Carries the agent's structured relevance split, harvested into ClinicalTrialsOutput by
    run_clinical_trials_agent. The trial-section prose is NO LONGER authored here — it is
    written post-loop by services.clinical_trials_summary fed the resolved dev_stage.
    """

    relevant_ncts: list[str] = []
    contaminated_ncts: list[str] = []
    relevance_reasoning: str = ""
