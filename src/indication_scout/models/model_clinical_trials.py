"""
Pydantic models for ClinicalTrials.gov data.

These are the data contracts between the ClinicalTrials.gov client and the agents.
Agents never see raw API responses.
"""

from pydantic import BaseModel, model_validator

# ------------------------------------------------------------------
# Trial-level models
# ------------------------------------------------------------------


class Intervention(BaseModel):
    """A drug, biological, device, or other intervention in a trial."""

    intervention_type: str = ""  # "Drug", "Biological", "Device", etc.
    intervention_name: str = ""  # e.g. "Semaglutide"
    description: str | None = None

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


class PrimaryOutcome(BaseModel):
    """A primary outcome measure for a trial."""

    measure: str = ""  # what they're measuring
    time_frame: str | None = None  # e.g. "72 weeks"

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


class Trial(BaseModel):
    """A single clinical trial record from ClinicalTrials.gov."""

    nct_id: str = ""
    title: str = ""
    brief_summary: str | None = None
    phase: str = ""  # "Phase 1", "Phase 2", "Phase 1/Phase 2", etc.
    overall_status: str = ""  # "Recruiting", "Completed", "Terminated", etc.
    why_stopped: str | None = None  # free text, only for Terminated/Withdrawn/Suspended
    conditions: list[str] = []
    interventions: list[Intervention] = []
    sponsor: str = ""
    collaborators: list[str] = []
    enrollment: int | None = None
    start_date: str | None = None
    completion_date: str | None = None
    study_type: str = "Interventional"
    primary_outcomes: list[PrimaryOutcome] = []
    results_posted: bool | None = None
    references: list[str] = []  # PMIDs

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


# ------------------------------------------------------------------
# Whitespace detection
# ------------------------------------------------------------------


class ConditionDrug(BaseModel):
    """A drug being tested for the same condition (when whitespace exists)."""

    nct_id: str = ""
    drug_name: str = ""
    condition: str = ""
    phase: str = ""
    status: str = ""

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


class WhitespaceResult(BaseModel):
    """Result of whitespace detection — is this drug-condition pair unexplored?"""

    is_whitespace: bool | None = None
    exact_match_count: int | None = None
    drug_only_trials: int | None = None
    condition_only_trials: int | None = None
    condition_drugs: list[ConditionDrug] = []

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


# ------------------------------------------------------------------
# Competitive landscape
# ------------------------------------------------------------------


class CompetitorEntry(BaseModel):
    """A sponsor + drug combination competing in a disease area."""

    sponsor: str = ""
    drug_name: str = ""
    drug_type: str | None = None
    max_phase: str = ""
    trial_count: int | None = None
    statuses: set[str] = set()
    total_enrollment: int | None = None
    most_recent_start: str | None = None

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


class RecentStart(BaseModel):
    """A trial that started recently in a condition's landscape."""

    nct_id: str = ""
    sponsor: str = ""
    drug: str = ""
    phase: str = ""

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


class ConditionLandscape(BaseModel):
    """Full competitive landscape for a condition."""

    total_trial_count: int | None = None
    competitors: list[CompetitorEntry] = []
    phase_distribution: dict[str, int] = {}
    recent_starts: list[RecentStart] = []

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


# ------------------------------------------------------------------
# Terminated trials (Critique Agent)
# ------------------------------------------------------------------


class TerminatedTrial(BaseModel):
    """A terminated, withdrawn, or suspended trial with stop classification."""

    nct_id: str = ""
    title: str = ""
    drug_name: str | None = None
    condition: str | None = None
    phase: str | None = None
    why_stopped: str | None = None
    stop_category: str | None = (
        None  # safety, efficacy, business, enrollment, other, unknown
    )
    enrollment: int | None = None
    sponsor: str | None = None
    start_date: str | None = None
    termination_date: str | None = None
    references: list[str] = []  # PMIDs

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values
