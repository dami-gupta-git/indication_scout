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
    indications: list[str] = []
    interventions: list[Intervention] = []
    sponsor: str = ""
    enrollment: int | None = None
    start_date: str | None = None
    completion_date: str | None = None
    primary_outcomes: list[PrimaryOutcome] = []
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


class IndicationDrug(BaseModel):
    """A drug being tested for the same indication (when whitespace exists)."""

    nct_id: str = ""
    drug_name: str = ""
    indication: str = ""
    phase: str = ""
    status: str = ""

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values

    @classmethod
    def from_trial(cls, trial: Trial, drug_name: str) -> "IndicationDrug":
        """Build an IndicationDrug from a Trial and its primary drug name."""
        return cls(
            nct_id=trial.nct_id,
            drug_name=drug_name,
            indication=trial.indications[0] if trial.indications else "",
            phase=trial.phase,
            status=trial.overall_status,
        )


class WhitespaceResult(BaseModel):
    """Result of whitespace detection — is this drug-indication pair unexplored?"""

    is_whitespace: bool | None = None
    no_data: bool | None = None
    exact_match_count: int | None = None
    drug_only_trials: int | None = None
    indication_only_trials: int | None = None
    indication_drugs: list[IndicationDrug] = []

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
    trial_count: int = 0
    statuses: set[str] = set()
    total_enrollment: int = 0

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


class RecentStart(BaseModel):
    """A trial that started recently in an indication's landscape."""

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


class IndicationLandscape(BaseModel):
    """Full competitive landscape for an indication."""

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
    indication: str | None = None
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
