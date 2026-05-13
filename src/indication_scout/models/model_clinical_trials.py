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


class MeshTerm(BaseModel):
    """A MeSH term from ClinicalTrials.gov's derived conditionBrowseModule."""

    id: str = ""  # e.g. "D003924"
    term: str = ""  # e.g. "Diabetes Mellitus, Type 2"

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
    mesh_conditions: list[MeshTerm] = []
    mesh_ancestors: list[MeshTerm] = []
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
# Per-pair trial query results (count + top-50 exemplars)
# ------------------------------------------------------------------


class SearchTrialsResult(BaseModel):
    """All-status trial query for a drug × indication pair.

    `total_count` is the exact number of trials matching the pair (via
    countTotal). `by_status` carries per-status counts for RECRUITING,
    ACTIVE_NOT_RECRUITING, WITHDRAWN, and UNKNOWN. TERMINATED and COMPLETED
    counts live on TerminatedTrialsResult and CompletedTrialsResult to avoid
    double-counting. `trials` is the top 50 by enrollment for the agent
    to inspect.
    """

    total_count: int = 0
    by_status: dict[str, int] = {}
    trials: list[Trial] = []

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


class CompletedTrialsResult(BaseModel):
    """Status=COMPLETED trial query for a drug × indication pair.

    `total_count` is all completed trials for the pair. `trials` is the top
    50 by enrollment; the agent reads phase information off each trial.
    """

    total_count: int = 0
    trials: list[Trial] = []

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


class TerminatedTrialsResult(BaseModel):
    """Status=TERMINATED trial query for a drug × indication pair.

    `total_count` is all terminated trials for the pair. `trials` is the
    top 50 by enrollment, each carrying `why_stopped` text. Stop-category
    classification is derived on read at the tool layer (no separate
    field stored).
    """

    total_count: int = 0
    trials: list[Trial] = []

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
    most_recent_start: str | None = None  # ISO date of latest trial start

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
# FDA approval check
# ------------------------------------------------------------------


class ApprovalCheck(BaseModel):
    """Result of an FDA-label lookup for a drug × indication pair.

    `is_approved` is True when the indication appears on a current
    FDA label for any known name of the drug. When False it means
    "not found on FDA labels" — it does not distinguish trial failure
    from approval pending from approval outside the US.

    `label_found` is True when FDA returned at least one label for any
    of the drug names checked. When False, no label exists in openFDA
    for this drug (e.g. withdrawn drugs like aducanumab after 2024) —
    approval status cannot be determined from available data.
    """

    is_approved: bool = False
    label_found: bool = False
    matched_indication: str | None = None
    drug_names_checked: list[str] = []

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values
