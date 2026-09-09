"""
Pydantic models for ClinicalTrials.gov data.

These are the data contracts between the ClinicalTrials.gov client and the agents.
Agents never see raw API responses.
"""

from typing import Literal

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
# Per-pair trial query results (count + reviewed records)
# ------------------------------------------------------------------


class SearchTrialsResult(BaseModel):
    """All-status trial query for a drug × indication pair.

    `total_count` is the exact number of registry query matches (via countTotal), before the
    downstream relevance review. `by_status` carries query-match counts for RECRUITING,
    ACTIVE_NOT_RECRUITING, WITHDRAWN, and UNKNOWN. TERMINATED and COMPLETED
    counts live on TerminatedTrialsResult and CompletedTrialsResult to avoid
    double-counting. `trials` is the deduplicated union of the top 50 by enrollment and
    every ongoing trial for the agent to inspect. `resolution_status` distinguishes a
    resolved query with zero matches from a query that could not be run because disease
    normalization failed. None means the provenance is unavailable, as with an older cache.
    """

    total_count: int = 0
    by_status: dict[str, int] = {}
    trials: list[Trial] = []
    resolution_status: Literal["resolved", "unresolved"] | None = None

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


class CompletedTrialsResult(BaseModel):
    """Status=COMPLETED trial query for a drug × indication pair.

    `total_count` is the completed-scope registry query-match count before relevance review.
    `trials` is the top 50 by enrollment; the agent reads phase information off each trial.
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

    `total_count` is the terminated-scope registry query-match count before relevance review.
    `trials` is the top 50 by enrollment, each carrying `why_stopped` text. Stop-category
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
# Relevance verdicts
# ------------------------------------------------------------------


class TrialVerdict(BaseModel):
    """One trial's relevance verdict, as returned by the clinical-trials agent's finalize call.

    `drug_role` is the part the pinned target drug plays in THIS trial, judged from its title,
    interventions and summary. Only "studied" can carry a "relevant" verdict — a comparator,
    background or absent drug makes the trial someone else's evidence. Asking for the role
    rather than a drug name keeps the check off string matching, which cannot separate
    "metformin" from "Dapagliflozin/Metformin".

    All three fields are required: an unparseable entry is rejected at finalize rather than
    silently dropping the trial from both the relevant and contaminated sets.
    """

    nct: str
    drug_role: Literal["studied", "comparator", "background", "absent"]
    verdict: Literal["relevant", "contaminated"]


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
