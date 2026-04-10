"""Structured output from the supervisor agent."""

from pydantic import BaseModel, Field

from indication_scout.agents.clinical_trials.clinical_trials_output import (
    ClinicalTrialsOutput,
)
from indication_scout.agents.literature.literature_output import LiteratureOutput
from indication_scout.agents.mechanism.mechanism_output import MechanismOutput


class CandidateFindings(BaseModel):
    """Per-candidate findings aggregated by the supervisor."""

    disease: str
    literature: LiteratureOutput | None = None
    clinical_trials: ClinicalTrialsOutput | None = None


class SupervisorOutput(BaseModel):
    """Final assembled output from a supervisor run."""

    drug_name: str = ""
    candidates: list[str] = Field(
        default_factory=list,
        description="Candidate diseases surfaced for the drug.",
    )
    mechanism: MechanismOutput | None = Field(
        default=None,
        description="Molecular target analysis from the mechanism agent.",
    )
    findings: list[CandidateFindings] = Field(
        default_factory=list,
        description="Per-candidate analyses from the sub-agents.",
    )
    summary: str = Field(
        default="",
        description="Supervisor's narrative summary of the most promising candidates.",
    )
