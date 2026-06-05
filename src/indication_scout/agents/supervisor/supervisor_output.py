"""Structured output from the supervisor agent."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from indication_scout.agents.clinical_trials.clinical_trials_output import (
    ClinicalTrialsOutput,
)
from indication_scout.agents.literature.literature_output import LiteratureOutput
from indication_scout.agents.mechanism.mechanism_output import MechanismOutput


class CandidateBlurb(BaseModel):
    """Structured per-candidate synthesis written by the supervisor.

    Single-line fields plus a 2-sentence prose synthesis. Fields are independently optional — an
    empty string means nothing to write for that field this run, and the formatter omits empty
    fields. Populated only for the top 5 ranked candidates in production runs; None in holdout runs
    and for un-ranked candidates.
    """

    stage: str = Field(
        default="",
        description=(
            "Where the (drug, indication) pair sits in development, e.g. "
            "'Post-Phase 3, regulatory-stalled' or 'Rationale-only, no trials'."
        ),
    )
    literature: str = Field(
        default="",
        description=(
            "Single-line summary of the literature evidence base: strength tag "
            "plus the shape of the evidence, e.g. 'Strong, 5 RCTs / meta-"
            "analyses' or 'Weak, case reports only' or 'None'. Mirrors the "
            "per-indication Literature section's strength + study_count."
        ),
    )
    blocker: str = Field(
        default="",
        description=(
            "What is currently holding the program back, if anything, e.g. "
            "'Regulatory negotiation, not efficacy'."
        ),
    )
    active_programs: str = Field(
        default="",
        description=(
            "Short summary of what is still moving, e.g. "
            "'5 recruiting/active, incl. pediatric'."
        ),
    )
    key_risk: str = Field(
        default="",
        description=(
            "The single biggest risk to the repurposing hypothesis, e.g. "
            "'Regulatory gap may not be closable without new data'."
        ),
    )
    verdict: str = Field(
        default="",
        description=(
            "Interpretive one-tag verdict, e.g. 'Live but bottlenecked', "
            "'Untested', 'Closed signal'."
        ),
    )
    watch: str = Field(
        default="",
        description=(
            "Next concrete data readout or trial worth watching (NCT id and/or "
            "expected timing if known). Empty when no scheduled readout exists."
        ),
    )
    prose: str = Field(
        default="",
        description=(
            "Exactly 2-sentence interpretive synthesis of the literature and "
            "clinical-trials sub-agent summaries for this disease."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values):
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


class CandidateFindings(BaseModel):
    """Per-candidate findings aggregated by the supervisor."""

    disease: str
    source: Literal["competitor", "mechanism", "both"] = "competitor"
    literature: LiteratureOutput | None = None
    clinical_trials: ClinicalTrialsOutput | None = None
    blurb: CandidateBlurb | None = Field(
        default=None,
        description=(
            "Structured supervisor-written synthesis of the literature and "
            "clinical-trials sub-agent summaries for this disease. Populated "
            "only for the supervisor's top 5 ranked candidates in production "
            "runs; None in holdout runs and for un-ranked candidates."
        ),
    )


class SupervisorOutput(BaseModel):
    """Final assembled output from a supervisor run."""

    drug_name: str = ""
    candidate_diseases: list[str] = Field(
        default_factory=list,
        description="Candidate diseases surfaced for the drug.",
    )
    mechanism: MechanismOutput | None = Field(
        default=None,
        description="Molecular target analysis from the mechanism agent.",
    )
    disease_findings: list[CandidateFindings] = Field(
        default_factory=list,
        description=(
            "Per-disease analyses from the sub-agents. Ordered: top_diseases "
            "first in rank order, then any other investigated diseases in "
            "insertion order."
        ),
    )
    top_diseases: list[str] = Field(
        default_factory=list,
        description=(
            "Ranked top diseases (max 5) selected by the supervisor for the "
            "Summary section. Strict subset of disease_findings."
        ),
    )
    summary: str = Field(
        default="",
        description="Supervisor's narrative summary of the most promising candidates.",
    )
