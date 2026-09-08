"""Typed safety-analysis contracts."""

from typing import Literal

from pydantic import BaseModel, model_validator


class SafetyPaperVerdict(BaseModel):
    """One disease-scoped abstract's indication-harm adjudication."""

    pmid: str
    status: Literal["confirmed_harm", "safety_assessed_only", "irrelevant", "unclear"]
    # Only the disease-scoped harm classifier asks for this; the holdout drug-level prompt
    # does not, and an absent answer is treated there as "not established as patient harm".
    study_subjects: (
        Literal["patients", "animals", "cells_or_tissue", "unclear"] | None
    ) = None
    adverse_outcome: str | None = None
    evidence_quote: str | None = None

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


class DrugSafetyAssessment(BaseModel):
    """Source-separated drug-wide safety facts for report consumers."""

    regulatory_summary: str
    regulatory_full_labels: str = ""
    pharmacovigilance_summary: str
    literature_summary: str
    safety_summary: str
    safety_pmids: list[str]
    safety_severity: (
        Literal["withdrawn", "black_box", "serious", "moderate", "none"] | None
    ) = None
    label_data_available: bool | None = None

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values
