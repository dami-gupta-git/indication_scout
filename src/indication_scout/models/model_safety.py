"""Typed safety-analysis contracts."""

from typing import Literal

from pydantic import BaseModel, model_validator


class SafetyPaperVerdict(BaseModel):
    """One disease-scoped abstract's indication-harm adjudication."""

    pmid: str
    status: Literal["confirmed_harm", "safety_assessed_only", "irrelevant", "unclear"]
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
