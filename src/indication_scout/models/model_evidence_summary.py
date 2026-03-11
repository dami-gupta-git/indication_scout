"""Pydantic model for the synthesized evidence summary produced by the RAG pipeline."""

from typing import Any, Literal

from pydantic import BaseModel, field_validator, model_validator


class EvidenceSummary(BaseModel):
    summary: str = ""
    study_count: int = 0
    study_types: list[str] = []
    strength: Literal["strong", "moderate", "weak", "none"] = "none"
    has_adverse_effects: bool = False
    key_findings: list[str] = []
    supporting_pmids: list[str] = []

    @field_validator("supporting_pmids", mode="before")
    @classmethod
    def coerce_pmids_to_str(cls, v: Any) -> list[str]:
        if isinstance(v, list):
            return [str(item) for item in v]
        return v

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values):
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values
