"""Pydantic model for the synthesized evidence summary produced by the RAG pipeline."""

from typing import Any, Literal

from pydantic import BaseModel, field_validator, model_validator


class EvidenceSummary(BaseModel):
    summary: str = ""
    study_count: int = 0
    # strength = evidence QUANTITY/QUALITY only (how much, how good), independent of
    # whether it supports or contradicts. direction = which way it points.
    strength: Literal["strong", "moderate", "weak", "none"] = "none"
    direction: Literal["supports", "contradicts", "mixed", "none"] = "none"
    # Whether the strength/direction above grade evidence FOR THIS DRUG or only for other drugs
    # in the same class. Set by the isolated judge_literature_strength call (authoritative).
    # "class_level" means the disease-relevant RCTs are for sibling drugs, not this one — the
    # strength is then NOT "strong" (services/literature_strength.py). Renderers surface the
    # basis so a card never claims "strong, RCT-backed" for class-level-only evidence.
    evidence_basis: Literal["drug_specific", "class_level", "none"] = "none"
    # True when the relevant evidence includes at least one RCT/controlled trial, False when
    # it is purely observational, None when undetermined (no-data). Lets the supervisor avoid
    # calling RCT-backed evidence "observational". None stays None (no default coercion).
    is_observational: bool | None = None
    key_findings: list[str] = []
    supporting_pmids: list[str] = []
    contradicting_pmids: list[str] = []

    @field_validator("supporting_pmids", "contradicting_pmids", mode="before")
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
