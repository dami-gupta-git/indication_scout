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
    # in the same class. Set by the combined synthesize call (the single author over the abstracts).
    # "class_level" means the disease-relevant RCTs are for sibling drugs, not this one — the
    # strength is then forced to NOT "strong" by the deterministic cap in services/retrieval.py.
    # "approved" means the only
    # relevant this-drug evidence studies an APPROVED sub-indication of a broad candidate (already
    # approved, not repurposing) — strength is also forced to none. Renderers surface the basis so
    # a card never claims "strong, RCT-backed" for class-level-only or already-approved evidence.
    evidence_basis: Literal["drug_specific", "approved", "class_level", "none"] = "none"
    # True when the relevant evidence includes at least one RCT/controlled trial, False when
    # it is purely observational, None when undetermined (no-data). Lets the supervisor avoid
    # calling RCT-backed evidence "observational". None stays None (no default coercion).
    is_observational: bool | None = None
    key_findings: list[str] = []
    supporting_pmids: list[str] = []
    contradicting_pmids: list[str] = []
    # Per-abstract relevance split from the combined synthesize+judge call. relevant_pmids are the
    # abstracts graded as this-drug-this-disease evidence (the set strength/direction grade over);
    # contaminated_pmids are the excluded ones (other-drug, off-disease, approved sub-indication,
    # therapeutic-intent mismatch). Lets renderers show an "N excluded" note like the trial gate.
    relevant_pmids: list[str] = []
    contaminated_pmids: list[str] = []

    @field_validator(
        "supporting_pmids",
        "contradicting_pmids",
        "relevant_pmids",
        "contaminated_pmids",
        mode="before",
    )
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
