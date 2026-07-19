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
    # True when EVERY relevant drug-specific study is a non-human model (animal/in-vitro), False when
    # at least one relevant study is in humans, None when there is no relevant drug-specific evidence
    # to grade (no-data). Lets the ranking critic demote a candidate whose only support is animal
    # data. None stays None (no default coercion).
    is_animal_only: bool | None = None
    key_findings: list[str] = []
    # supporting/contradicting/relevant/contaminated_pmids are BUILT IN CODE (services/retrieval.py)
    # from the per-PMID `verdicts` map the synthesize call emits (each PMID labeled contaminated /
    # supporting / contradicting / mixed). They are NOT emitted directly by the LLM — this removed
    # the loose second-pass bucketing that mis-placed a positive trial as contradicting (BRAVE-I).
    # supporting = supporting+mixed; contradicting = contradicting+mixed; relevant = non-contaminated.
    supporting_pmids: list[str] = []
    contradicting_pmids: list[str] = []
    # relevant_pmids = the abstracts graded as this-drug-this-disease evidence (strength/direction
    # grade over these); contaminated_pmids = the excluded ones. Renderers show an "N excluded" note.
    relevant_pmids: list[str] = []
    contaminated_pmids: list[str] = []
    # neutral_pmids = relevant abstracts with NO efficacy result (PK / safety-only / mechanism),
    # labeled "neutral" by the direction sub-call. They count toward study_count and may be cited in
    # the narrative as context, but are in NEITHER supporting nor contradicting. Surfaced so a reader
    # can see why a cited PMID is in neither list (rather than appearing dropped).
    neutral_pmids: list[str] = []
    # Populated by a SEPARATE safety-focused search+summarize pass (services/retrieval.py
    # RetrievalService.safety_search / summarize_safety) — reranks the SAME PMID pool by
    # relevance to adverse events/safety rather than efficacy. "" / [] means no abstract in the
    # safety-reranked pool contained safety-relevant content (not "drug is safe").
    # DRUG-LEVEL safety blurb — the drug's drug-wide safety signal (~identical across candidates;
    # the supervisor collapses these into one shown once at the top of the report). "" = no signal.
    safety_summary: str = ""
    safety_pmids: list[str] = []
    # Severity of the DRUG-LEVEL safety signal. Production: deterministic from OT warning_type
    # (withdrawn / black_box, else serious when an OT AE signal exists). Holdout (OT suppressed):
    # LLM-picked from pre-cutoff literature (serious / moderate). "none" = no signal.
    safety_severity: Literal[
        "withdrawn", "black_box", "serious", "moderate", "none"
    ] = "none"
    # DISEASE-SPECIFIC safety — whether the disease-scoped literature reports a harm for THIS drug
    # IN THIS INDICATION's context (validated concrete question, not the fuzzy "unique to disease").
    # Drives the per-candidate report/table flag. False when the indication's safety literature is
    # efficacy-only or absent (NOT "confirmed safe").
    indication_harm: bool = False
    indication_harm_summary: str = ""
    indication_harm_pmids: list[str] = []

    @field_validator(
        "supporting_pmids",
        "contradicting_pmids",
        "relevant_pmids",
        "contaminated_pmids",
        "neutral_pmids",
        "safety_pmids",
        "indication_harm_pmids",
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
