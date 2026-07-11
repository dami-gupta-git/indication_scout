"""Per-drug regression-spec schema.

Each entry in a spec is a single assertion. Every assertion is tagged with a
bucket from `failure_buckets.Bucket` so failures roll up into the taxonomy.

Spec assertions are intentionally narrow and named after their *contract* —
"this NCT appears in the completed-trials list for this indication" — rather
than the implementation. When the pipeline changes shape, the loader and
assertion functions move; the spec content stays.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from tests.regression.common.failure_buckets import Bucket


def _coerce_nones(cls, values):
    for field_name, field_info in cls.model_fields.items():
        if values.get(field_name) is None:
            if field_info.default_factory is not None:
                values[field_name] = field_info.default_factory()
            elif field_info.default is not None:
                values[field_name] = field_info.default
    return values


class RequiredNCTs(BaseModel):
    """`ncts` must appear in disease_findings[indication].clinical_trials.

    `section` selects which pool the NCTs must appear in:
    - "relevant"   — clinical_trials.relevant_nct_ids (the curated set the
                     report actually surfaces to the user). This is the
                     meaningful invariant.
    - "completed"  — completed.trials
    - "terminated" — terminated.trials
    - "search"     — search.trials
    - "any"        — union of completed + terminated + search
    """

    bucket: Bucket = Bucket.LITERATURE_COVERAGE
    indication: str
    ncts: list[str] = Field(default_factory=list)
    section: str = "relevant"

    @model_validator(mode="before")
    @classmethod
    def _v(cls, values):
        return _coerce_nones(cls, values)


class RequiredPMIDs(BaseModel):
    """`pmids` must be cited in disease_findings[indication].literature.

    `mode` selects which pool the PMIDs must appear in:
    - "cited" — evidence_summary.supporting_pmids + contradicting_pmids (the
                curated set the report actually cites). This is the meaningful
                invariant.
    - "pool"  — literature.pmids (the full retrieval pool, ~100+ PMIDs). A
                weak check — a PMID being present here does not mean the report
                surfaced it.
    """

    bucket: Bucket = Bucket.LITERATURE_COVERAGE
    indication: str
    pmids: list[str] = Field(default_factory=list)
    mode: str = "cited"

    @model_validator(mode="before")
    @classmethod
    def _v(cls, values):
        return _coerce_nones(cls, values)


class RequiredInRanked(BaseModel):
    """`indication` must appear in top_diseases."""

    bucket: Bucket = Bucket.RANKING
    indication: str

    @model_validator(mode="before")
    @classmethod
    def _v(cls, values):
        return _coerce_nones(cls, values)


class RankedOrder(BaseModel):
    """`indications` must appear in top_diseases in exactly this relative order.

    Compares only the listed indications, in order, ignoring any others in
    top_diseases. Every listed indication must be present; a missing one is a
    failure. Use to pin the ranked ordering the report presents to the user.
    """

    bucket: Bucket = Bucket.RANKING
    indications: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _v(cls, values):
        return _coerce_nones(cls, values)


class ForbiddenInRanked(BaseModel):
    """`indication` must NOT appear in top_diseases (demotion / gate worked)."""

    bucket: Bucket = Bucket.DEMOTION_LOGIC
    indication: str

    @model_validator(mode="before")
    @classmethod
    def _v(cls, values):
        return _coerce_nones(cls, values)


class ForbiddenPhrase(BaseModel):
    """`phrase` must NOT appear in the rendered report (case-insensitive)."""

    bucket: Bucket = Bucket.FACTUAL_ACCURACY
    phrase: str
    scope: str = "anywhere"  # "anywhere" | "summary" | "blurb"

    @model_validator(mode="before")
    @classmethod
    def _v(cls, values):
        return _coerce_nones(cls, values)


class CandidateSetContains(BaseModel):
    """`indications` must all appear in candidate_diseases."""

    bucket: Bucket = Bucket.STRUCTURAL_INTEGRITY
    indications: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _v(cls, values):
        return _coerce_nones(cls, values)


class DrugSpec(BaseModel):
    """A single drug's regression spec, loaded from `specs/<drug>.yaml`."""

    drug: str
    fixtures_date: str | None = None
    # Optional pointer to the dated `fixtures/<drug>/<date>/` directory the
    # spec was authored against. Used in the failure message when an assertion
    # fails so you can find the corresponding upstream snapshot.

    required_ncts_surfaced: list[RequiredNCTs] = Field(default_factory=list)
    required_pmids_cited: list[RequiredPMIDs] = Field(default_factory=list)
    required_in_ranked: list[RequiredInRanked] = Field(default_factory=list)
    ranked_order: RankedOrder | None = None
    forbidden_in_ranked: list[ForbiddenInRanked] = Field(default_factory=list)
    forbidden_phrases: list[ForbiddenPhrase] = Field(default_factory=list)
    candidate_set_contains: CandidateSetContains | None = None

    @model_validator(mode="before")
    @classmethod
    def _v(cls, values):
        return _coerce_nones(cls, values)
