"""Saved mechanism graph: how a run's drug connects to each final candidate through its targets.

Built once per run from Open Targets data and stored on SupervisorOutput, so `scout graph` renders it without network
calls. Every value is copied from Open Targets; nothing is inferred through pathways.
"""

from pydantic import BaseModel, model_validator

from indication_scout.models.model_open_targets import Pathway


class GraphTarget(BaseModel):
    """A drug target that links to at least one final candidate, with its Reactome pathways."""

    symbol: str = ""
    target_id: str = ""
    action_types: list[str] = []
    pathways: list[Pathway] = []

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


class TargetLink(BaseModel):
    """One Open Targets association between a drug target and a candidate disease.

    ``datatype_scores`` is the association's per-evidence-type breakdown as returned by Open Targets (e.g. clinical,
    literature, genetic_association).
    """

    target_symbol: str = ""
    overall_score: float | None = None
    datatype_scores: dict[str, float] = {}

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


class CandidateExplanation(BaseModel):
    """A final candidate, its resolved Open Targets disease, and the drug targets associated with it.

    ``disease_id`` is None when the name did not resolve; ``disease_name`` and ``therapeutic_areas`` come from the
    association and stay empty when no drug target links to the disease. ``links`` is ordered by overall score,
    descending.
    """

    candidate: str = ""
    disease_id: str | None = None
    disease_name: str = ""
    therapeutic_areas: list[str] = []
    links: list[TargetLink] = []

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


class MechanismGraph(BaseModel):
    """Drug → target → candidate disease graph for one run. ``candidates`` follows the run's ranking order."""

    drug_name: str = ""
    targets: list[GraphTarget] = []
    candidates: list[CandidateExplanation] = []

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values
