"""PubMed data models."""

from pydantic import BaseModel, model_validator


class PubmedAbstract(BaseModel):
    """Parsed PubMed abstract data."""

    pmid: str = ""
    title: str | None = None
    abstract: str | None = None
    authors: list[str] = []
    journal: str | None = None
    pub_date: str | None = None
    mesh_terms: list[str] = []
    keywords: list[str] = []

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values
