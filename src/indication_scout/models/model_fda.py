"""Typed openFDA drug-label safety records."""

from pydantic import BaseModel, model_validator


class FDALabelSafetyRecord(BaseModel):
    """Safety fields retained from one openFDA drug-label record."""

    set_id: str | None = None
    effective_time: str | None = None
    brand_names: list[str] = []
    generic_names: list[str] = []
    boxed_warnings: list[str] = []
    warnings: list[str] = []

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values
