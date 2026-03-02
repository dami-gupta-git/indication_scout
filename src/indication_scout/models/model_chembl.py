"""Pydantic models for ChEMBL API data."""

from pydantic import BaseModel, model_validator


class ATCDescription(BaseModel):
    """ATC classification hierarchy for a single ATC code.

    Populated from GET /atc_class/{code}.json.
    """

    level1: str = ""
    level1_description: str = ""
    level2: str = ""
    level2_description: str = ""
    level3: str = ""
    level3_description: str = ""
    level4: str = ""
    level4_description: str = ""
    level5: str = ""  # full code, e.g. "A10BA02"
    who_name: str = ""

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


class MoleculeData(BaseModel):
    """Data returned by the ChEMBL molecule endpoint for a single compound."""

    molecule_chembl_id: str = ""
    molecule_type: str = ""
    max_phase: str | None = None
    atc_classifications: list[str] = []
    black_box_warning: int | None = None
    first_approval: int | None = None
    oral: bool | None = None

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values
