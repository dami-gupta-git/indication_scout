"""Structured output from the mechanism agent."""

from pydantic import BaseModel, Field, model_validator

from indication_scout.models.model_open_targets import MechanismOfAction


class MechanismCandidate(BaseModel):
    """A repurposing candidate surfaced by the mechanism agent.

    A (target, disease) pair where the drug's action direction aligns with the disease mechanism per Open
    Targets evidence, and the disease is not an approved indication. Carries text context the LLM
    reasons over. No scores, no direction labels — those are used upstream for selection and discarded
    here.
    """

    target_symbol: str = ""
    action_type: str = ""
    disease_name: str = ""
    disease_id: str | None = None
    disease_description: str = ""
    target_function: str = ""

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


class MechanismOutput(BaseModel):
    """Final assembled output from a single mechanism agent run."""

    drug_targets: dict[str, str] = Field(
        default_factory=dict,
        description="Gene symbol → Ensembl target ID for each of the drug's targets",
    )
    mechanisms_of_action: list[MechanismOfAction] = Field(
        default_factory=list,
        description="Structured MoA entries from the drug entity: action type, mechanism string, and targets",
    )
    candidates: list[MechanismCandidate] = Field(
        default_factory=list,
        description="Top POSITIVE-direction, non-approved repurposing candidates",
    )
    summary: str = Field(default="", description="Narrative summary from the agent")

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None:
                if field_info.default_factory is not None:
                    values[field_name] = field_info.default_factory()
                elif field_info.default is not None:
                    values[field_name] = field_info.default
        return values
