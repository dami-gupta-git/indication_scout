"""Structured output from the mechanism agent."""

from typing import Literal

from pydantic import BaseModel, Field

from indication_scout.models.model_open_targets import Association, MechanismOfAction, Pathway


class ShapedAssociation(BaseModel):
    """A disease association annotated with mechanistic directionality."""

    target_symbol: str = Field(description="Gene symbol of the target")
    disease_name: str = Field(description="Disease name from Open Targets")
    disease_id: str = Field(description="Disease ID from Open Targets")
    shape: Literal["hypothesis", "contraindication", "neutral", "confirms_known"] = Field(
        description=(
            "hypothesis: drug action direction matches disease mechanism; "
            "contraindication: drug action opposes disease mechanism; "
            "neutral: insufficient evidence to determine direction; "
            "confirms_known: association is dominated by the drug's own clinical use"
        )
    )
    rationale: str = Field(
        description="One sentence explaining why this shape was assigned, referencing the action type and disease mechanism"
    )


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
    associations: dict[str, list[Association]] = Field(
        default_factory=dict,
        description="Top disease associations per target (symbol → associations)",
    )
    shaped_associations: list[ShapedAssociation] = Field(
        default_factory=list,
        description="Per-association directionality judgements from the agent",
    )
    pathways: dict[str, list[Pathway]] = Field(
        default_factory=dict,
        description="Reactome pathways per target (symbol → pathways)",
    )
    summary: str = Field(default="", description="Narrative summary from the agent")
