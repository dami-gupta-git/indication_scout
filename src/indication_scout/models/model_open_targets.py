"""
Pydantic models for Open Targets data.

These are the data contracts between the Open Targets client and the agents.
Agents receive these models — they never see raw GraphQL responses.
"""

from pydantic import BaseModel, model_validator

# ------------------------------------------------------------------
# Target-level models
# ------------------------------------------------------------------


class Association(BaseModel):
    """Target-disease association with per-datatype evidence breakdown."""

    disease_id: str = ""
    disease_name: str = ""
    overall_score: float | None = None
    datatype_scores: dict[str, float] = (
        {}
    )  # e.g. {"genetic_association": 0.7, "literature": 0.9}
    therapeutic_areas: list[str] = []

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


class Pathway(BaseModel):
    """Reactome pathway the target participates in."""

    pathway_id: str = ""
    pathway_name: str = ""
    top_level_pathway: str = ""

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


class Interaction(BaseModel):
    """Protein-protein interaction partner."""

    interacting_target_id: str = ""
    interacting_target_symbol: str = ""
    interaction_score: float | None = None  # null for reactome and signor
    source_database: str = ""  # IntAct, STRING, Signor, Reactome
    biological_role: str = ""
    evidence_count: int | None = None
    interaction_type: str | None = (
        None  # "physical", "functional", "signalling", "enzymatic"
    )

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


class CellTypeExpression(BaseModel):
    """Protein expression in a specific cell type within a tissue."""

    name: str = ""
    level: int | None = None
    reliability: bool | None = None

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


class RNAExpression(BaseModel):
    """RNA expression measurement."""

    value: float | None = None  # TPM
    quantile: int | None = None  # relative level across tissues
    unit: str | None = None

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


class ProteinExpression(BaseModel):
    """Protein expression measurement with cell type detail."""

    level: int | None = None  # 0-3
    reliability: bool | None = None
    cell_types: list[CellTypeExpression] = []

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


class TissueExpression(BaseModel):
    """Expression data for a single tissue."""

    tissue_id: str = ""  # UBERON ID
    tissue_name: str = ""
    tissue_anatomical_system: str = ""
    rna: RNAExpression | None = None
    protein: ProteinExpression | None = None

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


class BiologicalModel(BaseModel):
    """A specific mouse model (knockout, knock-in, etc.)."""

    allelic_composition: str = ""
    genetic_background: str = ""
    literature: list[str] = []  # PMIDs
    model_id: str = ""  # MGI ID

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


class MousePhenotype(BaseModel):
    """Phenotype observed in mouse models for this target."""

    phenotype_id: str = ""  # MP ontology ID
    phenotype_label: str = ""
    phenotype_categories: list[str] = []  # top-level MP categories
    biological_models: list[BiologicalModel] = []

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


class SafetyEffect(BaseModel):
    """A safety effect with direction and dosing conditions."""

    direction: str = ""
    dosing: str | None = None

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


class SafetyLiability(BaseModel):
    """Known target safety effect from Open Targets."""

    event: str | None = None
    event_id: str | None = None
    effects: list[SafetyEffect] = []
    datasource: str | None = None
    literature: str | None = None
    url: str | None = None

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


class AdverseEvent(BaseModel):
    """Significant adverse event from FAERS for a drug."""

    name: str = ""
    meddra_code: str | None = None
    count: int | None = None
    log_likelihood_ratio: float | None = None

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


class GeneticConstraint(BaseModel):
    """GnomAD loss-of-function intolerance score."""

    constraint_type: str = ""  # "syn", "mis", "lof"
    oe: float | None = None  # observed/expected
    oe_lower: float | None = None
    oe_upper: float | None = None
    score: float | None = None
    upper_bin: int | None = None  # 0 = most constrained, 5 = least

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


class DrugSummary(BaseModel):
    """A drug known to act on this target, with indication and phase info.

    Target-centric view of a drug — lives inside TargetData.drug_summaries.
    Inverse of DrugData: if get_target("GLP1R") returns a TargetData, its
    drug_summaries will include semaglutide, liraglutide, etc. The shared key
    with DrugData is drug_id (ChEMBL ID).
    """

    drug_id: str = ""  # ChEMBL ID
    drug_name: str = ""
    disease_id: str = ""
    disease_name: str = ""
    phase: float | None = None
    status: str | None = None
    mechanism_of_action: str = ""
    clinical_trial_ids: list[str] = []  # NCT IDs

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


class TargetData(BaseModel):
    """Everything Open Targets knows about a target. Populated once by get_target()."""

    target_id: str = ""
    symbol: str = ""
    name: str = ""
    associations: list[Association] = []
    pathways: list[Pathway] = []
    interactions: list[Interaction] = []
    drug_summaries: list[DrugSummary] = []
    expressions: list[TissueExpression] = []
    mouse_phenotypes: list[MousePhenotype] = []
    safety_liabilities: list[SafetyLiability] = []
    genetic_constraint: list[GeneticConstraint] = []

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


# ------------------------------------------------------------------
# Drug-level models
# ------------------------------------------------------------------


class DrugTarget(BaseModel):
    """A target linked to a drug via mechanism of action."""

    target_id: str = ""  # Ensembl gene ID
    target_symbol: str = ""  # e.g. "GLP1R"
    mechanism_of_action: str = ""  # e.g. "Glucagon-like peptide 1 receptor agonist"
    action_type: str | None = None  # e.g. "AGONIST", "INHIBITOR"

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


class DiseaseSynonyms(BaseModel):
    """Synonyms for a disease from Open Targets, grouped by relation type."""

    disease_id: str = ""
    disease_name: str = ""
    parent_names: list[str] = []
    exact: list[str] = []
    related: list[str] = []
    narrow: list[str] = []
    broad: list[str] = []

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values

    @property
    def all_synonyms(self) -> list[str]:
        """All synonym terms combined with parent names."""
        return self.exact + self.related + self.parent_names


class DrugWarning(BaseModel):
    """Black box warning or withdrawal."""

    warning_type: str = ""
    description: str | None = None
    toxicity_class: str | None = None
    country: str | None = None
    year: int | None = None
    efo_id: str | None = None

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


class Indication(BaseModel):
    """An approved or investigational indication for a drug."""

    disease_id: str = ""
    disease_name: str = ""
    max_phase: float | None = None  # 0-4
    references: list[dict] = []

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


class DrugData(BaseModel):
    """Everything Open Targets knows about a drug. Populated once by get_drug().

    Drug-centric view — queried by drug name. Contains the drug's targets,
    indications, warnings, and adverse events. Inverse of DrugSummary: a drug's
    targets list the same relationships that TargetData.drug_summaries captures
    from the target side. Shared key with DrugSummary is chembl_id / drug_id.
    Mechanism of action is not a top-level field; access it via targets[i].mechanism_of_action.
    """

    chembl_id: str = ""
    name: str = ""
    synonyms: list[str] = []
    trade_names: list[str] = []
    drug_type: str | None = None
    is_approved: bool | None = None
    max_clinical_phase: float | None = None
    year_first_approved: int | None = None
    warnings: list[DrugWarning] = []
    indications: list[Indication] = []
    targets: list[DrugTarget] = []
    adverse_events: list[AdverseEvent] = []
    adverse_events_critical_value: float | None = None
    atc_classifications: list[str] = []

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values

    @property
    def approved_disease_ids(self) -> set[str]:
        """Disease IDs with phase 4 / approved status — for filtering."""
        return {
            i.disease_id
            for i in self.indications
            if i.max_phase is not None and i.max_phase >= 4
        }

    @property
    def investigated_disease_ids(self) -> set[str]:
        """All disease IDs being actively investigated (any phase)."""
        return {i.disease_id for i in self.indications}


class RichDrugData(BaseModel):
    """DrugData combined with full TargetData for each of its targets.

    Returned by get_rich_drug_data(). Provides everything Open Targets knows
    about a drug and all its targets in a single object, for use by agents
    that need both drug-level and target-level data without making multiple calls.
    """

    drug: DrugData | None = None
    targets: list[TargetData] = []

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values
