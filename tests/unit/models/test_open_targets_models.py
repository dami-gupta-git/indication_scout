"""Unit tests for Open Targets models."""

import pytest

from indication_scout.models.model_open_targets import (
    DrugData,
    DrugTarget,
    GeneticConstraint,
    Indication,
    MechanismOfAction,
    TargetData,
)

# --- DrugData ---


@pytest.fixture
def sample_drug_data():
    """Create a sample DrugData for testing."""
    return DrugData(
        chembl_id="CHEMBL2108724",
        name="SEMAGLUTIDE",
        drug_type="Protein",
        maximum_clinical_stage="APPROVAL",
        targets=[
            DrugTarget(
                target_id="ENSG00000112164",
                target_symbol="GLP1R",
                mechanism_of_action="GLP-1 receptor agonist",
                action_type="AGONIST",
            ),
            DrugTarget(
                target_id="ENSG00000171016",
                target_symbol="PYGO1",
                mechanism_of_action="Pygo homolog 1 inhibitor",
                action_type="INHIBITOR",
            ),
        ],
        indications=[
            Indication(
                disease_id="MONDO_0005148",
                disease_name="type 2 diabetes mellitus",
                max_clinical_stage="APPROVAL",
            ),
            Indication(
                disease_id="EFO_0001073",
                disease_name="obesity",
                max_clinical_stage="APPROVAL",
            ),
            Indication(
                disease_id="MONDO_0005155",
                disease_name="non-alcoholic steatohepatitis",
                max_clinical_stage="PHASE_3",
            ),
        ],
    )


def test_drug_data_contains_drug_targets(sample_drug_data):
    """DrugData.targets should contain DrugTarget objects with all fields."""
    assert len(sample_drug_data.targets) == 2

    # First target
    assert sample_drug_data.targets[0].target_id == "ENSG00000112164"
    assert sample_drug_data.targets[0].target_symbol == "GLP1R"
    assert sample_drug_data.targets[0].mechanism_of_action == "GLP-1 receptor agonist"
    assert sample_drug_data.targets[0].action_type == "AGONIST"

    # Second target
    assert sample_drug_data.targets[1].target_id == "ENSG00000171016"
    assert sample_drug_data.targets[1].target_symbol == "PYGO1"
    assert sample_drug_data.targets[1].mechanism_of_action == "Pygo homolog 1 inhibitor"
    assert sample_drug_data.targets[1].action_type == "INHIBITOR"


def test_drug_data_mechanisms_of_action_field():
    """DrugData.mechanisms_of_action should store MechanismOfAction objects."""
    drug = DrugData(
        chembl_id="CHEMBL2108724",
        name="SEMAGLUTIDE",
        mechanisms_of_action=[
            MechanismOfAction(
                mechanism_of_action="GLP-1 receptor agonist",
                action_type="AGONIST",
                target_ids=["ENSG00000112164"],
                target_symbols=["GLP1R"],
            ),
            MechanismOfAction(
                mechanism_of_action="Pygo homolog 1 inhibitor",
                action_type="INHIBITOR",
                target_ids=["ENSG00000171016"],
                target_symbols=["PYGO1"],
            ),
        ],
    )

    assert len(drug.mechanisms_of_action) == 2

    glp1r_moa = drug.mechanisms_of_action[0]
    assert glp1r_moa.mechanism_of_action == "GLP-1 receptor agonist"
    assert glp1r_moa.action_type == "AGONIST"
    assert glp1r_moa.target_ids == ["ENSG00000112164"]
    assert glp1r_moa.target_symbols == ["GLP1R"]

    pygo_moa = drug.mechanisms_of_action[1]
    assert pygo_moa.mechanism_of_action == "Pygo homolog 1 inhibitor"
    assert pygo_moa.action_type == "INHIBITOR"
    assert pygo_moa.target_ids == ["ENSG00000171016"]
    assert pygo_moa.target_symbols == ["PYGO1"]


def test_drug_data_mechanisms_of_action_defaults_to_empty():
    """DrugData.mechanisms_of_action should default to [] when not provided."""
    drug = DrugData(chembl_id="CHEMBL9999", name="NO_MOA_DRUG")

    assert drug.mechanisms_of_action == []


def test_drug_data_coerce_nones_converts_null_mechanisms_of_action_to_empty():
    """DrugData with mechanisms_of_action=None must coerce to []."""
    drug = DrugData(
        chembl_id="CHEMBL9999",
        name="TEST_DRUG",
        mechanisms_of_action=None,
    )

    assert drug.mechanisms_of_action == []


def test_mechanism_of_action_all_fields():
    """MechanismOfAction should store all fields correctly."""
    moa = MechanismOfAction(
        mechanism_of_action="Glucagon-like peptide 1 receptor agonist",
        action_type="AGONIST",
        target_ids=["ENSG00000112164"],
        target_symbols=["GLP1R"],
    )

    assert moa.mechanism_of_action == "Glucagon-like peptide 1 receptor agonist"
    assert moa.action_type == "AGONIST"
    assert moa.target_ids == ["ENSG00000112164"]
    assert moa.target_symbols == ["GLP1R"]


def test_mechanism_of_action_multiple_targets():
    """MechanismOfAction can hold multiple targets sharing the same mechanism."""
    moa = MechanismOfAction(
        mechanism_of_action="Complex I inhibitor",
        action_type="INHIBITOR",
        target_ids=["ENSG00000001", "ENSG00000002", "ENSG00000003"],
        target_symbols=["NDUFV1", "NDUFS1", "NDUFS2"],
    )

    assert moa.mechanism_of_action == "Complex I inhibitor"
    assert moa.action_type == "INHIBITOR"
    assert moa.target_ids == ["ENSG00000001", "ENSG00000002", "ENSG00000003"]
    assert moa.target_symbols == ["NDUFV1", "NDUFS1", "NDUFS2"]


def test_mechanism_of_action_coerce_nones():
    """MechanismOfAction with None list fields must coerce to []."""
    moa = MechanismOfAction(
        mechanism_of_action="Some inhibitor",
        action_type=None,
        target_ids=None,
        target_symbols=None,
    )

    assert moa.action_type is None
    assert moa.target_ids == []
    assert moa.target_symbols == []


def test_drug_target_action_type_optional():
    """DrugTarget.action_type should be optional (None allowed)."""
    drug = DrugData(
        chembl_id="CHEMBL1234",
        name="TEST_DRUG",
        drug_type="Small molecule",
        maximum_clinical_stage="PHASE_2",
        targets=[
            DrugTarget(
                target_id="ENSG00000112164",
                target_symbol="GLP1R",
                mechanism_of_action="GLP-1 receptor modulator",
                action_type=None,
            ),
        ],
    )

    assert len(drug.targets) == 1
    assert drug.targets[0].target_id == "ENSG00000112164"
    assert drug.targets[0].target_symbol == "GLP1R"
    assert drug.targets[0].mechanism_of_action == "GLP-1 receptor modulator"
    assert drug.targets[0].action_type is None


def test_drug_data_empty_targets():
    """DrugData.targets should default to empty list."""
    drug = DrugData(
        chembl_id="CHEMBL9999",
        name="NO_TARGET_DRUG",
        drug_type="Small molecule",
        maximum_clinical_stage="PHASE_1",
    )

    assert drug.targets == []


def test_drug_data_coerce_nones_converts_null_lists_to_empty():
    """DrugData with all list fields set to None must coerce them to []."""
    drug = DrugData(
        chembl_id="CHEMBL9999",
        name="TEST_DRUG",
        synonyms=None,
        trade_names=None,
        mechanisms_of_action=None,
        warnings=None,
        indications=None,
        targets=None,
        adverse_events=None,
        atc_classifications=None,
    )

    assert drug.synonyms == []
    assert drug.trade_names == []
    assert drug.mechanisms_of_action == []
    assert drug.warnings == []
    assert drug.indications == []
    assert drug.targets == []
    assert drug.adverse_events == []
    assert drug.atc_classifications == []


def test_approved_disease_ids_returns_approval_only(sample_drug_data):
    """approved_disease_ids should return only APPROVAL stage disease IDs."""
    approved = sample_drug_data.approved_disease_ids
    assert approved == {"MONDO_0005148", "EFO_0001073"}
    assert "MONDO_0005155" not in approved


def test_investigated_disease_ids_returns_all(sample_drug_data):
    """investigated_disease_ids should return all disease IDs."""
    investigated = sample_drug_data.investigated_disease_ids
    assert investigated == {"MONDO_0005148", "EFO_0001073", "MONDO_0005155"}


def test_approved_disease_ids_empty_when_no_indications():
    """approved_disease_ids should be empty when no indications."""
    drug = DrugData(
        chembl_id="CHEMBL9999",
        name="NO_INDICATION_DRUG",
        drug_type="Small molecule",
        maximum_clinical_stage="PHASE_1",
        indications=[],
    )
    assert drug.approved_disease_ids == set()


# --- TargetData ---


def test_target_data_coerce_nones_converts_null_lists_to_empty():
    """TargetData with all list fields set to None must coerce them to []."""
    target = TargetData(
        target_id="ENSG00000099977",
        symbol="PRKAA1",
        name="5'-AMP-activated protein kinase catalytic subunit alpha-1",
        associations=None,
        pathways=None,
        interactions=None,
        drug_summaries=None,
        expressions=None,
        mouse_phenotypes=None,
        safety_liabilities=None,
        genetic_constraint=None,
    )

    assert target.associations == []
    assert target.pathways == []
    assert target.interactions == []
    assert target.drug_summaries == []
    assert target.expressions == []
    assert target.mouse_phenotypes == []
    assert target.safety_liabilities == []
    assert target.genetic_constraint == []


# --- GeneticConstraint ---


def test_genetic_constraint_all_fields():
    """GeneticConstraint should store all fields returned by the GraphQL query."""
    c = GeneticConstraint(
        constraint_type="lof",
        exp=12.3,
        obs=4.0,
        oe=0.325,
        oe_lower=0.18,
        oe_upper=0.59,
        score=0.91,
        upper_bin=0,
        upper_bin6=1,
    )

    assert c.constraint_type == "lof"
    assert c.exp == 12.3
    assert c.obs == 4.0
    assert c.oe == 0.325
    assert c.oe_lower == 0.18
    assert c.oe_upper == 0.59
    assert c.score == 0.91
    assert c.upper_bin == 0
    assert c.upper_bin6 == 1


def test_genetic_constraint_optional_fields_default_to_none():
    """GeneticConstraint fields are all optional except constraint_type."""
    c = GeneticConstraint(constraint_type="syn")

    assert c.constraint_type == "syn"
    assert c.exp is None
    assert c.obs is None
    assert c.oe is None
    assert c.oe_lower is None
    assert c.oe_upper is None
    assert c.score is None
    assert c.upper_bin is None
    assert c.upper_bin6 is None


def test_investigated_disease_ids_empty_when_no_indications():
    """investigated_disease_ids should be empty when no indications."""
    drug = DrugData(
        chembl_id="CHEMBL9999",
        name="NO_INDICATION_DRUG",
        drug_type="Small molecule",
        maximum_clinical_stage="PHASE_1",
        indications=[],
    )
    assert drug.investigated_disease_ids == set()
