"""Unit tests for ChEMBL Pydantic models."""

from indication_scout.models.model_chembl import ATCDescription, MoleculeData

# --- ATCDescription ---


def test_atc_description_coerce_nones_converts_nulls_to_empty_string():
    """ATCDescription with all fields set to None must coerce them to ''."""
    atc = ATCDescription(
        level1=None,
        level1_description=None,
        level2=None,
        level2_description=None,
        level3=None,
        level3_description=None,
        level4=None,
        level4_description=None,
        level5=None,
        who_name=None,
    )

    assert atc.level1 == ""
    assert atc.level1_description == ""
    assert atc.level2 == ""
    assert atc.level2_description == ""
    assert atc.level3 == ""
    assert atc.level3_description == ""
    assert atc.level4 == ""
    assert atc.level4_description == ""
    assert atc.level5 == ""
    assert atc.who_name == ""


# --- MoleculeData ---


def test_molecule_data_coerce_nones_converts_null_list_to_empty():
    """MoleculeData.atc_classifications=None must coerce to []."""
    mol = MoleculeData(
        molecule_chembl_id="CHEMBL894",
        molecule_type="Small molecule",
        atc_classifications=None,
    )

    assert mol.atc_classifications == []


def test_molecule_data_coerce_nones_preserves_genuine_nones():
    """MoleculeData fields with default=None must stay None when passed None."""
    mol = MoleculeData(
        molecule_chembl_id="CHEMBL894",
        molecule_type="Small molecule",
        max_phase=None,
        black_box_warning=None,
        first_approval=None,
        oral=None,
    )

    assert mol.max_phase is None
    assert mol.black_box_warning is None
    assert mol.first_approval is None
    assert mol.oral is None
