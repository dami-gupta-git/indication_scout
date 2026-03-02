"""Integration tests for ChEMBLClient — hits the live ChEMBL API."""

import pytest

from indication_scout.models.model_chembl import ATCDescription, MoleculeData


@pytest.mark.parametrize(
    "chembl_id, expected_atc, expected_type, expected_max_phase, expected_black_box, expected_first_approval, expected_oral",
    [
        ("CHEMBL894", ["N06AX12"], "Small molecule", "4.0", 1, 1985, True),
        ("CHEMBL2108724", ["A10BJ06"], "Protein", "4.0", 1, 2017, True),
    ],
)
async def test_get_molecule(
    chembl_client,
    chembl_id,
    expected_atc,
    expected_type,
    expected_max_phase,
    expected_black_box,
    expected_first_approval,
    expected_oral,
):
    result = await chembl_client.get_molecule(chembl_id)

    assert isinstance(result, MoleculeData)
    assert result.molecule_chembl_id == chembl_id
    assert result.molecule_type == expected_type
    assert result.max_phase == expected_max_phase
    assert result.atc_classifications == expected_atc
    assert result.black_box_warning == expected_black_box
    assert result.first_approval == expected_first_approval
    assert result.oral == expected_oral


@pytest.mark.parametrize(
    "atc_code, level1, level1_description, level2, level2_description, level3, level3_description, level4, level4_description, level5, who_name",
    [
        (
            "A10BA02",
            "A",
            "ALIMENTARY TRACT AND METABOLISM",
            "A10",
            "DRUGS USED IN DIABETES",
            "A10B",
            "BLOOD GLUCOSE LOWERING DRUGS, EXCL. INSULINS",
            "A10BA",
            "Biguanides",
            "A10BA02",
            "metformin",
        ),
    ],
)
async def test_get_atc_description(
    chembl_client,
    atc_code,
    level1,
    level1_description,
    level2,
    level2_description,
    level3,
    level3_description,
    level4,
    level4_description,
    level5,
    who_name,
):
    result = await chembl_client.get_atc_description(atc_code)

    assert isinstance(result, ATCDescription)
    assert result.level1 == level1
    assert result.level1_description == level1_description
    assert result.level2 == level2
    assert result.level2_description == level2_description
    assert result.level3 == level3
    assert result.level3_description == level3_description
    assert result.level4 == level4
    assert result.level4_description == level4_description
    assert result.level5 == level5
    assert result.who_name == who_name
