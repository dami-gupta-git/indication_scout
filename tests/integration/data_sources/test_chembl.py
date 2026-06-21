"""Integration tests for ChEMBLClient — hits the live ChEMBL API."""

import pytest

from indication_scout.models.model_chembl import (
    ATCDescription,
    MoleculeData,
    MoleculeSynonym,
)


@pytest.mark.parametrize(
    "chembl_id, expected_atc, expected_type, expected_max_phase, expected_black_box, expected_first_approval, expected_oral, expected_pref_name, expected_parent_chembl_id",
    [
        (
            "CHEMBL894",
            ["N06AX12"],
            "Small molecule",
            "4.0",
            1,
            1985,
            True,
            "bupropion",
            "CHEMBL894",
        ),
        (
            "CHEMBL2108724",
            ["A10BJ06"],
            "Protein",
            "4.0",
            1,
            2017,
            True,
            "semaglutide",
            "CHEMBL2108724",
        ),
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
    expected_pref_name,
    expected_parent_chembl_id,
):
    result = await chembl_client.get_molecule(chembl_id)

    assert isinstance(result, MoleculeData)
    assert result.molecule_chembl_id == chembl_id
    assert result.pref_name == expected_pref_name
    assert result.parent_chembl_id == expected_parent_chembl_id
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


# TODO Delete
async def test_single_drug(test_cache_dir):
    from indication_scout.data_sources.chembl import get_all_drug_names

    # chembl_id="CHEMBL894"  # bupropion
    # chembl_id = "CHEMBL1431"  # metformin
    chembl_id = "CHEMBL1201583"  # trastuzumab

    result = await get_all_drug_names(chembl_id, cache_dir=test_cache_dir)
    assert result


# --- get_all_drug_names ---


@pytest.mark.parametrize(
    "chembl_id, expected_pref_name, expected_names",
    [
        (
            "CHEMBL894",
            "bupropion",
            [
                "wellbutrin",
                "zyban",
                "aplenzin",
                "forfivo xl",
                "bupropion hcl",
                "bw-323",
            ],
        ),
        (
            "CHEMBL2108724",
            "semaglutide",
            ["ozempic", "rybelsus", "wegovy", "nn9535"],
        ),
        # sildenafil — trade names (Revatio/Viagra) live on SALT FORMS, fetched via
        # molecule_hierarchy. If that ChEMBL call fails, aliases lack the trade names → openFDA
        # label fetch returns nothing → approved indications empty → the approval labeler can't
        # mark systemic hypertension contaminated (it falls to none). This case pins that path;
        # a red here means ChEMBL salt-form discovery is down, not a logic regression.
        (
            "CHEMBL192",
            "sildenafil",
            ["revatio", "viagra"],
        ),
    ],
)
async def test_get_all_drug_names(
    chembl_id, expected_pref_name, expected_names, test_cache_dir
):
    from indication_scout.data_sources.chembl import get_all_drug_names

    result = await get_all_drug_names(chembl_id, cache_dir=test_cache_dir)

    # pref_name is always first
    assert result[0] == expected_pref_name

    # all expected trade names present
    for name in expected_names:
        assert (
            name in result
        ), f"Expected '{name}' in drug names for {chembl_id}, got {result}"

    # all names are lowercase
    for name in result:
        assert name == name.lower(), f"Expected lowercase, got '{name}'"


async def test_get_all_drug_names_sildenafil(test_cache_dir):
    """Standalone sildenafil alias check: Revatio/Viagra live on salt forms (molecule_hierarchy).
    A red here means ChEMBL salt-form discovery is down — which makes the approval labeler return
    'none' for systemic hypertension (no PAH/ED approvals to contaminate against)."""
    from indication_scout.data_sources.chembl import get_all_drug_names

    result = await get_all_drug_names("CHEMBL192", cache_dir=test_cache_dir)
    assert result[0] == "sildenafil"
    for name in ("revatio", "viagra"):
        assert name in result, f"Expected '{name}' in sildenafil names, got {result}"

    # "component of" entries should be filtered out
    for name in result:
        assert "component of" not in name


# --- resolve_drug_name ---


@pytest.mark.parametrize(
    "drug_name, expected_chembl_id",
    [
        ("metformin", "CHEMBL1431"),
        ("metformin hydrochloride", "CHEMBL1431"),
        ("semaglutide", "CHEMBL2108724"),
    ],
)
async def test_resolve_drug_name(drug_name, expected_chembl_id, test_cache_dir):
    """Parent inputs resolve to themselves; salt inputs follow to parent."""
    from indication_scout.data_sources.chembl import resolve_drug_name

    result = await resolve_drug_name(drug_name, cache_dir=test_cache_dir)
    assert result == expected_chembl_id
