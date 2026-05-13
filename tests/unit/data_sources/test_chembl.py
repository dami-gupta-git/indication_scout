"""Unit tests for ChEMBLClient.get_molecule(), get_atc_description(), get_all_drug_names(), and resolve_drug_name()."""

from unittest.mock import AsyncMock, patch

import pytest

from indication_scout.data_sources.base_client import DataSourceError
from indication_scout.data_sources.chembl import (
    ChEMBLClient,
    get_all_drug_names,
    resolve_drug_name,
)
from indication_scout.models.model_chembl import (
    ATCDescription,
    MoleculeData,
    MoleculeSynonym,
)

CHEMBL894_FIXTURE = {
    "molecule_chembl_id": "CHEMBL894",
    "pref_name": "BUPROPION",
    "molecule_hierarchy": {"parent_chembl_id": "CHEMBL894"},
    "molecule_type": "Small molecule",
    "max_phase": "4.0",
    "atc_classifications": ["N06AX12"],
    "black_box_warning": 1,
    "first_approval": 1985,
    "oral": True,
    "molecule_synonyms": [
        {"molecule_synonym": "Bupropion", "syn_type": "INN", "synonyms": "BUPROPION"},
    ],
}

CHEMBL2108724_FIXTURE = {
    "molecule_chembl_id": "CHEMBL2108724",
    "pref_name": "SEMAGLUTIDE",
    "molecule_hierarchy": {"parent_chembl_id": "CHEMBL2108724"},
    "molecule_type": "Protein",
    "max_phase": "4.0",
    "atc_classifications": ["A10BJ06"],
    "black_box_warning": 1,
    "first_approval": 2017,
    "oral": True,
    "molecule_synonyms": [
        {
            "molecule_synonym": "Ozempic",
            "syn_type": "TRADE_NAME",
            "synonyms": "OZEMPIC",
        },
        {
            "molecule_synonym": "Semaglutide",
            "syn_type": "INN",
            "synonyms": "SEMAGLUTIDE",
        },
    ],
}

# Sildenafil
CHEMBL192_FIXTURE = {
    "molecule_chembl_id": "CHEMBL192",
    "pref_name": "SILDENAFIL",
    "molecule_hierarchy": {"parent_chembl_id": "CHEMBL192"},
    "molecule_type": "Small molecule",
    "max_phase": "4.0",
    "atc_classifications": ["G04BE03"],
    "black_box_warning": 0,
    "first_approval": 1998,
    "oral": True,
    "molecule_synonyms": [],
}

# Rituximab
CHEMBL1201585_FIXTURE = {
    "molecule_chembl_id": "CHEMBL1201585",
    "pref_name": "RITUXIMAB",
    "molecule_hierarchy": {"parent_chembl_id": "CHEMBL1201585"},
    "molecule_type": "Antibody",
    "max_phase": "4.0",
    "atc_classifications": ["L01FD01"],
    "black_box_warning": 1,
    "first_approval": 1998,
    "oral": False,
    "molecule_synonyms": [],
}

# Compound with null max_phase and no ATC codes
CHEMBL426_FIXTURE = {
    "molecule_chembl_id": "CHEMBL426",
    "pref_name": None,
    "molecule_hierarchy": None,
    "molecule_type": "Small molecule",
    "max_phase": None,
    "atc_classifications": [],
    "black_box_warning": 0,
    "first_approval": None,
    "oral": False,
    "molecule_synonyms": None,
}


@pytest.mark.parametrize(
    "chembl_id, fixture, expected_atc, expected_type, expected_max_phase, expected_black_box, expected_first_approval, expected_oral",
    [
        (
            "CHEMBL894",
            CHEMBL894_FIXTURE,
            ["N06AX12"],
            "Small molecule",
            "4.0",
            1,
            1985,
            True,
        ),
        (
            "CHEMBL2108724",
            CHEMBL2108724_FIXTURE,
            ["A10BJ06"],
            "Protein",
            "4.0",
            1,
            2017,
            True,
        ),
        (
            "CHEMBL192",
            CHEMBL192_FIXTURE,
            ["G04BE03"],
            "Small molecule",
            "4.0",
            0,
            1998,
            True,
        ),
        (
            "CHEMBL1201585",
            CHEMBL1201585_FIXTURE,
            ["L01FD01"],
            "Antibody",
            "4.0",
            1,
            1998,
            False,
        ),
        ("CHEMBL426", CHEMBL426_FIXTURE, [], "Small molecule", None, 0, None, False),
    ],
)
async def test_get_molecule(
    chembl_id,
    fixture,
    expected_atc,
    expected_type,
    expected_max_phase,
    expected_black_box,
    expected_first_approval,
    expected_oral,
    tmp_path,
):
    client = ChEMBLClient(cache_dir=tmp_path)
    with patch.object(client, "_rest_get", new=AsyncMock(return_value=fixture)):
        result = await client.get_molecule(chembl_id)

    assert isinstance(result, MoleculeData)
    assert result.molecule_chembl_id == chembl_id
    assert result.molecule_type == expected_type
    assert result.max_phase == expected_max_phase
    assert result.atc_classifications == expected_atc
    assert result.black_box_warning == expected_black_box
    assert result.first_approval == expected_first_approval
    assert result.oral == expected_oral


async def test_get_molecule_raises_on_data_source_error(tmp_path):
    client = ChEMBLClient(cache_dir=tmp_path)
    with patch.object(
        client,
        "_rest_get",
        new=AsyncMock(side_effect=DataSourceError("chembl", "HTTP 404", 404)),
    ):
        with pytest.raises(DataSourceError):
            await client.get_molecule("CHEMBL999999")


async def test_get_molecule_null_atc_returns_empty_list(tmp_path):
    fixture = {**CHEMBL894_FIXTURE, "atc_classifications": None}
    client = ChEMBLClient(cache_dir=tmp_path)
    with patch.object(client, "_rest_get", new=AsyncMock(return_value=fixture)):
        result = await client.get_molecule("CHEMBL894")

    assert result.atc_classifications == []


# --- get_atc_description ---

ATC_A10BA02_FIXTURE = {
    "level1": "A",
    "level1_description": "ALIMENTARY TRACT AND METABOLISM",
    "level2": "A10",
    "level2_description": "DRUGS USED IN DIABETES",
    "level3": "A10B",
    "level3_description": "BLOOD GLUCOSE LOWERING DRUGS, EXCL. INSULINS",
    "level4": "A10BA",
    "level4_description": "Biguanides",
    "level5": "A10BA02",
    "who_name": "metformin",
}

ATC_N06AX12_FIXTURE = {
    "level1": "N",
    "level1_description": "NERVOUS SYSTEM",
    "level2": "N06",
    "level2_description": "PSYCHOANALEPTICS",
    "level3": "N06A",
    "level3_description": "ANTIDEPRESSANTS",
    "level4": "N06AX",
    "level4_description": "Other antidepressants",
    "level5": "N06AX12",
    "who_name": "bupropion",
}


@pytest.mark.parametrize(
    "atc_code, fixture, expected_level1, expected_level1_description, expected_level2, expected_level2_description, expected_level3, expected_level3_description, expected_level4, expected_level4_description, expected_level5, expected_who_name",
    [
        (
            "A10BA02",
            ATC_A10BA02_FIXTURE,
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
        (
            "N06AX12",
            ATC_N06AX12_FIXTURE,
            "N",
            "NERVOUS SYSTEM",
            "N06",
            "PSYCHOANALEPTICS",
            "N06A",
            "ANTIDEPRESSANTS",
            "N06AX",
            "Other antidepressants",
            "N06AX12",
            "bupropion",
        ),
    ],
)
async def test_get_atc_description(
    atc_code,
    fixture,
    expected_level1,
    expected_level1_description,
    expected_level2,
    expected_level2_description,
    expected_level3,
    expected_level3_description,
    expected_level4,
    expected_level4_description,
    expected_level5,
    expected_who_name,
    tmp_path,
):
    client = ChEMBLClient(cache_dir=tmp_path)
    with patch.object(client, "_rest_get", new=AsyncMock(return_value=fixture)):
        result = await client.get_atc_description(atc_code)

    assert isinstance(result, ATCDescription)
    assert result.level1 == expected_level1
    assert result.level1_description == expected_level1_description
    assert result.level2 == expected_level2
    assert result.level2_description == expected_level2_description
    assert result.level3 == expected_level3
    assert result.level3_description == expected_level3_description
    assert result.level4 == expected_level4
    assert result.level4_description == expected_level4_description
    assert result.level5 == expected_level5
    assert result.who_name == expected_who_name


async def test_get_atc_description_raises_on_data_source_error(tmp_path):
    client = ChEMBLClient(cache_dir=tmp_path)
    with patch.object(
        client,
        "_rest_get",
        new=AsyncMock(side_effect=DataSourceError("chembl", "HTTP 404", 404)),
    ):
        with pytest.raises(DataSourceError):
            await client.get_atc_description("A10BA02")


# --- get_all_drug_names ---

# Parent molecule (bupropion freebase) — no TRADE_NAME synonyms on parent
PARENT_FIXTURE = {
    "molecule_chembl_id": "CHEMBL894",
    "pref_name": "BUPROPION",
    "molecule_hierarchy": {"parent_chembl_id": "CHEMBL894"},
    "molecule_type": "Small molecule",
    "max_phase": "4.0",
    "atc_classifications": ["N06AX12"],
    "black_box_warning": 1,
    "first_approval": 1985,
    "oral": True,
    "molecule_synonyms": [
        {"molecule_synonym": "Bupropion", "syn_type": "INN", "synonyms": "BUPROPION"},
        {
            "molecule_synonym": "BW-323",
            "syn_type": "RESEARCH_CODE",
            "synonyms": "BW-323",
        },
    ],
}

# Salt form (bupropion HCl) — has TRADE_NAME synonyms
SALT_HCL_MOLECULE = {
    "molecule_chembl_id": "CHEMBL1698",
    "molecule_synonyms": [
        {
            "molecule_synonym": "Wellbutrin",
            "syn_type": "TRADE_NAME",
            "synonyms": "WELLBUTRIN",
        },
        {"molecule_synonym": "Zyban", "syn_type": "TRADE_NAME", "synonyms": "ZYBAN"},
        {
            "molecule_synonym": "Bupropion hydrochloride",
            "syn_type": "INN",
            "synonyms": "BUPROPION HYDROCHLORIDE",
        },
        {
            "molecule_synonym": "Bupropion hydrochloride component of contrave",
            "syn_type": "TRADE_NAME",
            "synonyms": "BUPROPION HYDROCHLORIDE COMPONENT OF CONTRAVE",
        },
    ],
}

SALT_HBR_MOLECULE = {
    "molecule_chembl_id": "CHEMBL1201735",
    "molecule_synonyms": [
        {
            "molecule_synonym": "Aplenzin",
            "syn_type": "TRADE_NAME",
            "synonyms": "APLENZIN",
        },
    ],
}

HIERARCHY_RESPONSE = {
    "molecules": [
        {
            "molecule_chembl_id": "CHEMBL894",
            "molecule_synonyms": PARENT_FIXTURE["molecule_synonyms"],
        },
        SALT_HCL_MOLECULE,
        SALT_HBR_MOLECULE,
    ],
}

# Biologic (semaglutide) — trade names directly on parent, no salts
BIOLOGIC_PARENT_FIXTURE = {
    "molecule_chembl_id": "CHEMBL2108724",
    "pref_name": "SEMAGLUTIDE",
    "molecule_hierarchy": {"parent_chembl_id": "CHEMBL2108724"},
    "molecule_type": "Protein",
    "max_phase": "4.0",
    "atc_classifications": ["A10BJ06"],
    "black_box_warning": 1,
    "first_approval": 2017,
    "oral": True,
    "molecule_synonyms": [
        {
            "molecule_synonym": "Ozempic",
            "syn_type": "TRADE_NAME",
            "synonyms": "OZEMPIC",
        },
        {
            "molecule_synonym": "Rybelsus",
            "syn_type": "TRADE_NAME",
            "synonyms": "RYBELSUS",
        },
        {"molecule_synonym": "Wegovy", "syn_type": "TRADE_NAME", "synonyms": "WEGOVY"},
        {
            "molecule_synonym": "Semaglutide",
            "syn_type": "INN",
            "synonyms": "SEMAGLUTIDE",
        },
    ],
}

BIOLOGIC_HIERARCHY_RESPONSE = {
    "molecules": [
        {
            "molecule_chembl_id": "CHEMBL2108724",
            "molecule_synonyms": BIOLOGIC_PARENT_FIXTURE["molecule_synonyms"],
        },
    ],
}


def _patch_rest_get(mock_fn):
    """Patch ChEMBLClient._rest_get at the class level for the module function to pick up."""
    return patch(
        "indication_scout.data_sources.chembl.ChEMBLClient._rest_get",
        new=AsyncMock(side_effect=mock_fn),
    )


async def test_get_all_drug_names_small_molecule_with_salts(tmp_path):
    """pref_name first, then every synonym from parent + salt forms, all lowercase."""

    async def mock_rest_get(url: str, params: dict):
        if "/molecule/CHEMBL894.json" in url:
            return PARENT_FIXTURE
        if "/molecule.json" in url:
            return HIERARCHY_RESPONSE
        raise AssertionError(f"Unexpected URL: {url}")

    with _patch_rest_get(mock_rest_get):
        result = await get_all_drug_names("CHEMBL894", cache_dir=tmp_path)

    # Parent (all syn_types): bupropion, bw-323
    # Salts: wellbutrin, zyban, bupropion hydrochloride, aplenzin
    # ("bupropion hydrochloride component of contrave" is filtered)
    assert result == [
        "bupropion",
        "bw-323",
        "wellbutrin",
        "zyban",
        "bupropion hydrochloride",
        "aplenzin",
    ]


async def test_get_all_drug_names_biologic_no_salts(tmp_path):
    """Biologics have trade names on the parent molecule directly."""

    async def mock_rest_get(url: str, params: dict):
        if "/molecule/CHEMBL2108724.json" in url:
            return BIOLOGIC_PARENT_FIXTURE
        if "/molecule.json" in url:
            return BIOLOGIC_HIERARCHY_RESPONSE
        raise AssertionError(f"Unexpected URL: {url}")

    with _patch_rest_get(mock_rest_get):
        result = await get_all_drug_names("CHEMBL2108724", cache_dir=tmp_path)

    # All syn_types from parent: ozempic, rybelsus, wegovy, semaglutide
    # pref_name first, duplicate semaglutide stripped from tail
    assert result == ["semaglutide", "ozempic", "rybelsus", "wegovy"]


async def test_get_all_drug_names_includes_non_trade_name_syn_types(tmp_path):
    """Synonyms tagged RESEARCH_CODE, INN, OTHER, etc. all appear in the list."""

    async def mock_rest_get(url: str, params: dict):
        if "/molecule/CHEMBL894.json" in url:
            return PARENT_FIXTURE
        if "/molecule.json" in url:
            return HIERARCHY_RESPONSE
        raise AssertionError(f"Unexpected URL: {url}")

    with _patch_rest_get(mock_rest_get):
        result = await get_all_drug_names("CHEMBL894", cache_dir=tmp_path)

    assert "bw-323" in result  # RESEARCH_CODE
    assert "bupropion hydrochloride" in result  # INN on salt


async def test_get_all_drug_names_filters_component_of(tmp_path):
    """Entries containing 'component of' are excluded regardless of syn_type."""

    async def mock_rest_get(url: str, params: dict):
        if "/molecule/CHEMBL894.json" in url:
            return PARENT_FIXTURE
        if "/molecule.json" in url:
            return HIERARCHY_RESPONSE
        raise AssertionError(f"Unexpected URL: {url}")

    with _patch_rest_get(mock_rest_get):
        result = await get_all_drug_names("CHEMBL894", cache_dir=tmp_path)

    assert "bupropion hydrochloride component of contrave" not in result


async def test_get_all_drug_names_no_synonyms_no_pref_name(tmp_path):
    """Molecule with no synonyms and no pref_name returns [chembl_id]."""
    bare_fixture = {
        "molecule_chembl_id": "CHEMBL999",
        "pref_name": None,
        "molecule_hierarchy": None,
        "molecule_type": "Small molecule",
        "max_phase": None,
        "atc_classifications": [],
        "black_box_warning": 0,
        "first_approval": None,
        "oral": False,
        "molecule_synonyms": [],
    }

    async def mock_rest_get(url: str, params: dict):
        if "/molecule/CHEMBL999.json" in url:
            return bare_fixture
        if "/molecule.json" in url:
            return {"molecules": []}
        raise AssertionError(f"Unexpected URL: {url}")

    with _patch_rest_get(mock_rest_get):
        result = await get_all_drug_names("CHEMBL999", cache_dir=tmp_path)

    assert result == ["chembl999"]


async def test_get_all_drug_names_hierarchy_failure_still_returns_parent_names(
    tmp_path,
):
    """If hierarchy lookup fails, still returns pref_name and parent synonyms."""

    async def mock_rest_get(url: str, params: dict):
        if "/molecule/CHEMBL2108724.json" in url:
            return BIOLOGIC_PARENT_FIXTURE
        if "/molecule.json" in url:
            raise DataSourceError("chembl", "HTTP 500", 500)
        raise AssertionError(f"Unexpected URL: {url}")

    with _patch_rest_get(mock_rest_get):
        result = await get_all_drug_names("CHEMBL2108724", cache_dir=tmp_path)

    assert result == ["semaglutide", "ozempic", "rybelsus", "wegovy"]


async def test_get_all_drug_names_deduplicates(tmp_path):
    """Same name from parent and salt is not repeated."""
    parent = {
        **BIOLOGIC_PARENT_FIXTURE,
        "molecule_synonyms": [
            {
                "molecule_synonym": "Ozempic",
                "syn_type": "TRADE_NAME",
                "synonyms": "OZEMPIC",
            },
        ],
    }
    hierarchy = {
        "molecules": [
            {
                "molecule_chembl_id": "CHEMBL_SALT",
                "molecule_synonyms": [
                    {
                        "molecule_synonym": "Ozempic",
                        "syn_type": "TRADE_NAME",
                        "synonyms": "OZEMPIC",
                    },
                ],
            },
        ],
    }

    async def mock_rest_get(url: str, params: dict):
        if "/molecule/CHEMBL2108724.json" in url:
            return parent
        if "/molecule.json" in url:
            return hierarchy
        raise AssertionError(f"Unexpected URL: {url}")

    with _patch_rest_get(mock_rest_get):
        result = await get_all_drug_names("CHEMBL2108724", cache_dir=tmp_path)

    assert result == ["semaglutide", "ozempic"]


async def test_get_all_drug_names_writes_cache(tmp_path):
    """Result is written to the `chembl_id_to_names` per-drug cache file."""
    from indication_scout.data_sources.chembl import _load_chembl_names

    async def mock_rest_get(url: str, params: dict):
        if "/molecule/CHEMBL2108724.json" in url:
            return BIOLOGIC_PARENT_FIXTURE
        if "/molecule.json" in url:
            return BIOLOGIC_HIERARCHY_RESPONSE
        raise AssertionError(f"Unexpected URL: {url}")

    with _patch_rest_get(mock_rest_get):
        result = await get_all_drug_names("CHEMBL2108724", cache_dir=tmp_path)

    assert result == ["semaglutide", "ozempic", "rybelsus", "wegovy"]
    cached = _load_chembl_names("CHEMBL2108724", tmp_path)
    assert cached == result


async def test_get_all_drug_names_cache_hit_skips_fetch(tmp_path):
    """Cache hit returns stored value without hitting the API."""
    from indication_scout.data_sources.chembl import _save_chembl_names

    cached_names = ["metformin", "glucophage", "fortamet"]
    _save_chembl_names("CHEMBL1431", cached_names, tmp_path)

    # No _rest_get patch — any API call would explode
    result = await get_all_drug_names("CHEMBL1431", cache_dir=tmp_path)
    assert result == cached_names


async def test_get_all_drug_names_pref_name_not_duplicated_with_trade_name(tmp_path):
    """If pref_name matches a trade name, it appears only once (first position)."""
    fixture = {
        **BIOLOGIC_PARENT_FIXTURE,
        "pref_name": "OZEMPIC",
        "molecule_synonyms": [
            {
                "molecule_synonym": "Ozempic",
                "syn_type": "TRADE_NAME",
                "synonyms": "OZEMPIC",
            },
            {
                "molecule_synonym": "Rybelsus",
                "syn_type": "TRADE_NAME",
                "synonyms": "RYBELSUS",
            },
        ],
    }

    async def mock_rest_get(url: str, params: dict):
        if "/molecule/CHEMBL2108724.json" in url:
            return fixture
        if "/molecule.json" in url:
            return BIOLOGIC_HIERARCHY_RESPONSE
        raise AssertionError(f"Unexpected URL: {url}")

    with _patch_rest_get(mock_rest_get):
        result = await get_all_drug_names("CHEMBL2108724", cache_dir=tmp_path)

    assert result == ["ozempic", "rybelsus"]
    assert result[0] == "ozempic"


# --- resolve_drug_name ---

# OT search response fixtures
OT_SEARCH_HIT_PARENT = {
    "data": {"search": {"hits": [{"id": "CHEMBL1431", "entity": "drug"}]}}
}

OT_SEARCH_HIT_SALT = {
    "data": {"search": {"hits": [{"id": "CHEMBL1703", "entity": "drug"}]}}
}

OT_SEARCH_NO_HITS = {"data": {"search": {"hits": []}}}

# ChEMBL molecule responses for parent normalization
METFORMIN_PARENT_MOLECULE = {
    "molecule_chembl_id": "CHEMBL1431",
    "pref_name": "METFORMIN",
    "molecule_hierarchy": {"parent_chembl_id": "CHEMBL1431"},
    "molecule_type": "Small molecule",
    "max_phase": "4.0",
    "atc_classifications": ["A10BA02"],
    "black_box_warning": 0,
    "first_approval": 1972,
    "oral": True,
    "molecule_synonyms": [],
}

METFORMIN_HCL_SALT_MOLECULE = {
    "molecule_chembl_id": "CHEMBL1703",
    "pref_name": "METFORMIN HYDROCHLORIDE",
    "molecule_hierarchy": {"parent_chembl_id": "CHEMBL1431"},
    "molecule_type": "Small molecule",
    "max_phase": "4.0",
    "atc_classifications": [],
    "black_box_warning": 0,
    "first_approval": None,
    "oral": True,
    "molecule_synonyms": [],
}


async def test_resolve_drug_name_parent_input(tmp_path):
    """Parent molecule input returns its own ChEMBL ID."""

    async def mock_ot_graphql(url, query, variables):
        return OT_SEARCH_HIT_PARENT

    async def mock_chembl_rest_get(url, params):
        if "/molecule/CHEMBL1431.json" in url:
            return METFORMIN_PARENT_MOLECULE
        raise AssertionError(f"Unexpected URL: {url}")

    with (
        patch(
            "indication_scout.data_sources.chembl._OTSearchClient._graphql",
            new=AsyncMock(side_effect=mock_ot_graphql),
        ),
        patch(
            "indication_scout.data_sources.chembl.ChEMBLClient._rest_get",
            new=AsyncMock(side_effect=mock_chembl_rest_get),
        ),
    ):
        result = await resolve_drug_name("metformin", cache_dir=tmp_path)

    assert result == "CHEMBL1431"


async def test_resolve_drug_name_salt_follows_to_parent(tmp_path):
    """Salt input (metformin HCl) follows molecule_hierarchy to parent."""

    async def mock_ot_graphql(url, query, variables):
        return OT_SEARCH_HIT_SALT

    async def mock_chembl_rest_get(url, params):
        if "/molecule/CHEMBL1703.json" in url:
            return METFORMIN_HCL_SALT_MOLECULE
        raise AssertionError(f"Unexpected URL: {url}")

    with (
        patch(
            "indication_scout.data_sources.chembl._OTSearchClient._graphql",
            new=AsyncMock(side_effect=mock_ot_graphql),
        ),
        patch(
            "indication_scout.data_sources.chembl.ChEMBLClient._rest_get",
            new=AsyncMock(side_effect=mock_chembl_rest_get),
        ),
    ):
        result = await resolve_drug_name("metformin hydrochloride", cache_dir=tmp_path)

    assert result == "CHEMBL1431"


async def test_resolve_drug_name_not_found_raises(tmp_path):
    """Unknown drug name raises DataSourceError."""

    async def mock_ot_graphql(url, query, variables):
        return OT_SEARCH_NO_HITS

    with patch(
        "indication_scout.data_sources.chembl._OTSearchClient._graphql",
        new=AsyncMock(side_effect=mock_ot_graphql),
    ):
        with pytest.raises(DataSourceError):
            await resolve_drug_name("notarealdrug", cache_dir=tmp_path)


async def test_resolve_drug_name_caches_result(tmp_path):
    """Second call uses cache, no additional API calls."""

    async def mock_ot_graphql(url, query, variables):
        return OT_SEARCH_HIT_PARENT

    async def mock_chembl_rest_get(url, params):
        if "/molecule/CHEMBL1431.json" in url:
            return METFORMIN_PARENT_MOLECULE
        raise AssertionError(f"Unexpected URL: {url}")

    ot_mock = AsyncMock(side_effect=mock_ot_graphql)
    chembl_mock = AsyncMock(side_effect=mock_chembl_rest_get)

    with (
        patch(
            "indication_scout.data_sources.chembl._OTSearchClient._graphql",
            new=ot_mock,
        ),
        patch(
            "indication_scout.data_sources.chembl.ChEMBLClient._rest_get",
            new=chembl_mock,
        ),
    ):
        first = await resolve_drug_name("metformin", cache_dir=tmp_path)
        second = await resolve_drug_name("metformin", cache_dir=tmp_path)

    assert first == second == "CHEMBL1431"
    # Only one OT search call — second call hits cache
    assert ot_mock.await_count == 1


# --- reverse index via chembl_id_to_names ---


async def test_get_all_drug_names_writes_reverse_index(tmp_path):
    """Every name in the result is reachable via the reverse-index scan."""
    from indication_scout.data_sources.chembl import _lookup_chembl_id_by_name

    async def mock_rest_get(url: str, params: dict):
        if "/molecule/CHEMBL2108724.json" in url:
            return BIOLOGIC_PARENT_FIXTURE
        if "/molecule.json" in url:
            return BIOLOGIC_HIERARCHY_RESPONSE
        raise AssertionError(f"Unexpected URL: {url}")

    with _patch_rest_get(mock_rest_get):
        result = await get_all_drug_names("CHEMBL2108724", cache_dir=tmp_path)

    # Every returned name should resolve back to CHEMBL2108724 via the reverse index
    for name in result:
        cached_id = _lookup_chembl_id_by_name(name, tmp_path)
        assert (
            cached_id == "CHEMBL2108724"
        ), f"Reverse index missing or wrong for {name!r}: got {cached_id!r}"


async def test_resolve_drug_name_uses_reverse_index(tmp_path):
    """A name previously seen by get_all_drug_names resolves without hitting OT."""
    from indication_scout.data_sources.chembl import _save_chembl_names

    # Simulate a previous get_all_drug_names run having stored these names
    _save_chembl_names(
        "CHEMBL2108724", ["semaglutide", "ozempic", "rybelsus"], tmp_path
    )

    ot_mock = AsyncMock()
    chembl_mock = AsyncMock()

    with (
        patch(
            "indication_scout.data_sources.chembl._OTSearchClient._graphql",
            new=ot_mock,
        ),
        patch(
            "indication_scout.data_sources.chembl.ChEMBLClient._rest_get",
            new=chembl_mock,
        ),
    ):
        result = await resolve_drug_name("Ozempic", cache_dir=tmp_path)

    assert result == "CHEMBL2108724"
    # Neither OT nor ChEMBL was touched — reverse index short-circuited
    assert ot_mock.await_count == 0
    assert chembl_mock.await_count == 0


async def test_resolve_drug_name_reverse_index_populates_primary_cache(tmp_path):
    """After a reverse-index hit, the primary resolve_drug_name cache is populated."""
    from indication_scout.data_sources.chembl import _save_chembl_names
    from indication_scout.utils.cache import cache_get

    _save_chembl_names("CHEMBL1431", ["metformin", "glucophage"], tmp_path)

    with (
        patch(
            "indication_scout.data_sources.chembl._OTSearchClient._graphql",
            new=AsyncMock(),
        ),
        patch(
            "indication_scout.data_sources.chembl.ChEMBLClient._rest_get",
            new=AsyncMock(),
        ),
    ):
        await resolve_drug_name("Glucophage", cache_dir=tmp_path)

    # Primary cache now contains the mapping so subsequent lookups skip the reverse index
    cached = cache_get("resolve_drug_name", {"drug_name": "glucophage"}, tmp_path)
    assert cached == "CHEMBL1431"
