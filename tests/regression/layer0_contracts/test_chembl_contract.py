"""ChEMBL client contract: molecule and ATC lookups parse into their models as recorded."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from indication_scout.data_sources.chembl import ChEMBLClient

pytestmark = pytest.mark.contract


async def test_get_molecule_parses_metformin(
    cassette: Callable[[str], Iterator[None]], contract_cache_dir: Path
) -> None:
    with cassette("chembl_molecule"):
        async with ChEMBLClient(cache_dir=contract_cache_dir) as client:
            molecule = await client.get_molecule("CHEMBL1431")

    assert molecule.molecule_chembl_id == "CHEMBL1431"
    assert molecule.pref_name == "metformin"
    assert molecule.parent_chembl_id == "CHEMBL1431"
    assert molecule.molecule_type == "Small molecule"
    assert molecule.max_phase == "4.0"
    assert molecule.atc_classifications == ["A10BA02"]
    assert molecule.black_box_warning == 1
    assert molecule.first_approval == 1995
    assert molecule.oral is True

    assert len(molecule.molecule_synonyms) == 10
    inn = [s for s in molecule.molecule_synonyms if s.syn_type == "INN"]
    assert len(inn) == 1
    assert inn[0].molecule_synonym == "metformin"
    assert inn[0].synonyms == "metformin"


async def test_get_atc_description_parses_full_hierarchy(
    cassette: Callable[[str], Iterator[None]], contract_cache_dir: Path
) -> None:
    with cassette("chembl_atc"):
        async with ChEMBLClient(cache_dir=contract_cache_dir) as client:
            atc = await client.get_atc_description("A10BG03")

    assert atc.level1 == "A"
    assert atc.level1_description == "ALIMENTARY TRACT AND METABOLISM"
    assert atc.level2 == "A10"
    assert atc.level2_description == "DRUGS USED IN DIABETES"
    assert atc.level3 == "A10B"
    assert atc.level3_description == "BLOOD GLUCOSE LOWERING DRUGS, EXCL. INSULINS"
    assert atc.level4 == "A10BG"
    assert atc.level4_description == "Thiazolidinediones"
    assert atc.level5 == "A10BG03"
    assert atc.who_name == "pioglitazone"
