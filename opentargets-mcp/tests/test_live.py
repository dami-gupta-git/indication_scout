"""One smoke test per tool against the live Open Targets API. Excluded from a default run; use `pytest -m live`."""

import pytest

from opentargets_mcp import tools
from opentargets_mcp.transport import OpenTargetsError

pytestmark = pytest.mark.live


async def test_resolve_finds_a_gene_a_disease_and_a_drug() -> None:
    assert "ENSG00000162434" in await tools.tool_resolve("JAK1", "target")
    assert "MONDO_0005083" in await tools.tool_resolve("psoriasis", "disease")
    assert "CHEMBL2105759" in await tools.tool_resolve("baricitinib", "drug")


async def test_resolve_reports_no_match() -> None:
    assert "no match" in await tools.tool_resolve("notagene123", "target")


async def test_target_profile_carries_identity_and_tractability() -> None:
    result = await tools.tool_target_profile("JAK1")
    assert "Janus kinase 1" in result
    assert "Tractability" in result


async def test_target_profile_lists_safety_liabilities_where_recorded() -> None:
    assert "prolongation of QT interval" in await tools.tool_target_profile("KCNH2")


async def test_target_diseases_is_capped_and_says_so() -> None:
    result = await tools.tool_target_diseases("JAK1", 5)
    assert "Showing 5 of" in result
    assert result.count("\n|") >= 5


async def test_disease_targets_returns_scored_rows() -> None:
    result = await tools.tool_disease_targets("psoriasis", 5)
    assert "Showing 5 of" in result
    assert "ENSG" in result


async def test_known_drugs_from_a_target() -> None:
    result = await tools.tool_known_drugs("JAK1", "target", 5)
    assert "CHEMBL" in result
    assert "APPROVAL" in result


async def test_known_drugs_from_a_disease() -> None:
    assert "CHEMBL" in await tools.tool_known_drugs("psoriasis", "disease", 5)


async def test_evidence_returns_rows_for_a_pair() -> None:
    result = await tools.tool_evidence("JAK1", "psoriasis", 5)
    assert "evidence rows" in result
    assert "clinical" in result


async def test_unresolvable_name_raises() -> None:
    with pytest.raises(OpenTargetsError):
        await tools.tool_target_diseases("notagene123", 5)
