"""Unit tests for mechanism_tools.build_mechanism_tools."""

import logging
from unittest.mock import AsyncMock, patch

from langchain_core.messages import ToolCall

from indication_scout.agents.mechanism.mechanism_tools import build_mechanism_tools
from indication_scout.models.model_open_targets import Association, MechanismOfAction

logger = logging.getLogger(__name__)

ASSOCIATION = Association(
    disease_id="EFO_0000400",
    disease_name="type 2 diabetes mellitus",
    overall_score=0.85,
    datatype_scores={"genetic_association": 0.9},
    therapeutic_areas=["metabolism"],
)

MOA = MechanismOfAction(
    mechanism_of_action="Complex I inhibitor",
    action_type="INHIBITOR",
    target_ids=["ENSG00000132356"],
    target_symbols=["PRKAA1"],
)


def _mock_drug(targets=None, mechanisms_of_action=None):
    drug = AsyncMock()
    drug.targets = targets or []
    drug.mechanisms_of_action = mechanisms_of_action or []
    return drug


def _mock_ot_client(**method_returns):
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    for name, return_value in method_returns.items():
        setattr(client, name, AsyncMock(return_value=return_value))
    return client


def _get_tool(name: str):
    tools = build_mechanism_tools()
    for t in tools:
        if t.name == name:
            return t
    raise ValueError(f"Tool '{name}' not found")


# ------------------------------------------------------------------
# get_drug
# ------------------------------------------------------------------


async def test_get_drug_returns_mechanisms_of_action():
    """get_drug artifact is list[MechanismOfAction] and content summarises them."""
    tools = build_mechanism_tools()
    tool_map = {t.name: t for t in tools}

    mock_drug = _mock_drug(
        targets=[AsyncMock(target_symbol="PRKAA1", target_id="ENSG00000132356")],
        mechanisms_of_action=[MOA],
    )
    mock_client = _mock_ot_client(get_drug=mock_drug)

    with patch(
        "indication_scout.agents.mechanism.mechanism_tools.OpenTargetsClient",
        return_value=mock_client,
    ):
        msg = await tool_map["get_drug"].ainvoke(
            ToolCall(name="get_drug", args={"drug_name": "metformin"}, id="tc0", type="tool_call")
        )

    assert isinstance(msg.artifact, list)
    assert len(msg.artifact) == 1
    assert msg.artifact[0].mechanism_of_action == "Complex I inhibitor"
    assert msg.artifact[0].action_type == "INHIBITOR"
    assert "metformin" in msg.content
    assert "INHIBITOR" in msg.content


# ------------------------------------------------------------------
# get_target_associations
# ------------------------------------------------------------------


async def test_get_target_associations_returns_dict_keyed_by_symbol():
    """get_target_associations artifact is {symbol: list[Association]}, not a bare list."""
    tools = build_mechanism_tools()
    tool_map = {t.name: t for t in tools}

    mock_drug = _mock_drug(
        targets=[AsyncMock(target_symbol="PRKAA1", target_id="ENSG00000132356")],
        mechanisms_of_action=[MOA],
    )
    mock_client = _mock_ot_client(get_drug=mock_drug, get_target_data_associations=[ASSOCIATION])

    with patch(
        "indication_scout.agents.mechanism.mechanism_tools.OpenTargetsClient",
        return_value=mock_client,
    ):
        # Populate the store via get_drug
        await tool_map["get_drug"].ainvoke(
            ToolCall(name="get_drug", args={"drug_name": "metformin"}, id="tc1", type="tool_call")
        )
        msg = await tool_map["get_target_associations"].ainvoke(
            ToolCall(name="get_target_associations", args={"target_symbol": "PRKAA1"}, id="tc2", type="tool_call")
        )

    assert isinstance(msg.artifact, dict)
    assert "PRKAA1" in msg.artifact
    assert len(msg.artifact["PRKAA1"]) == 1
    assert msg.artifact["PRKAA1"][0].disease_name == "type 2 diabetes mellitus"
    assert "PRKAA1" in msg.content


async def test_get_target_associations_returns_empty_dict_when_symbol_not_in_store():
    """get_target_associations returns {} artifact when target symbol is not in store."""
    tool = _get_tool("get_target_associations")

    msg = await tool.ainvoke(
        ToolCall(name="get_target_associations", args={"target_symbol": "UNKNOWN"}, id="tc3", type="tool_call")
    )

    assert msg.artifact == {}
    assert "not found" in msg.content


# ------------------------------------------------------------------
# finalize_analysis
# ------------------------------------------------------------------


async def test_finalize_analysis_returns_summary_as_artifact():
    """finalize_analysis returns the summary string as artifact and confirms completion in content."""
    tool = _get_tool("finalize_analysis")

    text = "AMPK activation via PRKAA1 is linked to metabolic and oncological diseases. Colorectal cancer shows strong genetic association scores."
    msg = await tool.ainvoke(
        ToolCall(
            name="finalize_analysis",
            args={"summary": text},
            id="tc4",
            type="tool_call",
        )
    )

    assert msg.artifact == text
    assert "Analysis complete" in msg.content
