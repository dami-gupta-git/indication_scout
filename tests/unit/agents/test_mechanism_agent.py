"""Unit tests for mechanism_agent.run_mechanism_agent.

The agent itself (create_react_agent) is not invoked — we mock agent.ainvoke
to return a fixed message history and verify that run_mechanism_agent correctly
extracts artifacts into a MechanismOutput.
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from indication_scout.agents.mechanism.mechanism_agent import run_mechanism_agent
from indication_scout.agents.mechanism.mechanism_output import MechanismOutput
from indication_scout.models.model_open_targets import Association, MechanismOfAction

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def _stub_candidate_assembly():
    """Skip the OT/FDA network work in _assemble_candidates.

    These unit tests only care about message-history plumbing.
    Candidate assembly is covered by test_mechanism_row_builder and
    test_mechanism_candidates; integration tests cover the full path.
    """
    with patch(
        "indication_scout.agents.mechanism.mechanism_agent._assemble_candidates",
        new=AsyncMock(return_value=[]),
    ):
        yield


MECHANISMS_OF_ACTION = [
    MechanismOfAction(
        mechanism_of_action="Complex I inhibitor",
        action_type="INHIBITOR",
        target_ids=["ENSG00000132356", "ENSG00000162409"],
        target_symbols=["PRKAA1", "PRKAA2"],
    )
]

ASSOCIATIONS_PRKAA1 = [
    Association(
        disease_id="EFO_0000400",
        disease_name="type 2 diabetes mellitus",
        overall_score=0.85,
        datatype_scores={"genetic_association": 0.9, "literature": 0.7},
        therapeutic_areas=["metabolism"],
    )
]

NARRATIVE = "PRKAA1 shows strong genetic association with type 2 diabetes. AMPK pathway membership supports metabolic repurposing candidates."


def _make_agent(messages: list) -> MagicMock:
    agent = MagicMock()
    agent.ainvoke = AsyncMock(return_value={"messages": messages})
    return agent


def _tool_msg(name: str, artifact, tool_call_id: str | None = None) -> ToolMessage:
    return ToolMessage(
        content=f"result of {name}",
        artifact=artifact,
        name=name,
        tool_call_id=tool_call_id or f"id_{name}",
    )


async def test_run_mechanism_agent_assembles_all_fields():
    """run_mechanism_agent extracts tool artifacts and populates all MechanismOutput fields."""
    messages = [
        HumanMessage(content="Analyse the targets of metformin"),
        _tool_msg("get_drug", MECHANISMS_OF_ACTION),
        _tool_msg("get_target_associations", {"PRKAA1": ASSOCIATIONS_PRKAA1}),
        _tool_msg("finalize_analysis", NARRATIVE),
    ]
    agent = _make_agent(messages)

    output = await run_mechanism_agent(agent, "metformin")

    assert isinstance(output, MechanismOutput)
    assert output.mechanisms_of_action == MECHANISMS_OF_ACTION
    assert output.drug_targets == {
        "PRKAA1": "ENSG00000132356",
        "PRKAA2": "ENSG00000162409",
    }
    assert output.summary == NARRATIVE


async def test_run_mechanism_agent_ignores_non_tool_messages():
    """AIMessages and HumanMessages in the history do not affect output assembly."""
    messages = [
        HumanMessage(content="Analyse the targets of metformin"),
        AIMessage(content="I will start by fetching the drug."),
        _tool_msg("get_drug", MECHANISMS_OF_ACTION),
        AIMessage(content="Now fetching associations."),
        _tool_msg("get_target_associations", {"PRKAA1": ASSOCIATIONS_PRKAA1}),
        _tool_msg("finalize_analysis", NARRATIVE),
    ]
    agent = _make_agent(messages)

    output = await run_mechanism_agent(agent, "metformin")

    assert output.drug_targets == {
        "PRKAA1": "ENSG00000132356",
        "PRKAA2": "ENSG00000162409",
    }
    assert output.summary == NARRATIVE


@pytest.mark.parametrize(
    "omit_tool,field,default",
    [
        ("get_drug", "mechanisms_of_action", []),
        ("get_drug", "drug_targets", {}),
        ("finalize_analysis", "summary", ""),
    ],
)
async def test_run_mechanism_agent_missing_tool_leaves_default(
    omit_tool, field, default
):
    """When a tool's ToolMessage is absent, the corresponding output field stays at its default."""
    all_messages = [
        HumanMessage(content="Analyse the targets of metformin"),
        _tool_msg("get_drug", MECHANISMS_OF_ACTION),
        _tool_msg("get_target_associations", {"PRKAA1": ASSOCIATIONS_PRKAA1}),
        _tool_msg("finalize_analysis", NARRATIVE),
    ]
    messages = [
        m
        for m in all_messages
        if not (isinstance(m, ToolMessage) and m.name == omit_tool)
    ]
    agent = _make_agent(messages)

    output = await run_mechanism_agent(agent, "metformin")

    assert getattr(output, field) == default
