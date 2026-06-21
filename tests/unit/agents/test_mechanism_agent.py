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
def _stub_candidate_assembly(request):
    """Skip the OT/FDA network work in _assemble_candidates.

    These unit tests only care about message-history plumbing.
    Candidate assembly is covered by test_mechanism_row_builder and
    test_mechanism_candidates; integration tests cover the full path.

    Tests that exercise _assemble_candidates directly request the
    `_patch_assemble_deps` fixture and opt out of this stub.
    """
    if "_patch_assemble_deps" in request.fixturenames:
        yield
        return
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
        HumanMessage(content="Analyze the targets of metformin"),
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
        HumanMessage(content="Analyze the targets of metformin"),
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
        HumanMessage(content="Analyze the targets of metformin"),
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


# --- _assemble_candidates FDA approval gating (holdout) --------------------

from datetime import date  # noqa: E402

from indication_scout.agents.mechanism import mechanism_agent as _mech  # noqa: E402

_ROWS = [{"disease_name": "systemic mastocytosis"}, {"disease_name": "glioblastoma"}]


@pytest.fixture
def _patch_assemble_deps():
    """Stub OT network + row builder so _assemble_candidates reaches the FDA filter.

    Yields (live_mock, table_mock) so each test can assert which path ran.
    select_top_candidates is stubbed to echo the approved set back for assertion.
    """
    live = AsyncMock(
        return_value={"systemic mastocytosis": "approved", "glioblastoma": "not_approved"}
    )
    table = AsyncMock(return_value={"systemic mastocytosis"})
    captured: dict = {}

    def _capture(rows, approved_diseases, limit):
        captured["approved"] = approved_diseases
        return []

    class _DummyOT:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    with (
        patch.object(_mech, "build_candidate_rows", new=AsyncMock(return_value=_ROWS)),
        patch.object(_mech, "OpenTargetsClient", new=lambda *a, **k: _DummyOT()),
        patch.object(_mech, "get_fda_approved_disease_mapping", new=live),
        patch.object(_mech, "get_approved_indications", new=table),
        patch.object(_mech, "select_top_candidates", new=_capture),
    ):
        yield live, table, captured


async def test_assemble_candidates_uses_date_gated_table_in_holdout(
    _patch_assemble_deps,
):
    """With date_before set, the date-gated table is used and live FDA is NOT called."""
    live, table, captured = _patch_assemble_deps
    cutoff = date(2002, 6, 1)

    await _mech._assemble_candidates(
        "imatinib",
        {"KIT": "ENSG00000157404"},
        MECHANISMS_OF_ACTION,
        date_before=cutoff,
    )

    live.assert_not_awaited()
    table.assert_called_once()
    assert table.call_args.kwargs["as_of"] == cutoff
    assert captured["approved"] == {"systemic mastocytosis"}


async def test_assemble_candidates_uses_live_fda_when_no_cutoff(_patch_assemble_deps):
    """With date_before None, the live FDA mapping is used and the table is NOT."""
    live, table, captured = _patch_assemble_deps

    await _mech._assemble_candidates(
        "imatinib",
        {"KIT": "ENSG00000157404"},
        MECHANISMS_OF_ACTION,
        date_before=None,
    )

    live.assert_awaited_once()
    table.assert_not_called()
    assert captured["approved"] == {"systemic mastocytosis"}
