"""Unit tests for mechanism_agent: run_mechanism_agent and _compute_shaped_associations.

The agent itself (create_react_agent) is not invoked — we mock agent.ainvoke
to return a fixed message history and verify that run_mechanism_agent correctly
extracts artifacts into a MechanismOutput, and that _compute_shaped_associations
produces correct shapes from datatype_scores.
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from indication_scout.agents.mechanism.mechanism_agent import (
    _compute_shaped_associations,
    run_mechanism_agent,
)
from indication_scout.agents.mechanism.mechanism_output import MechanismOutput, ShapedAssociation
from indication_scout.models.model_open_targets import Association, MechanismOfAction

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Shared test data
# ------------------------------------------------------------------

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


def _mock_ot_context():
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get_target_data_pathways = AsyncMock(return_value=[])
    return mock_client


# ------------------------------------------------------------------
# run_mechanism_agent — output assembly
# ------------------------------------------------------------------


async def test_run_mechanism_agent_assembles_all_fields():
    """run_mechanism_agent extracts tool artifacts and populates all MechanismOutput fields."""
    messages = [
        HumanMessage(content="Analyse the targets of metformin"),
        _tool_msg("get_drug", MECHANISMS_OF_ACTION),
        _tool_msg("get_target_associations", {"PRKAA1": ASSOCIATIONS_PRKAA1}),
        _tool_msg("finalize_analysis", NARRATIVE),
    ]
    agent = _make_agent(messages)

    with patch(
        "indication_scout.agents.mechanism.mechanism_agent.OpenTargetsClient",
        return_value=_mock_ot_context(),
    ):
        output = await run_mechanism_agent(agent, "metformin")

    assert isinstance(output, MechanismOutput)
    assert output.mechanisms_of_action == MECHANISMS_OF_ACTION
    assert output.drug_targets == {"PRKAA1": "ENSG00000132356", "PRKAA2": "ENSG00000162409"}
    assert output.associations == {"PRKAA1": ASSOCIATIONS_PRKAA1}
    assert output.summary == NARRATIVE
    # shaped_associations are computed deterministically — at least one entry for PRKAA1
    assert len(output.shaped_associations) == 1
    assert output.shaped_associations[0].target_symbol == "PRKAA1"
    assert output.shaped_associations[0].disease_id == "EFO_0000400"


async def test_run_mechanism_agent_multiple_targets_accumulate():
    """associations from multiple per-target calls are merged into a single dict."""
    associations_prkaa2 = [
        Association(
            disease_id="EFO_0000270",
            disease_name="asthma",
            overall_score=0.4,
            datatype_scores={"literature": 0.4},
            therapeutic_areas=["respiratory"],
        )
    ]

    messages = [
        HumanMessage(content="Analyse the targets of metformin"),
        _tool_msg("get_drug", MECHANISMS_OF_ACTION),
        _tool_msg("get_target_associations", {"PRKAA1": ASSOCIATIONS_PRKAA1}),
        _tool_msg("get_target_associations", {"PRKAA2": associations_prkaa2}),
        _tool_msg("finalize_analysis", NARRATIVE),
    ]
    agent = _make_agent(messages)

    with patch(
        "indication_scout.agents.mechanism.mechanism_agent.OpenTargetsClient",
        return_value=_mock_ot_context(),
    ):
        output = await run_mechanism_agent(agent, "metformin")

    assert output.associations == {
        "PRKAA1": ASSOCIATIONS_PRKAA1,
        "PRKAA2": associations_prkaa2,
    }


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

    with patch(
        "indication_scout.agents.mechanism.mechanism_agent.OpenTargetsClient",
        return_value=_mock_ot_context(),
    ):
        output = await run_mechanism_agent(agent, "metformin")

    assert output.drug_targets == {"PRKAA1": "ENSG00000132356", "PRKAA2": "ENSG00000162409"}
    assert output.summary == NARRATIVE


@pytest.mark.parametrize(
    "omit_tool,field,default",
    [
        ("get_drug", "mechanisms_of_action", []),
        ("get_drug", "drug_targets", {}),
        ("get_target_associations", "associations", {}),
        ("finalize_analysis", "summary", ""),
    ],
)
async def test_run_mechanism_agent_missing_tool_leaves_default(omit_tool, field, default):
    """When a tool's ToolMessage is absent, the corresponding output field stays at its default."""
    all_messages = [
        HumanMessage(content="Analyse the targets of metformin"),
        _tool_msg("get_drug", MECHANISMS_OF_ACTION),
        _tool_msg("get_target_associations", {"PRKAA1": ASSOCIATIONS_PRKAA1}),
        _tool_msg("finalize_analysis", NARRATIVE),
    ]
    messages = [m for m in all_messages if not (isinstance(m, ToolMessage) and m.name == omit_tool)]
    agent = _make_agent(messages)

    with patch(
        "indication_scout.agents.mechanism.mechanism_agent.OpenTargetsClient",
        return_value=_mock_ot_context(),
    ):
        output = await run_mechanism_agent(agent, "metformin")

    assert getattr(output, field) == default


# ------------------------------------------------------------------
# _compute_shaped_associations
# ------------------------------------------------------------------


def _inhibitor_moa(symbol: str) -> MechanismOfAction:
    return MechanismOfAction(
        mechanism_of_action="inhibitor",
        action_type="INHIBITOR",
        target_ids=["ENSG000001"],
        target_symbols=[symbol],
    )


def _activator_moa(symbol: str) -> MechanismOfAction:
    return MechanismOfAction(
        mechanism_of_action="activator",
        action_type="ACTIVATOR",
        target_ids=["ENSG000002"],
        target_symbols=[symbol],
    )


def _assoc(disease_id: str, disease_name: str, **scores) -> Association:
    return Association(
        disease_id=disease_id,
        disease_name=disease_name,
        overall_score=0.5,
        datatype_scores=scores,
        therapeutic_areas=[],
    )


def test_compute_shaped_confirms_known_when_clinical_dominates():
    """High clinical + low genetic → confirms_known regardless of action type."""
    moas = [_inhibitor_moa("EGFR")]
    assocs = {"EGFR": [_assoc("EFO_001", "non-small cell lung carcinoma", clinical=0.8, genetic_association=0.1)]}

    shaped = _compute_shaped_associations(moas, assocs)

    assert len(shaped) == 1
    assert shaped[0].shape == "confirms_known"
    assert shaped[0].disease_id == "EFO_001"
    assert shaped[0].disease_name == "non-small cell lung carcinoma"
    assert shaped[0].target_symbol == "EGFR"


def test_compute_shaped_hypothesis_inhibitor_plus_gof():
    """INHIBITOR + somatic_mutation ≥ 0.4 (GOF disease) → hypothesis."""
    moas = [_inhibitor_moa("BRAF")]
    assocs = {"BRAF": [_assoc("EFO_002", "melanoma", somatic_mutation=0.7)]}

    shaped = _compute_shaped_associations(moas, assocs)

    assert len(shaped) == 1
    assert shaped[0].shape == "hypothesis"


def test_compute_shaped_contraindication_inhibitor_plus_lof():
    """INHIBITOR + LOF disease name → contraindication."""
    moas = [_inhibitor_moa("PRKAA1")]
    assocs = {"PRKAA1": [_assoc("EFO_003", "AMPK deficiency syndrome", genetic_association=0.5)]}

    shaped = _compute_shaped_associations(moas, assocs)

    assert len(shaped) == 1
    assert shaped[0].shape == "contraindication"


def test_compute_shaped_hypothesis_activator_plus_lof():
    """ACTIVATOR + LOF disease name → hypothesis."""
    moas = [_activator_moa("GBA")]
    assocs = {"GBA": [_assoc("EFO_004", "glucocerebrosidase deficiency", genetic_association=0.6)]}

    shaped = _compute_shaped_associations(moas, assocs)

    assert len(shaped) == 1
    assert shaped[0].shape == "hypothesis"


def test_compute_shaped_neutral_when_no_directional_signal():
    """No somatic/pathway/LOF-name signals → neutral."""
    moas = [_inhibitor_moa("TARGET1")]
    assocs = {"TARGET1": [_assoc("EFO_005", "some disease", literature=0.5)]}

    shaped = _compute_shaped_associations(moas, assocs)

    assert len(shaped) == 1
    assert shaped[0].shape == "neutral"


def test_compute_shaped_pathway_treated_as_gof():
    """affected_pathway ≥ 0.4 with low somatic → GOF-like → inhibitor = hypothesis."""
    moas = [_inhibitor_moa("MTOR")]
    assocs = {"MTOR": [_assoc("EFO_006", "tuberous sclerosis", affected_pathway=0.6, somatic_mutation=0.1)]}

    shaped = _compute_shaped_associations(moas, assocs)

    assert len(shaped) == 1
    assert shaped[0].shape == "hypothesis"


def test_compute_shaped_empty_when_no_associations():
    """No associations → empty list."""
    moas = [_inhibitor_moa("TARGET1")]
    shaped = _compute_shaped_associations(moas, {})
    assert shaped == []


def test_compute_shaped_all_fields_populated():
    """All ShapedAssociation fields are populated for a computed entry."""
    moas = [_inhibitor_moa("BRAF")]
    assocs = {"BRAF": [_assoc("EFO_007", "colorectal cancer", somatic_mutation=0.5)]}

    shaped = _compute_shaped_associations(moas, assocs)

    assert len(shaped) == 1
    s = shaped[0]
    assert s.target_symbol == "BRAF"
    assert s.disease_name == "colorectal cancer"
    assert s.disease_id == "EFO_007"
    assert s.shape == "hypothesis"
    assert len(s.rationale) > 0
