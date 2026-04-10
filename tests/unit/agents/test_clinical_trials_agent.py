"""Unit tests for run_clinical_trials_agent output assembly.

The agent itself (create_react_agent) is not invoked — we mock agent.ainvoke
to return a fixed message history and verify that run_clinical_trials_agent
correctly extracts artifacts and the narrative summary into a ClinicalTrialsOutput.
"""

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from indication_scout.agents.clinical_trials.clinical_trials_agent import (
    run_clinical_trials_agent,
)
from indication_scout.agents.clinical_trials.clinical_trials_output import (
    ClinicalTrialsOutput,
)
from indication_scout.models.model_clinical_trials import (
    CompetitorEntry,
    IndicationDrug,
    IndicationLandscape,
    Intervention,
    PrimaryOutcome,
    RecentStart,
    TerminatedTrial,
    Trial,
    WhitespaceResult,
)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Shared test data
# ------------------------------------------------------------------

WHITESPACE = WhitespaceResult(
    is_whitespace=True,
    no_data=False,
    exact_match_count=0,
    drug_only_trials=120,
    indication_only_trials=250,
    indication_drugs=[
        IndicationDrug(
            nct_id="NCT00652457",
            drug_name="Memantine",
            indication="Huntington's Disease",
            phase="Phase 4",
            status="COMPLETED",
        )
    ],
)

LANDSCAPE = IndicationLandscape(
    total_trial_count=30,
    competitors=[
        CompetitorEntry(
            sponsor="Acme Pharma",
            drug_name="SomeDrug",
            drug_type="Drug",
            max_phase="Phase 3",
            trial_count=2,
            statuses={"COMPLETED"},
            total_enrollment=400,
        )
    ],
    phase_distribution={"Phase 2": 10, "Phase 3": 5},
    recent_starts=[
        RecentStart(
            nct_id="NCT09999999",
            sponsor="NewCo",
            drug="NewDrug",
            phase="Phase 2",
        )
    ],
)

TRIALS = [
    Trial(
        nct_id="NCT02970942",
        title="Riluzole ALS Trial",
        brief_summary="Testing riluzole in ALS.",
        phase="Phase 2",
        overall_status="COMPLETED",
        why_stopped=None,
        indications=["ALS"],
        interventions=[
            Intervention(
                intervention_type="Drug",
                intervention_name="Riluzole",
                description="Daily oral dose",
            )
        ],
        sponsor="Sponsor Inc",
        enrollment=100,
        start_date="2015-01-01",
        completion_date="2018-01-01",
        primary_outcomes=[PrimaryOutcome(measure="Survival", time_frame="18 months")],
        references=["12345678"],
    )
]

TERMINATED = [
    TerminatedTrial(
        nct_id="NCT01234567",
        title="Failed Trial",
        drug_name="SomeDrug",
        indication="Huntington's Disease",
        phase="Phase 2",
        why_stopped="Lack of efficacy",
        stop_category="efficacy",
        enrollment=50,
        sponsor="Sponsor Inc",
        start_date="2010-01-01",
        termination_date="2012-06-01",
    )
]

NARRATIVE = (
    "No trials exist for this drug-disease pair. One prior efficacy failure found."
)


def _make_agent(messages: list) -> MagicMock:
    agent = MagicMock()
    agent.ainvoke = AsyncMock(return_value={"messages": messages})
    return agent


def _tool_msg(name: str, artifact) -> ToolMessage:
    return ToolMessage(
        content=f"result of {name}",
        artifact=artifact,
        name=name,
        tool_call_id=f"id_{name}",
    )


# ------------------------------------------------------------------
# Whitespace path: whitespace + terminated + landscape, no trials
# ------------------------------------------------------------------


async def test_run_clinical_trials_agent_whitespace_path():
    """Assembles whitespace, terminated, and landscape correctly; trials stays empty."""
    messages = [
        HumanMessage(content="Analyze somedrug in huntingtons"),
        _tool_msg("detect_whitespace", WHITESPACE),
        _tool_msg("get_terminated", TERMINATED),
        _tool_msg("get_landscape", LANDSCAPE),
        _tool_msg("finalize_analysis", NARRATIVE),
    ]
    agent = _make_agent(messages)

    output = await run_clinical_trials_agent(agent, "somedrug", "huntingtons")

    assert isinstance(output, ClinicalTrialsOutput)

    # whitespace
    assert output.whitespace is not None
    assert output.whitespace.is_whitespace is True
    assert output.whitespace.no_data is False
    assert output.whitespace.exact_match_count == 0
    assert output.whitespace.drug_only_trials == 120
    assert output.whitespace.indication_only_trials == 250
    assert len(output.whitespace.indication_drugs) == 1
    assert output.whitespace.indication_drugs[0].nct_id == "NCT00652457"
    assert output.whitespace.indication_drugs[0].drug_name == "Memantine"
    assert output.whitespace.indication_drugs[0].indication == "Huntington's Disease"
    assert output.whitespace.indication_drugs[0].phase == "Phase 4"
    assert output.whitespace.indication_drugs[0].status == "COMPLETED"

    # terminated
    assert len(output.terminated) == 1
    assert output.terminated[0].nct_id == "NCT01234567"
    assert output.terminated[0].drug_name == "SomeDrug"
    assert output.terminated[0].indication == "Huntington's Disease"
    assert output.terminated[0].phase == "Phase 2"
    assert output.terminated[0].why_stopped == "Lack of efficacy"
    assert output.terminated[0].stop_category == "efficacy"
    assert output.terminated[0].enrollment == 50
    assert output.terminated[0].sponsor == "Sponsor Inc"

    # landscape
    assert output.landscape is not None
    assert output.landscape.total_trial_count == 30
    assert output.landscape.phase_distribution == {"Phase 2": 10, "Phase 3": 5}
    assert len(output.landscape.competitors) == 1
    assert output.landscape.competitors[0].drug_name == "SomeDrug"
    assert output.landscape.competitors[0].max_phase == "Phase 3"
    assert output.landscape.competitors[0].total_enrollment == 400
    assert len(output.landscape.recent_starts) == 1
    assert output.landscape.recent_starts[0].nct_id == "NCT09999999"

    # trials empty
    assert output.trials == []

    # summary
    assert output.summary == NARRATIVE


# ------------------------------------------------------------------
# Active trials path: whitespace + trials + landscape, no terminated
# ------------------------------------------------------------------


async def test_run_clinical_trials_agent_active_trials_path():
    """Assembles trials and landscape correctly; terminated stays empty."""
    active_whitespace = WhitespaceResult(
        is_whitespace=False,
        no_data=False,
        exact_match_count=5,
        drug_only_trials=200,
        indication_only_trials=1000,
        indication_drugs=[],
    )
    messages = [
        HumanMessage(content="Analyze riluzole in als"),
        _tool_msg("detect_whitespace", active_whitespace),
        _tool_msg("search_trials", TRIALS),
        _tool_msg("get_landscape", LANDSCAPE),
        _tool_msg("finalize_analysis", "5 trials found. ALS space is moderately active."),
    ]
    agent = _make_agent(messages)

    output = await run_clinical_trials_agent(agent, "riluzole", "als")

    assert output.whitespace is not None
    assert output.whitespace.is_whitespace is False
    assert output.whitespace.exact_match_count == 5
    assert output.whitespace.indication_drugs == []

    assert len(output.trials) == 1
    trial = output.trials[0]
    assert trial.nct_id == "NCT02970942"
    assert trial.title == "Riluzole ALS Trial"
    assert trial.brief_summary == "Testing riluzole in ALS."
    assert trial.phase == "Phase 2"
    assert trial.overall_status == "COMPLETED"
    assert trial.why_stopped is None
    assert trial.indications == ["ALS"]
    assert len(trial.interventions) == 1
    assert trial.interventions[0].intervention_type == "Drug"
    assert trial.interventions[0].intervention_name == "Riluzole"
    assert trial.interventions[0].description == "Daily oral dose"
    assert trial.sponsor == "Sponsor Inc"
    assert trial.enrollment == 100
    assert trial.start_date == "2015-01-01"
    assert trial.completion_date == "2018-01-01"
    assert len(trial.primary_outcomes) == 1
    assert trial.primary_outcomes[0].measure == "Survival"
    assert trial.primary_outcomes[0].time_frame == "18 months"
    assert trial.references == ["12345678"]

    assert output.landscape is not None
    assert output.landscape.total_trial_count == 30

    assert output.terminated == []
    assert output.summary == "5 trials found. ALS space is moderately active."


# ------------------------------------------------------------------
# Summary extraction: comes from finalize_analysis artifact
# ------------------------------------------------------------------


async def test_run_clinical_trials_agent_summary_from_finalize_analysis():
    """The narrative summary is taken from the finalize_analysis ToolMessage artifact."""
    final_narrative = "Final summary from finalize_analysis."
    messages = [
        HumanMessage(content="Analyze somedrug in huntingtons"),
        _tool_msg("detect_whitespace", WHITESPACE),
        _tool_msg("get_landscape", LANDSCAPE),
        _tool_msg("finalize_analysis", final_narrative),
    ]
    agent = _make_agent(messages)

    output = await run_clinical_trials_agent(agent, "somedrug", "huntingtons")

    assert output.summary == final_narrative


# ------------------------------------------------------------------
# Partial runs — missing tools leave defaults
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "present_tools,missing_field,expected_default",
    [
        (["search_trials", "get_landscape", "get_terminated"], "whitespace", None),
        (["detect_whitespace", "get_landscape", "get_terminated"], "trials", []),
        (["detect_whitespace", "search_trials", "get_terminated"], "landscape", None),
        (["detect_whitespace", "search_trials", "get_landscape"], "terminated", []),
    ],
)
async def test_run_clinical_trials_agent_missing_tool_leaves_default(
    present_tools, missing_field, expected_default
):
    """When a tool's ToolMessage is absent, the corresponding output field stays at its default."""
    artifact_map = {
        "detect_whitespace": WHITESPACE,
        "search_trials": TRIALS,
        "get_landscape": LANDSCAPE,
        "get_terminated": TERMINATED,
    }
    messages = [HumanMessage(content="Analyze somedrug in huntingtons")]
    for name in present_tools:
        messages.append(_tool_msg(name, artifact_map[name]))
    messages.append(_tool_msg("finalize_analysis", NARRATIVE))

    agent = _make_agent(messages)
    output = await run_clinical_trials_agent(agent, "somedrug", "huntingtons")

    assert getattr(output, missing_field) == expected_default
