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
    FinalizeClinicalTrialsArtifact,
)
from indication_scout.models.model_clinical_trials import (
    ApprovalCheck,
    CompetitorEntry,
    CompletedTrialsResult,
    IndicationLandscape,
    Intervention,
    PrimaryOutcome,
    RecentStart,
    SearchTrialsResult,
    TerminatedTrialsResult,
    Trial,
)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Shared test data
# ------------------------------------------------------------------

SEARCH = SearchTrialsResult(
    total_count=0,
    by_status={"RECRUITING": 0, "ACTIVE_NOT_RECRUITING": 0, "WITHDRAWN": 0},
    trials=[],
)

ACTIVE_SEARCH = SearchTrialsResult(
    total_count=5,
    by_status={"RECRUITING": 3, "ACTIVE_NOT_RECRUITING": 1, "WITHDRAWN": 0},
    trials=[
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
            primary_outcomes=[
                PrimaryOutcome(measure="Survival", time_frame="18 months")
            ],
            references=["12345678"],
        )
    ],
)

COMPLETED = CompletedTrialsResult(
    total_count=8,
    trials=[
        Trial(
            nct_id="NCT04111111",
            title="Phase 3 Completed Trial",
            phase="Phase 3",
            overall_status="COMPLETED",
            sponsor="Sponsor Inc",
            enrollment=500,
        )
    ],
)

TERMINATED = TerminatedTrialsResult(
    total_count=1,
    trials=[
        Trial(
            nct_id="NCT01234567",
            title="Failed Trial",
            phase="Phase 2",
            overall_status="TERMINATED",
            why_stopped="Lack of efficacy",
            sponsor="Sponsor Inc",
            enrollment=50,
            start_date="2010-01-01",
            completion_date="2012-06-01",
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

NARRATIVE = (
    "No trials exist for this drug-disease pair. One prior efficacy failure found."
)

APPROVAL = ApprovalCheck(
    is_approved=True,
    label_found=True,
    matched_indication="type 2 diabetes mellitus",
    drug_names_checked=["semaglutide", "ozempic", "wegovy", "rybelsus"],
)


def _make_agent(messages: list) -> MagicMock:
    agent = MagicMock()
    agent.ainvoke = AsyncMock(return_value={"messages": messages})
    return agent


def _tool_msg(name: str, artifact) -> ToolMessage:
    # finalize_analysis now returns a FinalizeClinicalTrialsArtifact, not a bare string.
    # Accept a plain summary string for brevity and wrap it so existing call sites still work.
    if name == "finalize_analysis" and isinstance(artifact, str):
        artifact = FinalizeClinicalTrialsArtifact(summary=artifact)
    return ToolMessage(
        content=f"result of {name}",
        artifact=artifact,
        name=name,
        tool_call_id=f"id_{name}",
    )


# ------------------------------------------------------------------
# Whitespace path: search (total=0) + terminated + landscape, no completed
# ------------------------------------------------------------------


async def test_run_clinical_trials_agent_whitespace_path():
    """Assembles search (zero), terminated, landscape correctly; completed stays None."""
    messages = [
        HumanMessage(content="Analyze somedrug in huntingtons"),
        _tool_msg("search_trials", SEARCH),
        _tool_msg("get_terminated", TERMINATED),
        _tool_msg("get_landscape", LANDSCAPE),
        _tool_msg("finalize_analysis", NARRATIVE),
    ]
    agent = _make_agent(messages)

    output = await run_clinical_trials_agent(agent, "somedrug", "huntingtons")

    assert isinstance(output, ClinicalTrialsOutput)

    # search: empty / whitespace
    assert isinstance(output.search, SearchTrialsResult)
    assert output.search.total_count == 0
    assert output.search.by_status == {
        "RECRUITING": 0,
        "ACTIVE_NOT_RECRUITING": 0,
        "WITHDRAWN": 0,
    }
    assert output.search.trials == []

    # terminated
    assert isinstance(output.terminated, TerminatedTrialsResult)
    assert output.terminated.total_count == 1
    assert len(output.terminated.trials) == 1
    t = output.terminated.trials[0]
    assert t.nct_id == "NCT01234567"
    assert t.title == "Failed Trial"
    assert t.phase == "Phase 2"
    assert t.why_stopped == "Lack of efficacy"
    assert t.enrollment == 50
    assert t.sponsor == "Sponsor Inc"

    # landscape
    assert isinstance(output.landscape, IndicationLandscape)
    assert output.landscape.total_trial_count == 30
    assert output.landscape.phase_distribution == {"Phase 2": 10, "Phase 3": 5}
    assert len(output.landscape.competitors) == 1
    assert output.landscape.competitors[0].drug_name == "SomeDrug"
    assert output.landscape.competitors[0].max_phase == "Phase 3"
    assert output.landscape.competitors[0].total_enrollment == 400
    assert len(output.landscape.recent_starts) == 1
    assert output.landscape.recent_starts[0].nct_id == "NCT09999999"

    # completed and approval not called → stay None
    assert output.completed is None
    assert output.approval is None

    # summary
    assert output.summary == NARRATIVE


# ------------------------------------------------------------------
# Active path: search + completed + landscape, no terminated
# ------------------------------------------------------------------


async def test_run_clinical_trials_agent_active_trials_path():
    """Assembles search + completed + landscape correctly; terminated stays None."""
    messages = [
        HumanMessage(content="Analyze riluzole in als"),
        _tool_msg("search_trials", ACTIVE_SEARCH),
        _tool_msg("get_completed", COMPLETED),
        _tool_msg("get_landscape", LANDSCAPE),
        _tool_msg(
            "finalize_analysis", "5 trials found. ALS space is moderately active."
        ),
    ]
    agent = _make_agent(messages)

    output = await run_clinical_trials_agent(agent, "riluzole", "als")

    assert isinstance(output.search, SearchTrialsResult)
    assert output.search.total_count == 5
    assert output.search.by_status == {
        "RECRUITING": 3,
        "ACTIVE_NOT_RECRUITING": 1,
        "WITHDRAWN": 0,
    }
    assert len(output.search.trials) == 1
    trial = output.search.trials[0]
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

    assert isinstance(output.completed, CompletedTrialsResult)
    assert output.completed.total_count == 8
    assert len(output.completed.trials) == 1
    assert output.completed.trials[0].nct_id == "NCT04111111"

    assert isinstance(output.landscape, IndicationLandscape)
    assert output.landscape.total_trial_count == 30

    assert output.terminated is None
    assert output.summary == "5 trials found. ALS space is moderately active."


# ------------------------------------------------------------------
# Summary extraction: comes from finalize_analysis artifact
# ------------------------------------------------------------------


async def test_run_clinical_trials_agent_summary_from_finalize_analysis():
    """The narrative summary is taken from the finalize_analysis ToolMessage artifact."""
    final_narrative = "Final summary from finalize_analysis."
    messages = [
        HumanMessage(content="Analyze somedrug in huntingtons"),
        _tool_msg("search_trials", SEARCH),
        _tool_msg("get_landscape", LANDSCAPE),
        _tool_msg("finalize_analysis", final_narrative),
    ]
    agent = _make_agent(messages)

    output = await run_clinical_trials_agent(agent, "somedrug", "huntingtons")

    assert output.summary == final_narrative


# ------------------------------------------------------------------
# Approval path: check_fda_approval artifact is threaded into output.approval
# ------------------------------------------------------------------


async def test_run_clinical_trials_agent_approval_path():
    """The check_fda_approval ToolMessage artifact is assigned to output.approval."""
    messages = [
        HumanMessage(content="Analyze semaglutide in type 2 diabetes mellitus"),
        _tool_msg("search_trials", SEARCH),
        _tool_msg("get_landscape", LANDSCAPE),
        _tool_msg("check_fda_approval", APPROVAL),
        _tool_msg(
            "finalize_analysis",
            "Semaglutide is FDA-approved for type 2 diabetes mellitus.",
        ),
    ]
    agent = _make_agent(messages)

    output = await run_clinical_trials_agent(
        agent, "semaglutide", "type 2 diabetes mellitus"
    )

    assert isinstance(output.approval, ApprovalCheck)
    assert output.approval.is_approved is True
    assert output.approval.label_found is True
    assert output.approval.matched_indication == "type 2 diabetes mellitus"
    assert output.approval.drug_names_checked == [
        "semaglutide",
        "ozempic",
        "wegovy",
        "rybelsus",
    ]
    assert output.summary == "Semaglutide is FDA-approved for type 2 diabetes mellitus."


async def test_run_clinical_trials_agent_approval_defaults_to_none_when_absent():
    """When check_fda_approval is never called, output.approval stays None."""
    messages = [
        HumanMessage(content="Analyze somedrug in huntingtons"),
        _tool_msg("search_trials", SEARCH),
        _tool_msg("get_landscape", LANDSCAPE),
        _tool_msg("get_terminated", TERMINATED),
        _tool_msg("finalize_analysis", NARRATIVE),
    ]
    agent = _make_agent(messages)

    output = await run_clinical_trials_agent(agent, "somedrug", "huntingtons")

    assert output.approval is None


# ------------------------------------------------------------------
# Partial runs — missing tools leave defaults (None for absent artifacts)
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "present_tools,missing_field",
    [
        (
            ["get_completed", "get_landscape", "get_terminated", "check_fda_approval"],
            "search",
        ),
        (
            ["search_trials", "get_landscape", "get_terminated", "check_fda_approval"],
            "completed",
        ),
        (
            ["search_trials", "get_completed", "get_terminated", "check_fda_approval"],
            "landscape",
        ),
        (
            ["search_trials", "get_completed", "get_landscape", "check_fda_approval"],
            "terminated",
        ),
        (
            ["search_trials", "get_completed", "get_landscape", "get_terminated"],
            "approval",
        ),
    ],
)
async def test_run_clinical_trials_agent_missing_tool_leaves_default(
    present_tools, missing_field
):
    """When a tool's ToolMessage is absent, the corresponding output field stays None."""
    artifact_map = {
        "search_trials": SEARCH,
        "get_completed": COMPLETED,
        "get_landscape": LANDSCAPE,
        "get_terminated": TERMINATED,
        "check_fda_approval": APPROVAL,
    }
    messages = [HumanMessage(content="Analyze somedrug in huntingtons")]
    for name in present_tools:
        messages.append(_tool_msg(name, artifact_map[name]))
    messages.append(_tool_msg("finalize_analysis", NARRATIVE))

    agent = _make_agent(messages)
    output = await run_clinical_trials_agent(agent, "somedrug", "huntingtons")

    assert getattr(output, missing_field) is None


# ------------------------------------------------------------------
# Relevance assembly: finalize artifact populates relevance fields + filtered signals
# ------------------------------------------------------------------


async def test_relevance_fields_and_filtered_signals_assembled():
    """The finalize artifact's relevance split flows onto the output, and signals are computed
    from RELEVANT trials only — a contaminating completed Phase 3 is excluded."""
    completed = CompletedTrialsResult(
        total_count=2,
        trials=[
            Trial(nct_id="NCT_REL", phase="Phase 2", overall_status="COMPLETED"),
            Trial(nct_id="NCT_CONTAM", phase="Phase 3", overall_status="COMPLETED"),
        ],
    )
    finalize = FinalizeClinicalTrialsArtifact(
        summary="Phase 2 on record; the Phase 3 is a different disease.",
        relevant_ncts=["NCT_REL"],
        contaminated_ncts=["NCT_CONTAM"],
        relevance_reasoning="NCT_CONTAM is a distinct disease sharing a parent term.",
    )
    messages = [
        HumanMessage(content="Analyze somedrug in somedisease"),
        _tool_msg("search_trials", SEARCH),
        _tool_msg("get_completed", completed),
        _tool_msg("get_landscape", LANDSCAPE),
        _tool_msg("finalize_analysis", finalize),
    ]
    agent = _make_agent(messages)

    output = await run_clinical_trials_agent(agent, "somedrug", "somedisease")

    assert output.relevant_nct_ids == ["NCT_REL"]
    assert output.contaminated_nct_ids == ["NCT_CONTAM"]
    assert (
        output.relevance_reasoning
        == "NCT_CONTAM is a distinct disease sharing a parent term."
    )
    # Signals computed from the RELEVANT set only: the contaminating Phase 3 is dropped.
    assert output.signals is not None
    assert output.signals.highest_completed_phase == "Phase 2"
    assert output.signals.has_completed_phase3 is False
    assert output.signals.completed_phase3_nct_ids == []


async def test_first_approval_reaches_agent_task_message():
    """first_approval is surfaced in the agent's task message so the closure judge can read it."""
    messages = [
        HumanMessage(content="Analyze bupropion in adhd"),
        _tool_msg("search_trials", SEARCH),
        _tool_msg("get_landscape", LANDSCAPE),
        _tool_msg("finalize_analysis", NARRATIVE),
    ]
    agent = _make_agent(messages)

    await run_clinical_trials_agent(agent, "bupropion", "adhd", first_approval=1985)

    sent = agent.ainvoke.call_args.args[0]["messages"][0].content
    assert "first_approval" in sent
    assert "1985" in sent


async def test_unknown_first_approval_passed_as_literal():
    """When first_approval is None, the literal 'unknown' is passed — never a default year."""
    messages = [
        HumanMessage(content="Analyze somedrug in somedisease"),
        _tool_msg("search_trials", SEARCH),
        _tool_msg("get_landscape", LANDSCAPE),
        _tool_msg("finalize_analysis", NARRATIVE),
    ]
    agent = _make_agent(messages)

    await run_clinical_trials_agent(agent, "somedrug", "somedisease")

    sent = agent.ainvoke.call_args.args[0]["messages"][0].content
    assert "first_approval" in sent
    assert "unknown" in sent
