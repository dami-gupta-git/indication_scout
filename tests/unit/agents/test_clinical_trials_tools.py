"""Unit tests for clinical_trials_tools.build_clinical_trials_tools."""

import logging
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import ToolCall as LCToolCall

from indication_scout.agents.clinical_trials.clinical_trials_tools import (
    build_clinical_trials_tools,
)
from indication_scout.data_sources.base_client import DataSourceError
from indication_scout.models.model_clinical_trials import (
    ApprovalCheck,
    CompetitorEntry,
    CompletedTrialsResult,
    IndicationLandscape,
    Intervention,
    MeshTerm,
    PrimaryOutcome,
    RecentStart,
    SearchTrialsResult,
    TerminatedTrialsResult,
    Trial,
)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _mock_client(**method_returns) -> AsyncMock:
    """Build an AsyncMock that works as an async context manager."""
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    for name, return_value in method_returns.items():
        setattr(client, name, AsyncMock(return_value=return_value))
    return client


def _get_tool(tools: list, name: str):
    for t in tools:
        if t.name == name:
            return t
    raise ValueError(f"Tool '{name}' not found")


# ------------------------------------------------------------------
# search_trials
# ------------------------------------------------------------------


async def test_search_trials_returns_search_trials_result_artifact():
    """search_trials returns a SearchTrialsResult artifact with all fields intact."""
    trial = Trial(
        nct_id="NCT00127933",
        title="XeNA Study",
        phase="Phase 4",
        overall_status="COMPLETED",
        why_stopped=None,
        indications=["Breast Cancer"],
        mesh_conditions=[MeshTerm(id="D001943", term="Breast Neoplasms")],
        interventions=[
            Intervention(intervention_type="Drug", intervention_name="Trastuzumab"),
            Intervention(intervention_type="Drug", intervention_name="Capecitabine"),
        ],
        sponsor="Hoffmann-La Roche",
        enrollment=157,
        start_date="2005-08",
        completion_date="2009-04",
        primary_outcomes=[PrimaryOutcome(measure="pCR rate", time_frame="4 cycles")],
        references=[],
    )
    mock_result = SearchTrialsResult(
        total_count=1,
        by_status={"RECRUITING": 0, "ACTIVE_NOT_RECRUITING": 0, "WITHDRAWN": 0},
        trials=[trial],
    )

    mock_client = _mock_client(search_trials=mock_result)
    tools = build_clinical_trials_tools(date_before=None)

    with (
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.resolve_mesh_id",
            new=AsyncMock(return_value=("D001943", "Breast Neoplasms")),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.ClinicalTrialsClient",
            return_value=mock_client,
        ),
    ):
        msg = await _get_tool(tools, "search_trials").ainvoke(
            LCToolCall(
                name="search_trials",
                args={"drug": "trastuzumab", "indication": "breast cancer"},
                id="tc4",
                type="tool_call",
            )
        )

    mock_client.search_trials.assert_awaited_once_with(
        "trastuzumab",
        "Breast Neoplasms",
        date_before=None,
    )
    assert isinstance(msg.artifact, SearchTrialsResult)
    assert msg.artifact.total_count == 1
    assert msg.artifact.by_status == {
        "RECRUITING": 0,
        "ACTIVE_NOT_RECRUITING": 0,
        "WITHDRAWN": 0,
    }
    assert len(msg.artifact.trials) == 1
    t = msg.artifact.trials[0]
    assert t.nct_id == "NCT00127933"
    assert t.title == "XeNA Study"
    assert t.phase == "Phase 4"
    assert t.overall_status == "COMPLETED"
    assert t.why_stopped is None
    assert t.indications == ["Breast Cancer"]
    assert t.interventions[0].intervention_type == "Drug"
    assert t.interventions[0].intervention_name == "Trastuzumab"
    assert t.interventions[1].intervention_type == "Drug"
    assert t.interventions[1].intervention_name == "Capecitabine"
    assert t.sponsor == "Hoffmann-La Roche"
    assert t.enrollment == 157
    assert t.start_date == "2005-08"
    assert t.completion_date == "2009-04"
    assert t.primary_outcomes[0].measure == "pCR rate"
    assert t.primary_outcomes[0].time_frame == "4 cycles"
    assert t.references == []
    assert "1 trials" in msg.content


async def test_search_trials_passes_date_before():
    """search_trials forwards date_before from closure into the client call."""
    cutoff = date(2018, 1, 1)
    mock_client = _mock_client(search_trials=SearchTrialsResult())
    tools = build_clinical_trials_tools(date_before=cutoff)

    with (
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.resolve_mesh_id",
            new=AsyncMock(return_value=("D000505", "Alopecia Areata")),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.ClinicalTrialsClient",
            return_value=mock_client,
        ),
    ):
        await _get_tool(tools, "search_trials").ainvoke(
            LCToolCall(
                name="search_trials",
                args={"drug": "tofacitinib", "indication": "alopecia areata"},
                id="tc5",
                type="tool_call",
            )
        )

    mock_client.search_trials.assert_awaited_once_with(
        "tofacitinib",
        "Alopecia Areata",
        date_before=cutoff,
    )


async def test_search_trials_returns_empty_when_resolver_returns_none():
    """search_trials returns a default SearchTrialsResult and skips the client when resolver is None."""
    client_factory = MagicMock()
    tools = build_clinical_trials_tools(date_before=None)

    with (
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.resolve_mesh_id",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.ClinicalTrialsClient",
            new=client_factory,
        ),
    ):
        msg = await _get_tool(tools, "search_trials").ainvoke(
            LCToolCall(
                name="search_trials",
                args={"drug": "metformin", "indication": "xyzzy_bogus"},
                id="tc_ms2",
                type="tool_call",
            )
        )

    assert isinstance(msg.artifact, SearchTrialsResult)
    assert msg.artifact.total_count == 0
    assert msg.artifact.by_status == {}
    assert msg.artifact.trials == []
    assert "MeSH unresolved" in msg.content
    client_factory.assert_not_called()


async def test_search_trials_content_notes_top_50_when_total_exceeds_shown():
    """When total_count exceeds len(trials), the content string flags the 50-cap."""
    trial = Trial(
        nct_id="NCT00000001",
        title="T",
        phase="Phase 2",
        overall_status="RECRUITING",
        sponsor="S",
        mesh_conditions=[MeshTerm(id="D003866", term="Depressive Disorder")],
        interventions=[
            Intervention(intervention_type="Drug", intervention_name="Bupropion"),
        ],
    )
    mock_result = SearchTrialsResult(
        total_count=131,
        by_status={"RECRUITING": 12, "ACTIVE_NOT_RECRUITING": 4, "WITHDRAWN": 1},
        trials=[trial],
    )
    mock_client = _mock_client(search_trials=mock_result)
    tools = build_clinical_trials_tools(date_before=None)

    with (
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.resolve_mesh_id",
            new=AsyncMock(return_value=("D003866", "Depressive Disorder")),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.ClinicalTrialsClient",
            return_value=mock_client,
        ),
    ):
        msg = await _get_tool(tools, "search_trials").ainvoke(
            LCToolCall(
                name="search_trials",
                args={"drug": "bupropion", "indication": "depression"},
                id="tc_search_cap",
                type="tool_call",
            )
        )

    assert "131 trials" in msg.content
    assert "top 50 shown" in msg.content


# ------------------------------------------------------------------
# get_completed
# ------------------------------------------------------------------


async def test_get_completed_returns_completed_trials_result_artifact():
    """get_completed returns a CompletedTrialsResult artifact and reports total count in content."""
    trial = Trial(
        nct_id="NCT04111111",
        title="Phase 3 Study",
        phase="Phase 3",
        overall_status="COMPLETED",
        sponsor="Sponsor",
        enrollment=500,
        mesh_conditions=[MeshTerm(id="D003924", term="Diabetes Mellitus, Type 2")],
        interventions=[
            Intervention(intervention_type="Drug", intervention_name="Semaglutide"),
        ],
    )
    mock_result = CompletedTrialsResult(total_count=12, trials=[trial])
    mock_client = _mock_client(get_completed_trials=mock_result)
    tools = build_clinical_trials_tools(date_before=None)

    with (
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.resolve_mesh_id",
            new=AsyncMock(return_value=("D003924", "Diabetes Mellitus, Type 2")),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.ClinicalTrialsClient",
            return_value=mock_client,
        ),
    ):
        msg = await _get_tool(tools, "get_completed").ainvoke(
            LCToolCall(
                name="get_completed",
                args={"drug": "semaglutide", "indication": "type 2 diabetes"},
                id="tc_completed",
                type="tool_call",
            )
        )

    mock_client.get_completed_trials.assert_awaited_once_with(
        "semaglutide",
        "Diabetes Mellitus, Type 2",
        date_before=None,
    )
    assert isinstance(msg.artifact, CompletedTrialsResult)
    assert msg.artifact.total_count == 12
    assert len(msg.artifact.trials) == 1
    assert msg.artifact.trials[0].nct_id == "NCT04111111"
    assert "12 total" in msg.content
    # only 1 shown of 12 → cap note present
    assert "top 50 shown" in msg.content


async def test_get_completed_passes_date_before():
    """get_completed forwards date_before from closure into the client call."""
    cutoff = date(2020, 1, 1)
    mock_client = _mock_client(get_completed_trials=CompletedTrialsResult())
    tools = build_clinical_trials_tools(date_before=cutoff)

    with (
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.resolve_mesh_id",
            new=AsyncMock(return_value=("D000001", "Test Term")),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.ClinicalTrialsClient",
            return_value=mock_client,
        ),
    ):
        await _get_tool(tools, "get_completed").ainvoke(
            LCToolCall(
                name="get_completed",
                args={"drug": "drug_x", "indication": "indication_y"},
                id="tc_completed_date",
                type="tool_call",
            )
        )

    mock_client.get_completed_trials.assert_awaited_once_with(
        "drug_x",
        "Test Term",
        date_before=cutoff,
    )


async def test_get_completed_returns_empty_when_resolver_returns_none():
    """get_completed returns a default CompletedTrialsResult and skips client when resolver is None."""
    client_factory = MagicMock()
    tools = build_clinical_trials_tools(date_before=None)

    with (
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.resolve_mesh_id",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.ClinicalTrialsClient",
            new=client_factory,
        ),
    ):
        msg = await _get_tool(tools, "get_completed").ainvoke(
            LCToolCall(
                name="get_completed",
                args={"drug": "metformin", "indication": "xyzzy_bogus"},
                id="tc_completed_none",
                type="tool_call",
            )
        )

    assert isinstance(msg.artifact, CompletedTrialsResult)
    assert msg.artifact.total_count == 0
    assert msg.artifact.trials == []
    assert "MeSH unresolved" in msg.content
    client_factory.assert_not_called()


# ------------------------------------------------------------------
# get_terminated
# ------------------------------------------------------------------


async def test_get_terminated_returns_terminated_trials_result_artifact():
    """get_terminated returns TerminatedTrialsResult and counts safety/efficacy stops in content."""
    safety_trial = Trial(
        nct_id="NCT04012255",
        title="Safety Stop Trial",
        phase="Phase 2",
        overall_status="TERMINATED",
        why_stopped="Serious adverse events observed",
        sponsor="Novo Nordisk",
        enrollment=40,
        mesh_conditions=[MeshTerm(id="D050177", term="Overweight")],
        interventions=[
            Intervention(intervention_type="Drug", intervention_name="Semaglutide"),
        ],
    )
    business_trial = Trial(
        nct_id="NCT04012256",
        title="Business Stop Trial",
        phase="Phase 1",
        overall_status="TERMINATED",
        why_stopped="The trial was terminated for strategic reasons.",
        sponsor="Novo Nordisk",
        enrollment=20,
        mesh_conditions=[MeshTerm(id="D050177", term="Overweight")],
        interventions=[
            Intervention(intervention_type="Drug", intervention_name="Semaglutide"),
        ],
    )
    mock_result = TerminatedTrialsResult(
        total_count=2, trials=[safety_trial, business_trial]
    )
    mock_client = _mock_client(get_terminated_trials=mock_result)
    tools = build_clinical_trials_tools(date_before=None)

    with (
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.resolve_mesh_id",
            new=AsyncMock(return_value=("D050177", "Overweight")),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.ClinicalTrialsClient",
            return_value=mock_client,
        ),
    ):
        msg = await _get_tool(tools, "get_terminated").ainvoke(
            LCToolCall(
                name="get_terminated",
                args={"drug": "semaglutide", "indication": "overweight"},
                id="tc_term",
                type="tool_call",
            )
        )

    mock_client.get_terminated_trials.assert_awaited_once_with(
        "semaglutide",
        "Overweight",
        date_before=None,
    )
    assert isinstance(msg.artifact, TerminatedTrialsResult)
    assert msg.artifact.total_count == 2
    assert len(msg.artifact.trials) == 2
    assert msg.artifact.trials[0].nct_id == "NCT04012255"
    assert msg.artifact.trials[0].why_stopped == "Serious adverse events observed"
    assert msg.artifact.trials[1].nct_id == "NCT04012256"
    # 1 safety stop, 1 business → "1 safety/efficacy"
    assert "2 total" in msg.content
    assert "1 safety/efficacy" in msg.content


async def test_get_terminated_content_notes_cap_when_total_exceeds_shown():
    """When total_count exceeds shown trials, the content string flags the 50-cap."""
    trial = Trial(
        nct_id="NCT04012255",
        title="Trial",
        phase="Phase 2",
        overall_status="TERMINATED",
        why_stopped="Lack of efficacy",
        sponsor="S",
        mesh_conditions=[MeshTerm(id="D050177", term="Overweight")],
        interventions=[
            Intervention(intervention_type="Drug", intervention_name="Drug_x"),
        ],
    )
    mock_result = TerminatedTrialsResult(total_count=80, trials=[trial])
    mock_client = _mock_client(get_terminated_trials=mock_result)
    tools = build_clinical_trials_tools(date_before=None)

    with (
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.resolve_mesh_id",
            new=AsyncMock(return_value=("D050177", "Overweight")),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.ClinicalTrialsClient",
            return_value=mock_client,
        ),
    ):
        msg = await _get_tool(tools, "get_terminated").ainvoke(
            LCToolCall(
                name="get_terminated",
                args={"drug": "drug_x", "indication": "indication_y"},
                id="tc_term_cap",
                type="tool_call",
            )
        )

    assert "80 total" in msg.content
    assert "top 50 shown" in msg.content
    assert "stop-category counts cover the 1 shown only" in msg.content


async def test_get_terminated_passes_date_before():
    """get_terminated forwards date_before from closure into the client call."""
    cutoff = date(2018, 1, 1)
    mock_client = _mock_client(get_terminated_trials=TerminatedTrialsResult())
    tools = build_clinical_trials_tools(date_before=cutoff)

    with (
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.resolve_mesh_id",
            new=AsyncMock(return_value=("D000001", "Test Term")),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.ClinicalTrialsClient",
            return_value=mock_client,
        ),
    ):
        await _get_tool(tools, "get_terminated").ainvoke(
            LCToolCall(
                name="get_terminated",
                args={"drug": "drug_x", "indication": "indication_y"},
                id="tc9",
                type="tool_call",
            )
        )

    mock_client.get_terminated_trials.assert_awaited_once_with(
        "drug_x",
        "Test Term",
        date_before=cutoff,
    )


async def test_get_terminated_returns_empty_when_resolver_returns_none():
    """get_terminated returns a default TerminatedTrialsResult and skips client when resolver is None."""
    client_factory = MagicMock()
    tools = build_clinical_trials_tools(date_before=None)

    with (
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.resolve_mesh_id",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.ClinicalTrialsClient",
            new=client_factory,
        ),
    ):
        msg = await _get_tool(tools, "get_terminated").ainvoke(
            LCToolCall(
                name="get_terminated",
                args={"drug": "metformin", "indication": "xyzzy_bogus"},
                id="tc_ms8",
                type="tool_call",
            )
        )

    assert isinstance(msg.artifact, TerminatedTrialsResult)
    assert msg.artifact.total_count == 0
    assert msg.artifact.trials == []
    assert "MeSH unresolved" in msg.content
    client_factory.assert_not_called()


# ------------------------------------------------------------------
# get_landscape
# ------------------------------------------------------------------


async def test_get_landscape_returns_landscape_artifact():
    """get_landscape returns IndicationLandscape artifact with all fields intact."""
    mock_result = IndicationLandscape(
        total_trial_count=95,
        competitors=[
            CompetitorEntry(
                sponsor="Chinese University of Hong Kong",
                drug_name="Esomeprazole",
                drug_type="Drug",
                max_phase="Phase 4",
                trial_count=1,
                statuses={"COMPLETED"},
                total_enrollment=155,
            ),
        ],
        phase_distribution={"Phase 2": 40, "Phase 3": 15, "Phase 4": 10},
        recent_starts=[
            RecentStart(
                nct_id="NCT06836557",
                sponsor="Vanda Pharmaceuticals",
                drug="Tradipitant",
                phase="Phase 3",
            ),
        ],
    )

    mock_client = _mock_client(get_landscape=mock_result)
    tools = build_clinical_trials_tools(date_before=None)

    with (
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.resolve_mesh_id",
            new=AsyncMock(return_value=("D018589", "Gastroparesis")),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.ClinicalTrialsClient",
            return_value=mock_client,
        ),
    ):
        msg = await _get_tool(tools, "get_landscape").ainvoke(
            LCToolCall(
                name="get_landscape",
                args={"indication": "gastroparesis"},
                id="tc6",
                type="tool_call",
            )
        )

    mock_client.get_landscape.assert_awaited_once_with(
        "Gastroparesis",
        date_before=None,
        top_n=10,
    )
    assert isinstance(msg.artifact, IndicationLandscape)
    assert msg.artifact.total_trial_count == 95
    assert msg.artifact.phase_distribution == {
        "Phase 2": 40,
        "Phase 3": 15,
        "Phase 4": 10,
    }
    assert len(msg.artifact.competitors) == 1
    assert msg.artifact.competitors[0].sponsor == "Chinese University of Hong Kong"
    assert msg.artifact.competitors[0].drug_name == "Esomeprazole"
    assert msg.artifact.competitors[0].drug_type == "Drug"
    assert msg.artifact.competitors[0].max_phase == "Phase 4"
    assert msg.artifact.competitors[0].trial_count == 1
    assert msg.artifact.competitors[0].total_enrollment == 155
    assert len(msg.artifact.recent_starts) == 1
    assert msg.artifact.recent_starts[0].nct_id == "NCT06836557"
    assert msg.artifact.recent_starts[0].sponsor == "Vanda Pharmaceuticals"
    assert msg.artifact.recent_starts[0].drug == "Tradipitant"
    assert msg.artifact.recent_starts[0].phase == "Phase 3"
    assert "1 competitors" in msg.content


async def test_get_landscape_does_not_call_client_when_date_before_set():
    """get_landscape early-returns under date_before WITHOUT calling the
    client. The landscape aggregates per-trial overall_status and phase
    across all competitors; those aggregates would leak post-cutoff trial
    state and cannot be reconstructed as-of-cutoff without per-competitor
    scrubbing. Disabled entirely under date_before — the supervisor's core
    reasoning runs off search/completed/terminated.
    """
    cutoff = date(2020, 1, 1)
    mock_result = IndicationLandscape(
        total_trial_count=10, competitors=[], phase_distribution={}, recent_starts=[]
    )
    mock_client = _mock_client(get_landscape=mock_result)
    tools = build_clinical_trials_tools(date_before=cutoff)

    with (
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.resolve_mesh_id",
            new=AsyncMock(return_value=("D018589", "Gastroparesis")),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.ClinicalTrialsClient",
            return_value=mock_client,
        ),
    ):
        msg = await _get_tool(tools, "get_landscape").ainvoke(
            LCToolCall(
                name="get_landscape",
                args={"indication": "gastroparesis"},
                id="tc7",
                type="tool_call",
            )
        )

    mock_client.get_landscape.assert_not_awaited()
    assert isinstance(msg.artifact, IndicationLandscape)
    assert msg.artifact.competitors == []
    assert msg.artifact.total_trial_count is None
    assert msg.artifact.phase_distribution == {}
    assert msg.artifact.recent_starts == []
    assert "skipped under date_before holdout" in msg.content


async def test_get_landscape_returns_empty_when_resolver_returns_none():
    """get_landscape returns default IndicationLandscape and skips client when resolver is None."""
    client_factory = MagicMock()
    tools = build_clinical_trials_tools(date_before=None)

    with (
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.resolve_mesh_id",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.ClinicalTrialsClient",
            new=client_factory,
        ),
    ):
        msg = await _get_tool(tools, "get_landscape").ainvoke(
            LCToolCall(
                name="get_landscape",
                args={"indication": "xyzzy_bogus"},
                id="tc_ms6",
                type="tool_call",
            )
        )

    assert isinstance(msg.artifact, IndicationLandscape)
    assert msg.artifact.total_trial_count is None
    assert msg.artifact.competitors == []
    assert msg.artifact.phase_distribution == {}
    assert msg.artifact.recent_starts == []
    assert "MeSH unresolved" in msg.content
    client_factory.assert_not_called()


# ------------------------------------------------------------------
# check_fda_approval
# ------------------------------------------------------------------


def _fda_client_mock(label_texts: list[str]) -> AsyncMock:
    """AsyncMock FDAClient that works as an async context manager."""
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.get_all_label_indications = AsyncMock(return_value=label_texts)
    return client


async def test_check_fda_approval_approved_match():
    """Drug resolved, labels found, indication approved → is_approved=True with all fields populated."""
    drug_names = ["semaglutide", "ozempic", "wegovy", "rybelsus"]
    label_texts = ["INDICATIONS AND USAGE: treatment of type 2 diabetes mellitus"]
    fda_client = _fda_client_mock(label_texts)
    tools = build_clinical_trials_tools(date_before=None)

    with (
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.resolve_drug_name",
            new=AsyncMock(return_value="CHEMBL2108724"),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.get_all_drug_names",
            new=AsyncMock(return_value=drug_names),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.FDAClient",
            return_value=fda_client,
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.extract_approved_from_labels",
            new=AsyncMock(return_value={"type 2 diabetes mellitus"}),
        ),
    ):
        msg = await _get_tool(tools, "check_fda_approval").ainvoke(
            LCToolCall(
                name="check_fda_approval",
                args={"drug": "semaglutide", "indication": "type 2 diabetes mellitus"},
                id="tc_fda1",
                type="tool_call",
            )
        )

    fda_client.get_all_label_indications.assert_awaited_once_with(drug_names)
    assert isinstance(msg.artifact, ApprovalCheck)
    assert msg.artifact.is_approved is True
    assert msg.artifact.label_found is True
    assert msg.artifact.matched_indication == "type 2 diabetes mellitus"
    assert msg.artifact.drug_names_checked == drug_names
    assert "APPROVED" in msg.content


async def test_check_fda_approval_label_found_but_not_approved():
    """Labels returned but the indication is not in the approved set → is_approved=False, label_found=True."""
    drug_names = ["semaglutide", "ozempic", "wegovy", "rybelsus"]
    label_texts = ["INDICATIONS AND USAGE: treatment of type 2 diabetes mellitus"]
    fda_client = _fda_client_mock(label_texts)
    tools = build_clinical_trials_tools(date_before=None)

    with (
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.resolve_drug_name",
            new=AsyncMock(return_value="CHEMBL2108724"),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.get_all_drug_names",
            new=AsyncMock(return_value=drug_names),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.FDAClient",
            return_value=fda_client,
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.extract_approved_from_labels",
            new=AsyncMock(return_value=set()),
        ),
    ):
        msg = await _get_tool(tools, "check_fda_approval").ainvoke(
            LCToolCall(
                name="check_fda_approval",
                args={"drug": "semaglutide", "indication": "huntington disease"},
                id="tc_fda2",
                type="tool_call",
            )
        )

    assert isinstance(msg.artifact, ApprovalCheck)
    assert msg.artifact.is_approved is False
    assert msg.artifact.label_found is True
    assert msg.artifact.matched_indication is None
    assert msg.artifact.drug_names_checked == drug_names
    assert "not on FDA label" in msg.content


async def test_check_fda_approval_indication_match_is_case_insensitive():
    """Indication match strips and lowercases both sides before comparing."""
    drug_names = ["semaglutide"]
    fda_client = _fda_client_mock(["some label text"])
    tools = build_clinical_trials_tools(date_before=None)

    with (
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.resolve_drug_name",
            new=AsyncMock(return_value="CHEMBL2108724"),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.get_all_drug_names",
            new=AsyncMock(return_value=drug_names),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.FDAClient",
            return_value=fda_client,
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.extract_approved_from_labels",
            new=AsyncMock(return_value={"TYPE 2 Diabetes Mellitus"}),
        ),
    ):
        msg = await _get_tool(tools, "check_fda_approval").ainvoke(
            LCToolCall(
                name="check_fda_approval",
                args={
                    "drug": "semaglutide",
                    "indication": "  type 2 diabetes mellitus  ",
                },
                id="tc_fda3",
                type="tool_call",
            )
        )

    assert msg.artifact.is_approved is True
    assert msg.artifact.label_found is True
    assert msg.artifact.matched_indication == "  type 2 diabetes mellitus  "
    assert msg.artifact.drug_names_checked == drug_names


async def test_check_fda_approval_no_labels_found():
    """Drug names resolved but FDA has no labels → label_found=False, is_approved=False."""
    drug_names = ["aducanumab", "aduhelm"]
    fda_client = _fda_client_mock([])
    tools = build_clinical_trials_tools(date_before=None)
    extract_mock = AsyncMock()

    with (
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.resolve_drug_name",
            new=AsyncMock(return_value="CHEMBL4650346"),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.get_all_drug_names",
            new=AsyncMock(return_value=drug_names),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.FDAClient",
            return_value=fda_client,
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.extract_approved_from_labels",
            new=extract_mock,
        ),
    ):
        msg = await _get_tool(tools, "check_fda_approval").ainvoke(
            LCToolCall(
                name="check_fda_approval",
                args={"drug": "aducanumab", "indication": "alzheimer disease"},
                id="tc_fda4",
                type="tool_call",
            )
        )

    assert isinstance(msg.artifact, ApprovalCheck)
    assert msg.artifact.is_approved is False
    assert msg.artifact.label_found is False
    assert msg.artifact.matched_indication is None
    assert msg.artifact.drug_names_checked == drug_names
    assert "no FDA label" in msg.content
    extract_mock.assert_not_awaited()


async def test_check_fda_approval_unresolved_drug_returns_default():
    """resolve_drug_name raises DataSourceError → default ApprovalCheck, no FDA call, no extract call."""
    fda_factory = MagicMock()
    extract_mock = AsyncMock()
    get_names_mock = AsyncMock()
    tools = build_clinical_trials_tools(date_before=None)

    with (
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.resolve_drug_name",
            new=AsyncMock(
                side_effect=DataSourceError("chembl", "No drug found for 'xyzzybogus'")
            ),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.get_all_drug_names",
            new=get_names_mock,
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.FDAClient",
            new=fda_factory,
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.extract_approved_from_labels",
            new=extract_mock,
        ),
    ):
        msg = await _get_tool(tools, "check_fda_approval").ainvoke(
            LCToolCall(
                name="check_fda_approval",
                args={"drug": "xyzzybogus", "indication": "alzheimer disease"},
                id="tc_fda5",
                type="tool_call",
            )
        )

    assert isinstance(msg.artifact, ApprovalCheck)
    assert msg.artifact.is_approved is False
    assert msg.artifact.label_found is False
    assert msg.artifact.matched_indication is None
    assert msg.artifact.drug_names_checked == []
    assert "drug not resolved" in msg.content
    get_names_mock.assert_not_awaited()
    fda_factory.assert_not_called()
    extract_mock.assert_not_awaited()


async def test_check_fda_approval_no_drug_names_returns_default():
    """get_all_drug_names returns empty → default ApprovalCheck, no FDA call, no extract call."""
    fda_factory = MagicMock()
    extract_mock = AsyncMock()
    tools = build_clinical_trials_tools(date_before=None)

    with (
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.resolve_drug_name",
            new=AsyncMock(return_value="CHEMBL2108724"),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.get_all_drug_names",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.FDAClient",
            new=fda_factory,
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.extract_approved_from_labels",
            new=extract_mock,
        ),
    ):
        msg = await _get_tool(tools, "check_fda_approval").ainvoke(
            LCToolCall(
                name="check_fda_approval",
                args={"drug": "semaglutide", "indication": "type 2 diabetes mellitus"},
                id="tc_fda6",
                type="tool_call",
            )
        )

    assert isinstance(msg.artifact, ApprovalCheck)
    assert msg.artifact.is_approved is False
    assert msg.artifact.label_found is False
    assert msg.artifact.matched_indication is None
    assert msg.artifact.drug_names_checked == []
    assert "no drug names" in msg.content
    fda_factory.assert_not_called()
    extract_mock.assert_not_awaited()


# ------------------------------------------------------------------
# Tool registration
# ------------------------------------------------------------------


async def test_check_fda_approval_is_registered_in_tool_list():
    """build_clinical_trials_tools exposes check_fda_approval."""
    tools = build_clinical_trials_tools(date_before=None)
    names = {t.name for t in tools}
    assert "check_fda_approval" in names


async def test_tool_list_contains_expected_tools():
    """build_clinical_trials_tools returns the documented set of tools and excludes detect_whitespace."""
    tools = build_clinical_trials_tools(date_before=None)
    names = {t.name for t in tools}
    assert names == {
        "search_trials",
        "get_completed",
        "get_terminated",
        "get_landscape",
        "check_fda_approval",
        "finalize_analysis",
    }


# ------------------------------------------------------------------
# finalize_analysis
# ------------------------------------------------------------------


async def test_finalize_analysis_returns_relevance_artifact():
    """finalize_analysis returns a FinalizeClinicalTrialsArtifact carrying the summary
    and the structured relevance split, and confirms completion in content."""
    tools = build_clinical_trials_tools(date_before=None)

    text = "Completed Phase 3 on record for this pair; PAH trials excluded as a different disease."
    msg = await _get_tool(tools, "finalize_analysis").ainvoke(
        LCToolCall(
            name="finalize_analysis",
            args={
                "summary": text,
                "relevant_ncts": ["NCT00000001", "NCT00000002"],
                "contaminated_ncts": ["NCT00000099"],
                "relevance_reasoning": "NCT99 is a PAH trial; the query is systemic hypertension.",
            },
            id="tc_fin",
            type="tool_call",
        )
    )

    art = msg.artifact
    assert art.summary == text
    assert art.relevant_ncts == ["NCT00000001", "NCT00000002"]
    assert art.contaminated_ncts == ["NCT00000099"]
    assert (
        art.relevance_reasoning
        == "NCT99 is a PAH trial; the query is systemic hypertension."
    )
    assert "Analysis complete" in msg.content


async def test_finalize_analysis_rejects_empty_summary():
    """An empty/whitespace summary is rejected with an empty-string artifact so the loop retries."""
    tools = build_clinical_trials_tools(date_before=None)

    msg = await _get_tool(tools, "finalize_analysis").ainvoke(
        LCToolCall(
            name="finalize_analysis",
            args={
                "summary": "   ",
                "relevant_ncts": [],
                "contaminated_ncts": [],
                "relevance_reasoning": "",
            },
            id="tc_fin_empty",
            type="tool_call",
        )
    )

    assert msg.artifact == ""
    assert "REJECTED" in msg.content


# ------------------------------------------------------------------
# Holdout scrubber — _scrub_post_cutoff_outcome and its application
# in the four trial tools.
# ------------------------------------------------------------------

from indication_scout.agents.clinical_trials.clinical_trials_tools import (  # noqa: E402
    _scrub_post_cutoff_outcome,
)


def _trial(
    *,
    nct_id: str,
    overall_status: str = "TERMINATED",
    why_stopped: str | None = "lack of efficacy",
    completion_date: str | None = "2024-06-15",
    start_date: str = "2018-01-01",
    phase: str = "Phase 3",
    drug_name: str = "trastuzumab",
) -> Trial:
    """Minimal Trial for scrubber tests — only the fields the scrubber touches."""
    return Trial(
        nct_id=nct_id,
        title=f"trial {nct_id}",
        phase=phase,
        overall_status=overall_status,
        why_stopped=why_stopped,
        indications=["Breast Cancer"],
        mesh_conditions=[MeshTerm(id="D001943", term="Breast Neoplasms")],
        interventions=[
            Intervention(intervention_type="Drug", intervention_name=drug_name)
        ],
        sponsor="Sponsor",
        enrollment=100,
        start_date=start_date,
        completion_date=completion_date,
        primary_outcomes=[],
        references=[],
    )


# --- _scrub_post_cutoff_outcome (direct unit) ------------------------------


def test_scrub_no_completion_date_terminal_status_scrubbed():
    """A terminal trial (COMPLETED/TERMINATED) with no completion_date
    can't be proven pre-cutoff — scrub it conservatively.
    """
    t = _trial(nct_id="NCT001", overall_status="TERMINATED", completion_date=None)
    out, was_scrubbed = _scrub_post_cutoff_outcome(t, date(2020, 1, 1))
    assert was_scrubbed is True
    assert out.overall_status == "UNKNOWN"
    assert out.why_stopped is None
    assert out.completion_date is None


def test_scrub_no_completion_date_non_terminal_unchanged():
    """A non-terminal trial with no completion_date has no future
    outcome to hide — return unchanged.
    """
    t = _trial(
        nct_id="NCT001",
        overall_status="RECRUITING",
        why_stopped=None,
        completion_date=None,
    )
    out, was_scrubbed = _scrub_post_cutoff_outcome(t, date(2020, 1, 1))
    assert was_scrubbed is False
    assert out is t  # same object, no copy made
    assert out.overall_status == "RECRUITING"
    assert out.completion_date is None


def test_scrub_completion_before_cutoff_returns_unchanged():
    """Completion before cutoff is real history; leave it alone."""
    t = _trial(nct_id="NCT002", completion_date="2019-12-31")
    out, was_scrubbed = _scrub_post_cutoff_outcome(t, date(2020, 1, 1))
    assert was_scrubbed is False
    assert out is t
    assert out.overall_status == "TERMINATED"
    assert out.completion_date == "2019-12-31"


def test_scrub_completion_after_cutoff_strips_outcome_fields():
    """Completion after cutoff → scrub status, why_stopped, completion_date."""
    t = _trial(
        nct_id="NCT003",
        overall_status="TERMINATED",
        why_stopped="lack of efficacy",
        completion_date="2024-06-15",
    )
    out, was_scrubbed = _scrub_post_cutoff_outcome(t, date(2020, 1, 1))
    assert was_scrubbed is True
    assert out is not t  # a copy was made — original is untouched
    assert out.overall_status == "UNKNOWN"
    assert out.why_stopped is None
    assert out.completion_date is None
    # Fields knowable at the cutoff are preserved verbatim
    assert out.nct_id == "NCT003"
    assert out.start_date == "2018-01-01"
    assert out.phase == "Phase 3"
    assert out.enrollment == 100
    # Original trial must be unmodified (no in-place mutation)
    assert t.overall_status == "TERMINATED"
    assert t.why_stopped == "lack of efficacy"
    assert t.completion_date == "2024-06-15"


def test_scrub_completion_equals_cutoff_strips():
    """A completion on the cutoff date is on/after the cutoff, so scrub."""
    t = _trial(nct_id="NCT004", completion_date="2020-01-01")
    out, was_scrubbed = _scrub_post_cutoff_outcome(t, date(2020, 1, 1))
    assert was_scrubbed is True
    assert out.overall_status == "UNKNOWN"


def test_scrub_partial_completion_date_lexicographic():
    """CT.gov partial dates ('YYYY-MM' or 'YYYY') compare lexicographically;
    '2020-06' >= '2020-01-01' so it scrubs.
    """
    t = _trial(nct_id="NCT005", completion_date="2020-06")
    out, was_scrubbed = _scrub_post_cutoff_outcome(t, date(2020, 1, 1))
    assert was_scrubbed is True
    assert out.overall_status == "UNKNOWN"


# --- search_trials: scrubber keeps trials, sets status to UNKNOWN ---------


async def test_search_trials_scrubs_post_cutoff_outcomes_under_date_before():
    """search_trials with date_before set: a trial whose completion_date is
    after the cutoff is KEPT (it was running at the cutoff) but its
    overall_status, why_stopped, and completion_date are stripped.
    """
    pre_trial = _trial(
        nct_id="NCT_PRE",
        overall_status="COMPLETED",
        why_stopped=None,
        completion_date="2019-08-01",
    )
    post_trial = _trial(
        nct_id="NCT_POST",
        overall_status="TERMINATED",
        why_stopped="lack of efficacy",
        completion_date="2024-06-15",
    )
    mock_result = SearchTrialsResult(
        total_count=2,
        by_status={
            "RECRUITING": 0,
            "ACTIVE_NOT_RECRUITING": 0,
            "WITHDRAWN": 0,
            "UNKNOWN": 0,
        },
        trials=[pre_trial, post_trial],
    )

    mock_client = _mock_client(search_trials=mock_result)
    tools = build_clinical_trials_tools(date_before=date(2020, 1, 1))

    with (
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.resolve_mesh_id",
            new=AsyncMock(return_value=("D001943", "Breast Neoplasms")),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.ClinicalTrialsClient",
            return_value=mock_client,
        ),
    ):
        msg = await _get_tool(tools, "search_trials").ainvoke(
            LCToolCall(
                name="search_trials",
                args={"drug": "trastuzumab", "indication": "breast cancer"},
                id="tc_scrub_search",
                type="tool_call",
            )
        )

    result: SearchTrialsResult = msg.artifact
    # Both trials kept (search_trials does not drop scrubbed entries)
    assert result.total_count == 2
    assert len(result.trials) == 2
    by_id = {t.nct_id: t for t in result.trials}

    # Pre-cutoff trial untouched
    assert by_id["NCT_PRE"].overall_status == "COMPLETED"
    assert by_id["NCT_PRE"].why_stopped is None
    assert by_id["NCT_PRE"].completion_date == "2019-08-01"

    # Post-cutoff trial scrubbed
    assert by_id["NCT_POST"].overall_status == "UNKNOWN"
    assert by_id["NCT_POST"].why_stopped is None
    assert by_id["NCT_POST"].completion_date is None

    # Content string surfaces the scrub note
    assert "scrubbed post-cutoff outcomes from 1 trial" in msg.content


# --- get_completed: scrubbed trials are DROPPED ---------------------------


async def test_get_completed_drops_post_cutoff_trials_under_date_before():
    """get_completed with date_before: a trial that completed AFTER the cutoff
    was not completed at the cutoff, so drop it from this scope. total_count
    decrements accordingly.
    """
    pre_trial = _trial(
        nct_id="NCT_PRE",
        overall_status="COMPLETED",
        why_stopped=None,
        completion_date="2019-08-01",
    )
    post_trial = _trial(
        nct_id="NCT_POST",
        overall_status="COMPLETED",
        why_stopped=None,
        completion_date="2024-06-15",
    )
    mock_result = CompletedTrialsResult(total_count=2, trials=[pre_trial, post_trial])

    mock_client = _mock_client(get_completed_trials=mock_result)
    tools = build_clinical_trials_tools(date_before=date(2020, 1, 1))

    with (
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.resolve_mesh_id",
            new=AsyncMock(return_value=("D001943", "Breast Neoplasms")),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.ClinicalTrialsClient",
            return_value=mock_client,
        ),
    ):
        msg = await _get_tool(tools, "get_completed").ainvoke(
            LCToolCall(
                name="get_completed",
                args={"drug": "trastuzumab", "indication": "breast cancer"},
                id="tc_scrub_completed",
                type="tool_call",
            )
        )

    result: CompletedTrialsResult = msg.artifact
    assert result.total_count == 1
    assert len(result.trials) == 1
    assert result.trials[0].nct_id == "NCT_PRE"
    assert "dropped 1 post-cutoff completion" in msg.content


# --- get_terminated: scrubbed trials are DROPPED --------------------------


async def test_get_terminated_drops_post_cutoff_trials_under_date_before():
    """get_terminated with date_before: a trial that terminated AFTER the
    cutoff is dropped. The safety/efficacy count over the remaining shown
    set is recomputed naturally.
    """
    pre_trial = _trial(
        nct_id="NCT_PRE",
        overall_status="TERMINATED",
        why_stopped="safety concerns",
        completion_date="2019-05-01",
    )
    post_trial = _trial(
        nct_id="NCT_POST",
        overall_status="TERMINATED",
        why_stopped="lack of efficacy",
        completion_date="2024-06-15",
    )
    mock_result = TerminatedTrialsResult(total_count=2, trials=[pre_trial, post_trial])

    mock_client = _mock_client(get_terminated_trials=mock_result)
    tools = build_clinical_trials_tools(date_before=date(2020, 1, 1))

    with (
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.resolve_mesh_id",
            new=AsyncMock(return_value=("D001943", "Breast Neoplasms")),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.ClinicalTrialsClient",
            return_value=mock_client,
        ),
    ):
        msg = await _get_tool(tools, "get_terminated").ainvoke(
            LCToolCall(
                name="get_terminated",
                args={"drug": "trastuzumab", "indication": "breast cancer"},
                id="tc_scrub_terminated",
                type="tool_call",
            )
        )

    result: TerminatedTrialsResult = msg.artifact
    assert result.total_count == 1
    assert len(result.trials) == 1
    assert result.trials[0].nct_id == "NCT_PRE"
    # The post-cutoff "lack of efficacy" termination must NOT be counted
    assert "1 safety/efficacy in shown set" in msg.content
    assert "dropped 1 post-cutoff termination" in msg.content


# --- get_landscape (no-cutoff path) ----------------------------------------
# The date_before-set path is covered by
# test_get_landscape_does_not_call_client_when_date_before_set above.


async def test_get_landscape_runs_normally_without_date_before():
    """No date_before → landscape executes against the CT.gov client as usual."""
    landscape = IndicationLandscape(
        total_trial_count=42,
        competitors=[
            CompetitorEntry(
                sponsor="Sponsor",
                drug_name="drug",
                drug_type="Drug",
                max_phase="Phase 3",
                trial_count=3,
                statuses=set(),
                total_enrollment=300,
            )
        ],
    )
    mock_client = _mock_client(get_landscape=landscape)
    tools = build_clinical_trials_tools(date_before=None)

    with (
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.resolve_mesh_id",
            new=AsyncMock(return_value=("D001943", "Breast Neoplasms")),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.ClinicalTrialsClient",
            return_value=mock_client,
        ),
    ):
        msg = await _get_tool(tools, "get_landscape").ainvoke(
            LCToolCall(
                name="get_landscape",
                args={"indication": "breast cancer"},
                id="tc_landscape_normal",
                type="tool_call",
            )
        )

    assert msg.artifact is landscape
    mock_client.get_landscape.assert_awaited_once()
