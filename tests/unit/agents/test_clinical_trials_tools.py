"""Unit tests for clinical_trials_tools.build_clinical_trials_tools."""

import logging
from datetime import date
from unittest.mock import AsyncMock, patch

from langchain_core.messages import ToolCall as LCToolCall

from indication_scout.agents.clinical_trials.clinical_trials_tools import (
    build_clinical_trials_tools,
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
# detect_whitespace
# ------------------------------------------------------------------


async def test_detect_whitespace_whitespace_case():
    """detect_whitespace returns WhitespaceResult artifact when whitespace exists."""
    mock_result = WhitespaceResult(
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
            ),
            IndicationDrug(
                nct_id="NCT00029874",
                drug_name="Tetrabenazine",
                indication="Huntington's Disease",
                phase="Phase 4",
                status="COMPLETED",
            ),
        ],
    )

    mock_client = _mock_client(detect_whitespace=mock_result)
    tools = build_clinical_trials_tools(date_before=None)

    with patch(
        "indication_scout.agents.clinical_trials.clinical_trials_tools.ClinicalTrialsClient",
        return_value=mock_client,
    ):
        msg = await _get_tool(tools, "detect_whitespace").ainvoke(
            LCToolCall(
                name="detect_whitespace",
                args={"drug": "tirzepatide", "indication": "Huntington disease"},
                id="tc1",
                type="tool_call",
            )
        )

    mock_client.detect_whitespace.assert_awaited_once_with(
        "tirzepatide", "Huntington disease", date_before=None
    )
    assert isinstance(msg.artifact, WhitespaceResult)
    assert msg.artifact.is_whitespace is True
    assert msg.artifact.no_data is False
    assert msg.artifact.exact_match_count == 0
    assert msg.artifact.drug_only_trials == 120
    assert msg.artifact.indication_only_trials == 250
    assert len(msg.artifact.indication_drugs) == 2
    assert msg.artifact.indication_drugs[0].nct_id == "NCT00652457"
    assert msg.artifact.indication_drugs[0].drug_name == "Memantine"
    assert msg.artifact.indication_drugs[0].phase == "Phase 4"
    assert msg.artifact.indication_drugs[0].status == "COMPLETED"
    assert msg.artifact.indication_drugs[1].nct_id == "NCT00029874"
    assert msg.artifact.indication_drugs[1].drug_name == "Tetrabenazine"
    assert "True" in msg.content


async def test_detect_whitespace_not_whitespace_case():
    """detect_whitespace returns artifact with is_whitespace=False when trials exist."""
    mock_result = WhitespaceResult(
        is_whitespace=False,
        no_data=False,
        exact_match_count=25,
        drug_only_trials=500,
        indication_only_trials=30000,
        indication_drugs=[],
    )

    mock_client = _mock_client(detect_whitespace=mock_result)
    tools = build_clinical_trials_tools(date_before=None)

    with patch(
        "indication_scout.agents.clinical_trials.clinical_trials_tools.ClinicalTrialsClient",
        return_value=mock_client,
    ):
        msg = await _get_tool(tools, "detect_whitespace").ainvoke(
            LCToolCall(
                name="detect_whitespace",
                args={"drug": "semaglutide", "indication": "diabetes"},
                id="tc2",
                type="tool_call",
            )
        )

    assert msg.artifact.is_whitespace is False
    assert msg.artifact.exact_match_count == 25
    assert msg.artifact.drug_only_trials == 500
    assert msg.artifact.indication_only_trials == 30000
    assert msg.artifact.indication_drugs == []
    assert "False" in msg.content


async def test_detect_whitespace_passes_date_before():
    """detect_whitespace forwards date_before from closure to the client."""
    cutoff = date(2020, 1, 1)
    mock_result = WhitespaceResult(
        is_whitespace=True,
        no_data=False,
        exact_match_count=0,
        drug_only_trials=10,
        indication_only_trials=20,
        indication_drugs=[],
    )
    mock_client = _mock_client(detect_whitespace=mock_result)
    tools = build_clinical_trials_tools(date_before=cutoff)

    with patch(
        "indication_scout.agents.clinical_trials.clinical_trials_tools.ClinicalTrialsClient",
        return_value=mock_client,
    ):
        await _get_tool(tools, "detect_whitespace").ainvoke(
            LCToolCall(
                name="detect_whitespace",
                args={"drug": "drug_x", "indication": "indication_y"},
                id="tc3",
                type="tool_call",
            )
        )

    mock_client.detect_whitespace.assert_awaited_once_with(
        "drug_x", "indication_y", date_before=cutoff
    )


# ------------------------------------------------------------------
# search_trials
# ------------------------------------------------------------------


async def test_search_trials_returns_trial_artifact():
    """search_trials returns list[Trial] as artifact with all fields intact."""
    trial = Trial(
        nct_id="NCT00127933",
        title="XeNA Study",
        phase="Phase 4",
        overall_status="COMPLETED",
        why_stopped=None,
        indications=["Breast Cancer"],
        interventions=[
            Intervention(intervention_type="Drug", intervention_name="Capecitabine"),
        ],
        sponsor="Hoffmann-La Roche",
        enrollment=157,
        start_date="2005-08",
        completion_date="2009-04",
        primary_outcomes=[PrimaryOutcome(measure="pCR rate", time_frame="4 cycles")],
        references=[],
    )

    mock_client = _mock_client(search_trials=[trial])
    tools = build_clinical_trials_tools(date_before=None)

    with patch(
        "indication_scout.agents.clinical_trials.clinical_trials_tools.ClinicalTrialsClient",
        return_value=mock_client,
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
        "breast cancer",
        date_before=None,
        max_results=50,
        sort="EnrollmentCount:desc",
    )
    assert len(msg.artifact) == 1
    t = msg.artifact[0]
    assert t.nct_id == "NCT00127933"
    assert t.title == "XeNA Study"
    assert t.phase == "Phase 4"
    assert t.overall_status == "COMPLETED"
    assert t.why_stopped is None
    assert t.indications == ["Breast Cancer"]
    assert t.interventions[0].intervention_type == "Drug"
    assert t.interventions[0].intervention_name == "Capecitabine"
    assert t.sponsor == "Hoffmann-La Roche"
    assert t.enrollment == 157
    assert t.start_date == "2005-08"
    assert t.completion_date == "2009-04"
    assert t.primary_outcomes[0].measure == "pCR rate"
    assert t.primary_outcomes[0].time_frame == "4 cycles"
    assert t.references == []
    assert "1 trials" in msg.content


async def test_search_trials_passes_date_before_and_max_results():
    """search_trials forwards date_before and max_search_results from closure."""
    cutoff = date(2018, 1, 1)
    mock_client = _mock_client(search_trials=[])
    tools = build_clinical_trials_tools(date_before=cutoff, max_search_results=30)

    with patch(
        "indication_scout.agents.clinical_trials.clinical_trials_tools.ClinicalTrialsClient",
        return_value=mock_client,
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
        "alopecia areata",
        date_before=cutoff,
        max_results=30,
        sort="EnrollmentCount:desc",
    )


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

    with patch(
        "indication_scout.agents.clinical_trials.clinical_trials_tools.ClinicalTrialsClient",
        return_value=mock_client,
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
        "gastroparesis", date_before=None, top_n=10
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


async def test_get_landscape_passes_date_before():
    """get_landscape forwards date_before from closure to the client."""
    cutoff = date(2020, 1, 1)
    mock_result = IndicationLandscape(
        total_trial_count=10, competitors=[], phase_distribution={}, recent_starts=[]
    )
    mock_client = _mock_client(get_landscape=mock_result)
    tools = build_clinical_trials_tools(date_before=cutoff)

    with patch(
        "indication_scout.agents.clinical_trials.clinical_trials_tools.ClinicalTrialsClient",
        return_value=mock_client,
    ):
        await _get_tool(tools, "get_landscape").ainvoke(
            LCToolCall(
                name="get_landscape",
                args={"indication": "gastroparesis"},
                id="tc7",
                type="tool_call",
            )
        )

    mock_client.get_landscape.assert_awaited_once_with(
        "gastroparesis", date_before=cutoff, top_n=10
    )


# ------------------------------------------------------------------
# get_terminated
# ------------------------------------------------------------------


async def test_get_terminated_returns_terminated_trial_artifact():
    """get_terminated returns list[TerminatedTrial] artifact with all fields intact."""
    terminated = TerminatedTrial(
        nct_id="NCT04012255",
        title="Semaglutide Overweight Trial",
        drug_name="Semaglutide",
        indication="Overweight",
        phase="Phase 1",
        why_stopped="The trial was terminated for strategic reasons.",
        stop_category="business",
        enrollment=40,
        sponsor="Novo Nordisk",
        start_date="2019-01-01",
        termination_date="2020-06-01",
    )

    mock_client = _mock_client(get_terminated=[terminated])
    tools = build_clinical_trials_tools(date_before=None)

    with patch(
        "indication_scout.agents.clinical_trials.clinical_trials_tools.ClinicalTrialsClient",
        return_value=mock_client,
    ):
        msg = await _get_tool(tools, "get_terminated").ainvoke(
            LCToolCall(
                name="get_terminated",
                args={"drug": "semaglutide", "indication": "overweight"},
                id="tc8",
                type="tool_call",
            )
        )

    mock_client.get_terminated.assert_awaited_once_with(
        "semaglutide", "overweight", date_before=None, sort="EnrollmentCount:desc"
    )
    assert len(msg.artifact) == 1
    t = msg.artifact[0]
    assert t.nct_id == "NCT04012255"
    assert t.title == "Semaglutide Overweight Trial"
    assert t.drug_name == "Semaglutide"
    assert t.indication == "Overweight"
    assert t.phase == "Phase 1"
    assert t.why_stopped == "The trial was terminated for strategic reasons."
    assert t.stop_category == "business"
    assert t.enrollment == 40
    assert t.sponsor == "Novo Nordisk"
    assert t.start_date == "2019-01-01"
    assert t.termination_date == "2020-06-01"
    assert "1 indication-specific terminations" in msg.content


async def test_get_terminated_passes_date_before():
    """get_terminated forwards date_before from closure to the client."""
    cutoff = date(2018, 1, 1)
    mock_client = _mock_client(get_terminated=[])
    tools = build_clinical_trials_tools(date_before=cutoff)

    with patch(
        "indication_scout.agents.clinical_trials.clinical_trials_tools.ClinicalTrialsClient",
        return_value=mock_client,
    ):
        await _get_tool(tools, "get_terminated").ainvoke(
            LCToolCall(
                name="get_terminated",
                args={"drug": "drug_x", "indication": "indication_y"},
                id="tc9",
                type="tool_call",
            )
        )

    mock_client.get_terminated.assert_awaited_once_with(
        "drug_x", "indication_y", date_before=cutoff, sort="EnrollmentCount:desc"
    )


# ------------------------------------------------------------------
# finalize_analysis
# ------------------------------------------------------------------


async def test_finalize_analysis_returns_summary_as_artifact():
    """finalize_analysis returns the summary string as artifact and confirms completion in content."""
    tools = build_clinical_trials_tools(date_before=None)

    text = "No trials found for tirzepatide in Huntington disease. The landscape shows 5 competitors."
    msg = await _get_tool(tools, "finalize_analysis").ainvoke(
        LCToolCall(
            name="finalize_analysis",
            args={"summary": text},
            id="tc_fin",
            type="tool_call",
        )
    )

    assert msg.artifact == text
    assert "Analysis complete" in msg.content
