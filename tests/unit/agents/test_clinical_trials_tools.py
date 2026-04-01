"""Unit tests for clinical_trials_tools."""

import logging
from datetime import date
from unittest.mock import AsyncMock, patch

from indication_scout.agents.clinical_trials_tools import build_clinical_trials_tools
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
# Helper: build a mock ClinicalTrialsClient
# ------------------------------------------------------------------


def _mock_client(**method_returns: AsyncMock) -> AsyncMock:
    """Build an AsyncMock that works as an async context manager."""
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    for name, return_value in method_returns.items():
        setattr(client, name, AsyncMock(return_value=return_value))
    return client


def _get_tool(tools: list, name: str):
    """Find a tool by name from the tools list."""
    for t in tools:
        if t.name == name:
            return t
    raise ValueError(f"Tool '{name}' not found in {[t.name for t in tools]}")


# ------------------------------------------------------------------
# detect_whitespace
# ------------------------------------------------------------------


async def test_detect_whitespace_whitespace_case():
    """detect_whitespace tool returns model_dump of WhitespaceResult when whitespace exists."""
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
    detect_whitespace = _get_tool(tools, "detect_whitespace")

    with patch(
        "indication_scout.agents.clinical_trials_tools.ClinicalTrialsClient",
        return_value=mock_client,
    ):
        result = await detect_whitespace.ainvoke(
            {"drug": "tirzepatide", "indication": "Huntington disease"}
        )

    mock_client.detect_whitespace.assert_awaited_once_with(
        "tirzepatide", "Huntington disease", date_before=None
    )

    assert result["is_whitespace"] is True
    assert result["no_data"] is False
    assert result["exact_match_count"] == 0
    assert result["drug_only_trials"] == 120
    assert result["indication_only_trials"] == 250
    assert len(result["indication_drugs"]) == 2

    memantine = result["indication_drugs"][0]
    assert memantine["nct_id"] == "NCT00652457"
    assert memantine["drug_name"] == "Memantine"
    assert memantine["indication"] == "Huntington's Disease"
    assert memantine["phase"] == "Phase 4"
    assert memantine["status"] == "COMPLETED"

    tetrabenazine = result["indication_drugs"][1]
    assert tetrabenazine["nct_id"] == "NCT00029874"
    assert tetrabenazine["drug_name"] == "Tetrabenazine"
    assert tetrabenazine["indication"] == "Huntington's Disease"
    assert tetrabenazine["phase"] == "Phase 4"
    assert tetrabenazine["status"] == "COMPLETED"


async def test_detect_whitespace_not_whitespace_case():
    """detect_whitespace tool returns model_dump when exact matches exist."""
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
    detect_whitespace = _get_tool(tools, "detect_whitespace")

    with patch(
        "indication_scout.agents.clinical_trials_tools.ClinicalTrialsClient",
        return_value=mock_client,
    ):
        result = await detect_whitespace.ainvoke(
            {"drug": "semaglutide", "indication": "diabetes"}
        )

    mock_client.detect_whitespace.assert_awaited_once_with(
        "semaglutide", "diabetes", date_before=None
    )

    assert result["is_whitespace"] is False
    assert result["no_data"] is False
    assert result["exact_match_count"] == 25
    assert result["drug_only_trials"] == 500
    assert result["indication_only_trials"] == 30000
    assert result["indication_drugs"] == []


async def test_detect_whitespace_passes_date_before():
    """detect_whitespace passes date_before from closure to client."""
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
    detect_whitespace = _get_tool(tools, "detect_whitespace")

    with patch(
        "indication_scout.agents.clinical_trials_tools.ClinicalTrialsClient",
        return_value=mock_client,
    ):
        await detect_whitespace.ainvoke(
            {"drug": "drug_x", "indication": "indication_y"}
        )

    mock_client.detect_whitespace.assert_awaited_once_with(
        "drug_x", "indication_y", date_before=cutoff
    )


# ------------------------------------------------------------------
# search_trials
# ------------------------------------------------------------------


async def test_search_trials_returns_trial_dicts():
    """search_trials tool returns list of model_dump dicts."""
    trial = Trial(
        nct_id="NCT00127933",
        title="XeNA Study - A Study of Xeloda (Capecitabine) in Patients With Invasive Breast Cancer",
        phase="Phase 4",
        overall_status="COMPLETED",
        why_stopped=None,
        indications=["Breast Cancer"],
        interventions=[
            Intervention(
                intervention_type="Drug",
                intervention_name="Herceptin (HER2-neu positive patients only)",
            ),
        ],
        sponsor="Hoffmann-La Roche",
        enrollment=157,
        start_date="2005-08",
        completion_date="2009-04",
        primary_outcomes=[
            PrimaryOutcome(
                measure="pCR plus npCR rate",
                time_frame="after four 3-week cycles",
            ),
        ],
        references=[],
    )

    mock_client = _mock_client(search_trials=[trial])
    tools = build_clinical_trials_tools(date_before=None)
    search_trials = _get_tool(tools, "search_trials")

    with patch(
        "indication_scout.agents.clinical_trials_tools.ClinicalTrialsClient",
        return_value=mock_client,
    ):
        result = await search_trials.ainvoke(
            {"drug": "trastuzumab", "indication": "breast cancer"}
        )

    mock_client.search_trials.assert_awaited_once_with(
        "trastuzumab", "breast cancer", date_before=None, max_results=50
    )

    assert len(result) == 1
    t = result[0]
    assert t["nct_id"] == "NCT00127933"
    assert (
        t["title"]
        == "XeNA Study - A Study of Xeloda (Capecitabine) in Patients With Invasive Breast Cancer"
    )
    assert t["phase"] == "Phase 4"
    assert t["overall_status"] == "COMPLETED"
    assert t["why_stopped"] is None
    assert t["indications"] == ["Breast Cancer"]
    assert len(t["interventions"]) == 1
    assert t["interventions"][0]["intervention_type"] == "Drug"
    assert (
        t["interventions"][0]["intervention_name"]
        == "Herceptin (HER2-neu positive patients only)"
    )
    assert t["sponsor"] == "Hoffmann-La Roche"
    assert t["enrollment"] == 157
    assert t["start_date"] == "2005-08"
    assert t["completion_date"] == "2009-04"
    assert len(t["primary_outcomes"]) == 1
    assert t["primary_outcomes"][0]["measure"] == "pCR plus npCR rate"
    assert t["primary_outcomes"][0]["time_frame"] == "after four 3-week cycles"
    assert t["references"] == []


async def test_search_trials_returns_all():
    """search_trials tool returns all trials from client without capping."""
    trials = [
        Trial(nct_id=f"NCT{i:08d}", title=f"Trial {i}", phase="Phase 2")
        for i in range(25)
    ]

    mock_client = _mock_client(search_trials=trials)
    tools = build_clinical_trials_tools(date_before=None)
    search_trials = _get_tool(tools, "search_trials")

    with patch(
        "indication_scout.agents.clinical_trials_tools.ClinicalTrialsClient",
        return_value=mock_client,
    ):
        result = await search_trials.ainvoke(
            {"drug": "drug_x", "indication": "indication_y"}
        )

    assert len(result) == 25
    assert result[0]["nct_id"] == "NCT00000000"
    assert result[24]["nct_id"] == "NCT00000024"


# ------------------------------------------------------------------
# get_landscape
# ------------------------------------------------------------------


async def test_get_landscape_returns_landscape_dict():
    """get_landscape tool returns model_dump of IndicationLandscape."""
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
    get_landscape = _get_tool(tools, "get_landscape")

    with patch(
        "indication_scout.agents.clinical_trials_tools.ClinicalTrialsClient",
        return_value=mock_client,
    ):
        result = await get_landscape.ainvoke({"indication": "gastroparesis"})

    mock_client.get_landscape.assert_awaited_once_with(
        "gastroparesis", date_before=None, top_n=10
    )

    assert result["total_trial_count"] == 95
    assert result["phase_distribution"] == {"Phase 2": 40, "Phase 3": 15, "Phase 4": 10}

    assert len(result["competitors"]) == 1
    comp = result["competitors"][0]
    assert comp["sponsor"] == "Chinese University of Hong Kong"
    assert comp["drug_name"] == "Esomeprazole"
    assert comp["drug_type"] == "Drug"
    assert comp["max_phase"] == "Phase 4"
    assert comp["trial_count"] == 1
    assert comp["total_enrollment"] == 155

    assert len(result["recent_starts"]) == 1
    rs = result["recent_starts"][0]
    assert rs["nct_id"] == "NCT06836557"
    assert rs["sponsor"] == "Vanda Pharmaceuticals"
    assert rs["drug"] == "Tradipitant"
    assert rs["phase"] == "Phase 3"


# ------------------------------------------------------------------
# get_terminated
# ------------------------------------------------------------------


async def test_get_terminated_returns_terminated_dicts():
    """get_terminated tool returns list of model_dump dicts."""
    terminated = TerminatedTrial(
        nct_id="NCT04012255",
        drug_name="Semaglutide (administered by DV3396 pen)",
        indication="Overweight",
        phase="Phase 1",
        why_stopped="The trial was terminated for strategic reasons.",
        stop_category="business",
    )

    mock_client = _mock_client(get_terminated=[terminated])
    tools = build_clinical_trials_tools(date_before=None)
    get_terminated = _get_tool(tools, "get_terminated")

    with patch(
        "indication_scout.agents.clinical_trials_tools.ClinicalTrialsClient",
        return_value=mock_client,
    ):
        result = await get_terminated.ainvoke({"drug": "semaglutide", "indication": "overweight"})

    mock_client.get_terminated.assert_awaited_once_with("semaglutide", "overweight", date_before=None)

    assert len(result) == 1
    t = result[0]
    assert t["nct_id"] == "NCT04012255"
    assert t["drug_name"] == "Semaglutide (administered by DV3396 pen)"
    assert t["indication"] == "Overweight"
    assert t["phase"] == "Phase 1"
    assert t["why_stopped"] == "The trial was terminated for strategic reasons."
    assert t["stop_category"] == "business"


async def test_get_terminated_returns_all():
    """get_terminated tool returns all trials from client without capping."""
    trials = [TerminatedTrial(nct_id=f"NCT{i:08d}") for i in range(25)]

    mock_client = _mock_client(get_terminated=trials)
    tools = build_clinical_trials_tools(date_before=None)
    get_terminated = _get_tool(tools, "get_terminated")

    with patch(
        "indication_scout.agents.clinical_trials_tools.ClinicalTrialsClient",
        return_value=mock_client,
    ):
        result = await get_terminated.ainvoke({"drug": "some_drug", "indication": "some_indication"})

    assert len(result) == 25
    assert result[0]["nct_id"] == "NCT00000000"
    assert result[24]["nct_id"] == "NCT00000024"
