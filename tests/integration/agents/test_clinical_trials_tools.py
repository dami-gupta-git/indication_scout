"""Integration tests for clinical_trials_tools.

These tests hit real ClinicalTrials.gov APIs.
"""

import logging

from indication_scout.agents.clinical_trials_tools import build_clinical_trials_tools

logger = logging.getLogger(__name__)

# Build tools once at module level (no date_before for integration tests)
_tools = build_clinical_trials_tools(date_before=None)
_tool_map = {t.name: t for t in _tools}

detect_whitespace = _tool_map["detect_whitespace"]
search_trials = _tool_map["search_trials"]
get_landscape = _tool_map["get_landscape"]
get_terminated = _tool_map["get_terminated"]


# ------------------------------------------------------------------
# detect_whitespace
# ------------------------------------------------------------------


async def test_detect_whitespace_whitespace():
    """detect_whitespace tool identifies unexplored drug-indication pair.

    Tirzepatide + Huntington disease = whitespace (no exact matches).
    Validates the tool returns a dict with correct structure and values,
    including indication_drugs ranked by phase.
    """
    result = await detect_whitespace.ainvoke(
        {"drug": "tirzepatide", "indication": "Huntington disease"}
    )

    assert result["is_whitespace"] is True
    assert result["exact_match_count"] == 0
    assert 150 < result["drug_only_trials"] < 300
    assert 200 < result["indication_only_trials"] < 400

    assert 40 < len(result["indication_drugs"]) < 60

    # Verify deduplication: all drug_names should be unique
    drug_names = [cd["drug_name"] for cd in result["indication_drugs"]]
    assert len(drug_names) == len(set(drug_names))

    # Known Phase 4 HD drugs must be present
    assert "Memantine" in drug_names
    assert "Tetrabenazine" in drug_names

    # Ranking: first entries should be Phase 4
    assert result["indication_drugs"][0]["phase"] == "Phase 4"
    assert result["indication_drugs"][1]["phase"] == "Phase 4"
    assert result["indication_drugs"][2]["phase"] == "Phase 4"

    # Verify all IndicationDrug fields on a known entry
    [memantine] = [
        cd for cd in result["indication_drugs"] if cd["drug_name"] == "Memantine"
    ]
    assert memantine["nct_id"] == "NCT00652457"
    assert memantine["drug_name"] == "Memantine"
    assert memantine["indication"] == "Huntington's Disease"
    assert memantine["phase"] == "Phase 4"
    assert memantine["status"] == "COMPLETED"

    # No Phase 1 trials (Phase 2+ filter)
    for cd in result["indication_drugs"]:
        assert "Phase 1" not in cd["phase"]
        assert "Early Phase 1" not in cd["phase"]


async def test_detect_whitespace_not_whitespace():
    """detect_whitespace tool returns is_whitespace=False when trials exist.

    Semaglutide + diabetes = NOT whitespace (many exact matches).
    indication_drugs should be empty.
    """
    result = await detect_whitespace.ainvoke(
        {"drug": "semaglutide", "indication": "diabetes"}
    )

    assert result["is_whitespace"] is False
    assert result["exact_match_count"] >= 10
    assert 400 < result["drug_only_trials"] < 800
    assert 10000 < result["indication_only_trials"] < 100000
    assert result["indication_drugs"] == []


# ------------------------------------------------------------------
# search_trials
# ------------------------------------------------------------------


async def test_search_trials_drug_and_indication():
    """search_trials tool returns trial dicts for a drug-indication pair."""
    result = await search_trials.ainvoke(
        {"drug": "trastuzumab", "indication": "breast cancer"}
    )

    # Client default max_results=200, no tool cap
    assert len(result) >= 20

    # Find NCT01702571 - Roche T-DM1 post-marketing safety study
    [tdm1] = [t for t in result if t["nct_id"] == "NCT01702571"]

    assert tdm1["nct_id"] == "NCT01702571"
    assert (
        tdm1["title"]
        == "A Study of Trastuzumab Emtansine in Participants With Human Epidermal Growth Factor Receptor 2 (HER2)-Positive Breast Cancer Who Have Received Prior Anti-HER2 And Chemotherapy-based Treatment"
    )
    assert tdm1["phase"] == "Phase 3"
    assert tdm1["overall_status"] == "COMPLETED"
    assert tdm1["why_stopped"] is None
    assert tdm1["indications"] == ["Breast Cancer"]
    assert tdm1["sponsor"] == "Hoffmann-La Roche"
    assert tdm1["enrollment"] == 2185
    assert tdm1["start_date"] == "2012-11-27"
    assert tdm1["completion_date"] == "2020-07-31"
    assert tdm1["references"] == ["36084395", "34741021", "32634611"]

    assert len(tdm1["interventions"]) == 1
    assert tdm1["interventions"][0]["intervention_type"] == "Drug"
    assert tdm1["interventions"][0]["intervention_name"] == "Trastuzumab Emtansine"

    assert len(tdm1["primary_outcomes"]) == 1
    assert (
        tdm1["primary_outcomes"][0]["measure"]
        == "Percentage of Participants With Adverse Events of Primary Interest (AEPIs)"
    )
    assert (
        tdm1["primary_outcomes"][0]["time_frame"]
        == "Baseline up to approximately 7 years"
    )


async def test_search_trials_nonexistent_drug():
    """search_trials tool returns empty list for nonexistent drug."""
    result = await search_trials.ainvoke(
        {"drug": "xyzzy_not_a_real_drug_12345", "indication": "diabetes"}
    )

    assert result == []


# ------------------------------------------------------------------
# get_landscape
# ------------------------------------------------------------------


async def test_get_landscape_gastroparesis():
    """get_landscape tool returns competitive landscape dict."""
    result = await get_landscape.ainvoke({"indication": "gastroparesis"})

    # Gastroparesis has ~300+ trials
    assert result["total_trial_count"] > 200

    # Tool passes top_n=10
    assert len(result["competitors"]) == 10

    # Phase distribution
    assert 10 < result["phase_distribution"]["Phase 2"] < 100
    assert 5 < result["phase_distribution"]["Phase 3"] < 50
    assert 1 < result["phase_distribution"]["Phase 4"] < 30

    # Known recent start
    assert len(result["recent_starts"]) >= 1
    [tradipitant] = [
        rs for rs in result["recent_starts"] if rs["nct_id"] == "NCT06836557"
    ]
    assert tradipitant["sponsor"] == "Vanda Pharmaceuticals"
    assert tradipitant["drug"] == "Tradipitant"
    assert tradipitant["phase"] == "Phase 3"

    # Known competitor - verify all fields
    [cuhk] = [
        c
        for c in result["competitors"]
        if c["sponsor"] == "Chinese University of Hong Kong"
        and c["drug_name"] == "Esomeprazole"
    ]
    assert cuhk["sponsor"] == "Chinese University of Hong Kong"
    assert cuhk["drug_name"] == "Esomeprazole"
    assert cuhk["drug_type"] == "Drug"
    assert cuhk["max_phase"] == "Phase 4"
    assert cuhk["trial_count"] == 1
    assert cuhk["total_enrollment"] == 155


async def test_get_landscape_nonexistent_indication():
    """get_landscape tool returns empty landscape for nonexistent indication."""
    result = await get_landscape.ainvoke({"indication": "xyzzy_fake_indication_99999"})

    assert result["total_trial_count"] == 0
    assert result["competitors"] == []
    assert result["phase_distribution"] == {}


# ------------------------------------------------------------------
# get_terminated
# ------------------------------------------------------------------


async def test_get_terminated_semaglutide():
    """get_terminated tool returns terminated trial dicts."""
    result = await get_terminated.ainvoke({"drug": "semaglutide", "indication": "overweight"})

    assert len(result) >= 1

    # NCT02499705 — safety termination from overweight indication query
    [safety_trial] = [t for t in result if t["nct_id"] == "NCT02499705"]
    assert safety_trial["stop_category"] == "safety"


async def test_get_terminated_nonexistent_query():
    """get_terminated tool returns empty list for nonexistent drug and indication."""
    result = await get_terminated.ainvoke(
        {"drug": "xyzzy_not_a_real_term_12345", "indication": "xyzzy_not_a_real_indication_12345"}
    )

    assert result == []
