"""Integration tests for ClinicalTrialsAgent.

These tests hit real ClinicalTrials.gov and Anthropic APIs.
They verify the agent calls the right tools and produces structured output.
"""

import logging

from indication_scout.agents.clinical_trials import ClinicalTrialsAgent

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Whitespace path: drug-disease pair with no trials
# ------------------------------------------------------------------


async def test_agent_whitespace_path():
    """Agent detects whitespace for tirzepatide + Huntington disease.

    Expected behavior: detect_whitespace → whitespace found →
    get_terminated → get_landscape → summary.
    """
    agent = ClinicalTrialsAgent()
    result = await agent.run(
        {
            "drug_name": "tirzepatide",
            "disease_name": "Huntington disease",
        }
    )

    output = result["clinical_trials_output"]

    # Whitespace detected
    assert output.whitespace is not None
    assert output.whitespace.is_whitespace is True
    assert output.whitespace.exact_match_count == 0
    assert output.whitespace.drug_only_trials > 100
    assert output.whitespace.indication_only_trials > 100

    # Agent should have populated indication_drugs
    assert len(output.whitespace.indication_drugs) > 10
    drug_names = [d.drug_name for d in output.whitespace.indication_drugs]
    assert "Memantine" in drug_names
    assert "Tetrabenazine" in drug_names

    # No exact-match trials for this pair
    assert output.trials == []

    # Agent should have produced a summary
    assert len(output.summary) > 50


# ------------------------------------------------------------------
# Active trials path: drug-disease pair with trials
# ------------------------------------------------------------------


async def test_agent_active_trials_path():
    """Agent finds active trials for semaglutide + diabetes.

    Expected behavior: detect_whitespace → not whitespace →
    search_trials → get_landscape → summary.
    """
    agent = ClinicalTrialsAgent()
    result = await agent.run(
        {
            "drug_name": "semaglutide",
            "disease_name": "diabetes",
        }
    )

    output = result["clinical_trials_output"]

    # Not whitespace
    assert output.whitespace is not None
    assert output.whitespace.is_whitespace is False
    assert output.whitespace.exact_match_count >= 10

    # Agent should have fetched trial details
    assert len(output.trials) >= 10
    nct_ids = [t.nct_id for t in output.trials]
    # All NCT IDs should be valid format
    for nct_id in nct_ids:
        assert nct_id.startswith("NCT")

    # Agent should have produced a summary
    assert len(output.summary) > 50


# ------------------------------------------------------------------
# Nonexistent drug: no data at all
# ------------------------------------------------------------------


async def test_agent_no_data():
    """Agent handles nonexistent drug gracefully.

    Expected behavior: detect_whitespace → whitespace + no_data →
    brief summary noting lack of data.
    """
    agent = ClinicalTrialsAgent()
    result = await agent.run(
        {
            "drug_name": "xyzzy_fake_drug_99999",
            "disease_name": "xyzzy_fake_disease_99999",
        }
    )

    output = result["clinical_trials_output"]

    # Whitespace with no data
    assert output.whitespace is not None
    assert output.whitespace.is_whitespace is True
    assert output.whitespace.exact_match_count == 0
    assert output.whitespace.drug_only_trials == 0
    assert output.whitespace.indication_only_trials == 0

    # No trials or terminated trials should exist
    assert output.trials == []

    # Agent should still produce a summary
    assert len(output.summary) > 20
