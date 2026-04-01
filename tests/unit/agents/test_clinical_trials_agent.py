"""Unit tests for ClinicalTrialsAgent._parse_result."""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from indication_scout.agents.clinical_trials import ClinicalTrialsAgent

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Helpers: fake message objects matching langchain's message protocol
# ------------------------------------------------------------------


def _tool_msg(name: str, content):
    """Simulate a ToolMessage with .name and .content."""
    return SimpleNamespace(name=name, content=content)


def _ai_msg(content):
    """Simulate an AIMessage with .content and no .name."""
    return SimpleNamespace(content=content)


# ------------------------------------------------------------------
# _parse_result: whitespace path
# ------------------------------------------------------------------


def test_parse_result_whitespace_path():
    """Agent called detect_whitespace + get_terminated + get_landscape."""
    whitespace_data = {
        "is_whitespace": True,
        "no_data": False,
        "exact_match_count": 0,
        "drug_only_trials": 120,
        "indication_only_trials": 250,
        "indication_drugs": [
            {
                "nct_id": "NCT00652457",
                "drug_name": "Memantine",
                "indication": "Huntington's Disease",
                "phase": "Phase 4",
                "status": "COMPLETED",
            }
        ],
    }

    terminated_data = [
        {
            "nct_id": "NCT01234567",
            "drug_name": "SomeDrug",
            "indication": "Huntington's Disease",
            "phase": "Phase 2",
            "why_stopped": "Lack of efficacy",
            "stop_category": "efficacy",
        }
    ]

    landscape_data = {
        "total_trial_count": 30,
        "competitors": [],
        "phase_distribution": {"Phase 2": 10, "Phase 3": 5},
        "recent_starts": [],
    }

    summary_text = "No trials exist for this drug-disease pair. One prior attempt failed for efficacy."

    result = {
        "messages": [
            _ai_msg("Let me check whitespace first."),
            _tool_msg("detect_whitespace", whitespace_data),
            _ai_msg("Whitespace found. Checking terminated trials."),
            _tool_msg("get_terminated", terminated_data),
            _ai_msg("Found a failure. Checking landscape."),
            _tool_msg("get_landscape", landscape_data),
            _ai_msg(summary_text),
        ]
    }

    output = ClinicalTrialsAgent._parse_result(result)

    # Whitespace
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

    # Terminated
    assert len(output.terminated) == 1
    assert output.terminated[0].nct_id == "NCT01234567"
    assert output.terminated[0].drug_name == "SomeDrug"
    assert output.terminated[0].indication == "Huntington's Disease"
    assert output.terminated[0].phase == "Phase 2"
    assert output.terminated[0].why_stopped == "Lack of efficacy"
    assert output.terminated[0].stop_category == "efficacy"

    # Landscape
    assert output.landscape is not None
    assert output.landscape.total_trial_count == 30
    assert output.landscape.competitors == []
    assert output.landscape.phase_distribution == {"Phase 2": 10, "Phase 3": 5}
    assert output.landscape.recent_starts == []

    # No trials called
    assert output.trials == []

    # Summary
    assert output.summary == summary_text


# ------------------------------------------------------------------
# _parse_result: active trials path
# ------------------------------------------------------------------


def test_parse_result_active_trials_path():
    """Agent called detect_whitespace + search_trials + get_landscape."""
    whitespace_data = {
        "is_whitespace": False,
        "no_data": False,
        "exact_match_count": 5,
        "drug_only_trials": 200,
        "indication_only_trials": 1000,
        "indication_drugs": [],
    }

    trials_data = [
        {
            "nct_id": "NCT02970942",
            "title": "Semaglutide NASH Trial",
            "brief_summary": "Testing semaglutide in NASH",
            "phase": "Phase 2",
            "overall_status": "COMPLETED",
            "why_stopped": None,
            "indications": ["NASH"],
            "interventions": [
                {
                    "intervention_type": "Drug",
                    "intervention_name": "Semaglutide",
                    "description": "Once weekly injection",
                }
            ],
            "sponsor": "Novo Nordisk",
            "enrollment": 320,
            "start_date": "2017-03-01",
            "completion_date": "2020-06-01",
            "primary_outcomes": [
                {"measure": "NASH resolution", "time_frame": "72 weeks"}
            ],
            "references": ["33185364"],
        }
    ]

    landscape_data = {
        "total_trial_count": 47,
        "competitors": [
            {
                "sponsor": "Madrigal Pharmaceuticals",
                "drug_name": "Resmetirom",
                "drug_type": "Drug",
                "max_phase": "Phase 3",
                "trial_count": 3,
                "statuses": ["COMPLETED", "RECRUITING"],
                "total_enrollment": 2000,
            }
        ],
        "phase_distribution": {"Phase 2": 25, "Phase 3": 12},
        "recent_starts": [],
    }

    summary_text = "5 trials exist. NASH is a crowded space with 47 active trials."

    result = {
        "messages": [
            _ai_msg("Checking whitespace."),
            _tool_msg("detect_whitespace", whitespace_data),
            _ai_msg("Trials exist. Fetching details."),
            _tool_msg("search_trials", trials_data),
            _ai_msg("Got trial details. Checking landscape."),
            _tool_msg("get_landscape", landscape_data),
            _ai_msg(summary_text),
        ]
    }

    output = ClinicalTrialsAgent._parse_result(result)

    # Whitespace
    assert output.whitespace is not None
    assert output.whitespace.is_whitespace is False
    assert output.whitespace.exact_match_count == 5

    # Trials
    assert len(output.trials) == 1
    assert output.trials[0].nct_id == "NCT02970942"
    assert output.trials[0].title == "Semaglutide NASH Trial"
    assert output.trials[0].brief_summary == "Testing semaglutide in NASH"
    assert output.trials[0].phase == "Phase 2"
    assert output.trials[0].overall_status == "COMPLETED"
    assert output.trials[0].why_stopped is None
    assert output.trials[0].indications == ["NASH"]
    assert len(output.trials[0].interventions) == 1
    assert output.trials[0].interventions[0].intervention_name == "Semaglutide"
    assert output.trials[0].sponsor == "Novo Nordisk"
    assert output.trials[0].enrollment == 320
    assert output.trials[0].references == ["33185364"]

    # Landscape
    assert output.landscape is not None
    assert output.landscape.total_trial_count == 47
    assert len(output.landscape.competitors) == 1
    assert output.landscape.competitors[0].drug_name == "Resmetirom"
    assert output.landscape.competitors[0].sponsor == "Madrigal Pharmaceuticals"
    assert output.landscape.competitors[0].max_phase == "Phase 3"

    # No terminated
    assert output.terminated == []

    # Summary
    assert output.summary == summary_text


# ------------------------------------------------------------------
# _parse_result: minimal — only whitespace called
# ------------------------------------------------------------------


def test_parse_result_minimal():
    """Agent only called detect_whitespace and returned a summary."""
    whitespace_data = {
        "is_whitespace": True,
        "no_data": True,
        "exact_match_count": 0,
        "drug_only_trials": 0,
        "indication_only_trials": 0,
        "indication_drugs": [],
    }

    summary_text = "No data found for this drug or disease."

    result = {
        "messages": [
            _ai_msg("Let me check."),
            _tool_msg("detect_whitespace", whitespace_data),
            _ai_msg(summary_text),
        ]
    }

    output = ClinicalTrialsAgent._parse_result(result)

    assert output.whitespace is not None
    assert output.whitespace.is_whitespace is True
    assert output.whitespace.no_data is True
    assert output.whitespace.exact_match_count == 0
    assert output.whitespace.drug_only_trials == 0
    assert output.whitespace.indication_only_trials == 0
    assert output.whitespace.indication_drugs == []
    assert output.trials == []
    assert output.landscape is None
    assert output.terminated == []
    assert output.summary == summary_text


# ------------------------------------------------------------------
# _parse_result: empty messages
# ------------------------------------------------------------------


def test_parse_result_empty_messages():
    """Handles empty message list gracefully."""
    output = ClinicalTrialsAgent._parse_result({"messages": []})

    assert output.whitespace is None
    assert output.trials == []
    assert output.landscape is None
    assert output.terminated == []
    assert output.summary == ""


# ------------------------------------------------------------------
# _parse_result: AI message with list-of-blocks content
# ------------------------------------------------------------------


def test_parse_result_ai_message_block_content():
    """Handles AIMessage where content is a list of text blocks."""
    result = {
        "messages": [
            _ai_msg(
                [
                    {"type": "text", "text": "First part."},
                    {"type": "text", "text": "Second part."},
                ]
            ),
        ]
    }

    output = ClinicalTrialsAgent._parse_result(result)

    assert output.summary == "First part.\nSecond part."


# ------------------------------------------------------------------
# run(): input normalisation — drug_name and disease_name are lowercased
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "drug_input, disease_input, expected_drug, expected_disease",
    [
        ("Metformin", "Type 2 Diabetes", "metformin", "type 2 diabetes"),
        ("SEMAGLUTIDE", "OBESITY", "semaglutide", "obesity"),
        ("Trastuzumab", "HER2-Positive Breast Cancer", "trastuzumab", "her2-positive breast cancer"),
        ("aspirin", "heart disease", "aspirin", "heart disease"),
    ],
)
async def test_run_lowercases_drug_and_disease(
    drug_input, disease_input, expected_drug, expected_disease
):
    """ClinicalTrialsAgent.run() lowercases drug_name and disease_name before use."""
    fake_agent_result = {
        "messages": [
            _ai_msg("No notable findings."),
        ]
    }

    mock_agent = AsyncMock()
    mock_agent.ainvoke = AsyncMock(return_value=fake_agent_result)

    with patch(
        "indication_scout.agents.clinical_trials.create_agent",
        return_value=mock_agent,
    ), patch(
        "indication_scout.agents.clinical_trials.build_clinical_trials_tools",
        return_value=[],
    ), patch(
        "indication_scout.agents.clinical_trials.ChatAnthropic",
        return_value=MagicMock(),
    ):
        agent = ClinicalTrialsAgent()
        await agent.run({"drug_name": drug_input, "disease_name": disease_input})

    call_args = mock_agent.ainvoke.call_args
    user_message = call_args[0][0]["messages"][0]["content"]
    assert expected_drug in user_message
    assert expected_disease in user_message
