"""Unit tests for ClinicalTrialsClient helper functions."""

import pytest

from indication_scout.agents.clinical_trials.clinical_trials_tools import (
    _classify_stop_reason,
)
from indication_scout.data_sources.clinical_trials import ClinicalTrialsClient

# --- _classify_stop_reason keyword-based classification ---


@pytest.mark.parametrize(
    "why_stopped,expected_category",
    [
        # Efficacy keywords
        ("Lack of efficacy in interim analysis", "efficacy"),
        ("Study stopped due to futility", "efficacy"),
        ("No benefit observed in treatment arm", "efficacy"),
        ("Primary endpoint not met - lack of efficacy", "efficacy"),
        # Safety keywords
        ("Safety signal observed in treatment arm", "safety"),
        ("Serious adverse events reported", "safety"),
        ("Stopped due to toxicity concerns", "safety"),
        ("Unexpected side effect in patients", "safety"),
        # Enrollment keywords
        ("Difficulty with patient enrollment", "enrollment"),
        ("Slow accrual rate", "enrollment"),
        ("Recruitment challenges", "enrollment"),
        # Business keywords
        ("The trial was terminated for strategic reasons.", "business"),
        ("Sponsor business decision", "business"),
        ("Funding withdrawn", "business"),
        ("Commercial considerations", "business"),
        # New safety keywords
        ("safety signal detected in interim review", "safety"),
        ("safety concern raised by DSMB", "safety"),
        ("FDA issued a clinical hold", "safety"),
        # New efficacy keyword
        ("no significant difference between arms", "efficacy"),
        # Unknown - no text
        (None, "unknown"),
        ("", "unknown"),
    ],
)
def test_classify_stop_reason(why_stopped, expected_category):
    """_classify_stop_reason should classify based on keywords."""
    assert _classify_stop_reason(why_stopped) == expected_category


def test_classify_stop_reason_case_insensitive():
    """_classify_stop_reason should match keywords case-insensitively."""
    assert _classify_stop_reason("LACK OF EFFICACY") == "efficacy"
    assert _classify_stop_reason("Safety Concerns") == "safety"
    assert _classify_stop_reason("BUSINESS Decision") == "business"


# --- ClinicalTrialsClient static helper methods ---


@pytest.mark.parametrize(
    "phases,expected",
    [
        # Single phases
        (["PHASE1"], "Phase 1"),
        (["PHASE2"], "Phase 2"),
        (["PHASE3"], "Phase 3"),
        (["PHASE4"], "Phase 4"),
        (["EARLY_PHASE1"], "Early Phase 1"),
        (["NA"], "Not Applicable"),
        # Combined phases
        (["PHASE1", "PHASE2"], "Phase 1/Phase 2"),
        (["PHASE2", "PHASE3"], "Phase 2/Phase 3"),
        (["PHASE3", "PHASE4"], "Phase 3/Phase 4"),
        # Empty list
        ([], "Not Applicable"),
    ],
)
def test_normalize_phase(phases, expected):
    """_normalize_phase should convert API phase list to display string."""
    assert ClinicalTrialsClient._normalize_phase(phases) == expected


@pytest.mark.parametrize(
    "phase,expected_rank",
    [
        ("Not Applicable", 0),
        ("Early Phase 1", 1),
        ("Phase 1", 2),
        ("Phase 1/Phase 2", 3),
        ("Phase 2", 4),
        ("Phase 2/Phase 3", 5),
        ("Phase 3", 6),
        ("Phase 3/Phase 4", 7),
        ("Phase 4", 8),
        # Unknown phase defaults to 0
        ("Unknown", 0),
        ("", 0),
    ],
)
def test_phase_rank(phase, expected_rank):
    """_phase_rank should return numeric ranking for phase comparison."""
    assert ClinicalTrialsClient._phase_rank(phase) == expected_rank


def test_phase_rank_ordering():
    """_phase_rank should order phases correctly: Phase 4 > Phase 3 > Phase 2 > Phase 1."""
    assert ClinicalTrialsClient._phase_rank(
        "Phase 4"
    ) > ClinicalTrialsClient._phase_rank("Phase 3")
    assert ClinicalTrialsClient._phase_rank(
        "Phase 3"
    ) > ClinicalTrialsClient._phase_rank("Phase 2")
    assert ClinicalTrialsClient._phase_rank(
        "Phase 2"
    ) > ClinicalTrialsClient._phase_rank("Phase 1")
    assert ClinicalTrialsClient._phase_rank(
        "Phase 1"
    ) > ClinicalTrialsClient._phase_rank("Early Phase 1")


@pytest.mark.parametrize(
    "date_struct,expected",
    [
        ({"date": "2021-03-15"}, "2021-03-15"),
        ({"date": "2024-01"}, "2024-01"),
        ({}, None),
        (None, None),
    ],
)
def test_extract_date(date_struct, expected):
    """_extract_date should extract date string from v2 date struct."""
    assert ClinicalTrialsClient._extract_date(date_struct) == expected


# --- _parse_trial populates mesh_ancestors ---


def test_parse_trial_populates_mesh_ancestors(tmp_path):
    """_parse_trial extracts ancestors from derivedSection.conditionBrowseModule."""
    study = {
        "protocolSection": {
            "identificationModule": {
                "nctId": "NCT77777777",
                "briefTitle": "Ancestor fixture trial",
            },
            "statusModule": {"overallStatus": "COMPLETED"},
            "designModule": {"phases": ["PHASE2"]},
            "conditionsModule": {"conditions": ["Type 2 Diabetes Mellitus"]},
            "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Sponsor"}},
            "armsInterventionsModule": {
                "interventions": [{"type": "DRUG", "name": "DrugY"}]
            },
            "outcomesModule": {},
            "referencesModule": {},
            "descriptionModule": {},
        },
        "derivedSection": {
            "conditionBrowseModule": {
                "meshes": [{"id": "D003924", "term": "Diabetes Mellitus, Type 2"}],
                "ancestors": [
                    {"id": "D003920", "term": "Diabetes Mellitus"},
                    {"id": "D044882", "term": "Glucose Metabolism Disorders"},
                ],
            }
        },
    }

    client = ClinicalTrialsClient(cache_dir=tmp_path)
    trial = client._parse_trial(study)

    assert len(trial.mesh_conditions) == 1
    assert trial.mesh_conditions[0].id == "D003924"
    assert trial.mesh_conditions[0].term == "Diabetes Mellitus, Type 2"

    assert len(trial.mesh_ancestors) == 2
    assert trial.mesh_ancestors[0].id == "D003920"
    assert trial.mesh_ancestors[0].term == "Diabetes Mellitus"
    assert trial.mesh_ancestors[1].id == "D044882"
    assert trial.mesh_ancestors[1].term == "Glucose Metabolism Disorders"


def test_parse_trial_missing_ancestors_defaults_empty(tmp_path):
    """_parse_trial returns empty mesh_ancestors when derivedSection lacks ancestors."""
    study = {
        "protocolSection": {
            "identificationModule": {
                "nctId": "NCT88888888",
                "briefTitle": "No ancestors fixture",
            },
            "statusModule": {"overallStatus": "COMPLETED"},
            "designModule": {"phases": ["PHASE1"]},
            "conditionsModule": {"conditions": ["Condition X"]},
            "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Sponsor"}},
            "armsInterventionsModule": {
                "interventions": [{"type": "DRUG", "name": "DrugZ"}]
            },
            "outcomesModule": {},
            "referencesModule": {},
            "descriptionModule": {},
        },
    }

    client = ClinicalTrialsClient(cache_dir=tmp_path)
    trial = client._parse_trial(study)

    assert trial.mesh_ancestors == []
