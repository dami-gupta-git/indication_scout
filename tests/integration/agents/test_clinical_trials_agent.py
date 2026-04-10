"""Integration tests for the clinical trials agent.

Hits the real ClinicalTrials.gov API and real Anthropic API.

Expected values verified by a live run on 2026-04-08 with:
  drug=riluzole, indication=ALS, date_before=2025-01-01, max_search_results=30
"""

import logging
from datetime import date

import pytest
from langchain_anthropic import ChatAnthropic

from indication_scout.agents.clinical_trials.clinical_trials_agent import (
    build_clinical_trials_agent,
    run_clinical_trials_agent,
)
from indication_scout.agents.clinical_trials.clinical_trials_output import (
    ClinicalTrialsOutput,
)

logger = logging.getLogger(__name__)

_CUTOFF = date(2025, 1, 1)

# Trials that must appear in search_trials output (stable, pre-cutoff)
_EXPECTED_NCT_IDS = {
    "NCT01709149",  # CK-2017357 + riluzole Phase 2, enrollment 711
    "NCT00868166",  # Olesoxime + riluzole Phase 3 (Roche), enrollment 512
    "NCT03127267",  # Masitinib + riluzole Phase 3 (AB Science), enrollment 495
    "NCT00542412",  # CARE study riluzole Phase 4 (Sanofi), enrollment 414
}

# Landscape competitors verified by live run
_EXPECTED_COMPETITORS = [
    ("ChaodongWang", "L-Carnitine Injection，1000mg once daily", "Phase 4", 4),
    ("Macquarie University, Australia", "Abacavir 600mg, Lamivudine 300mg and Dolutegravir 50mg (Triumeq)", "Phase 3", 12),
    ("Alector Inc.", "Latozinemab", "Phase 3", 17),
    ("Ferrer Internacional S.A.", "FAB122", "Phase 3", 201),
    ("Beijing Tiantan Hospital", "Nerve Growth Factor", "Phase 2/Phase 3", 60),
    ("Merit E. Cudkowicz, MD", "DNL343", "Phase 2/Phase 3", 249),
    ("Merit E. Cudkowicz, MD", "ABBV-CLS-7262 Dose 1", "Phase 2/Phase 3", 310),
    ("Oliver Blanchard", "Darifenacin 7.5 MG Extended Release Oral Tablet", "Phase 2", 30),
    ("Axoltis Pharma", "NX210c", "Phase 2", 80),
    ("argenx", "ARGX-119", "Phase 2", 60),
]


@pytest.fixture
def clinical_trials_agent():
    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0, max_tokens=4096)
    return build_clinical_trials_agent(llm, date_before=_CUTOFF, max_search_results=30)


async def test_riluzole_als_clinical_trials_agent(clinical_trials_agent):
    """End-to-end: clinical trials agent produces correct ClinicalTrialsOutput for riluzole + ALS.

    Verifies:
    - whitespace shows no whitespace (riluzole is established in ALS)
    - trials list contains known NCT IDs
    - landscape competitors match expected entries exactly
    - summary is non-empty and contains expected content
    """
    output = await run_clinical_trials_agent(clinical_trials_agent, "riluzole", "ALS")

    assert isinstance(output, ClinicalTrialsOutput)

    # --- whitespace ---
    assert output.whitespace is not None
    assert output.whitespace.is_whitespace is False
    assert output.whitespace.no_data is False
    assert output.whitespace.exact_match_count == 38
    assert output.whitespace.drug_only_trials == 88
    assert output.whitespace.indication_only_trials == 897
    assert output.whitespace.indication_drugs == []

    # --- trials ---
    assert len(output.trials) >= 10
    found_nct_ids = {t.nct_id for t in output.trials}
    assert _EXPECTED_NCT_IDS.issubset(found_nct_ids)

    # Spot-check the highest-enrollment trial in detail
    ck_trial = next(t for t in output.trials if t.nct_id == "NCT01709149")
    assert (
        ck_trial.title
        == "Study of Safety, Tolerability & Efficacy of CK-2017357 in Amyotrophic Lateral Sclerosis (ALS)"
    )
    assert ck_trial.phase == "Phase 2"
    assert ck_trial.overall_status == "COMPLETED"
    assert ck_trial.why_stopped is None
    assert ck_trial.indications == ["Amyotrophic Lateral Sclerosis"]
    assert ck_trial.sponsor == "Cytokinetics"
    assert ck_trial.enrollment == 711
    assert ck_trial.start_date == "2012-10"
    assert ck_trial.completion_date == "2014-03"
    assert len(ck_trial.interventions) == 3
    drug_interventions = [
        i for i in ck_trial.interventions if i.intervention_type == "Drug"
    ]
    ck_drug = next(i for i in drug_interventions if i.intervention_name == "CK-2017357")
    assert ck_drug.description == "CK-2017357 125 mg tablets twice daily"
    assert len(ck_trial.primary_outcomes) == 1
    assert "ALSFRS-R" in ck_trial.primary_outcomes[0].measure
    assert ck_trial.references == []

    # --- landscape ---
    assert output.landscape is not None
    assert output.landscape.total_trial_count == 897
    assert output.landscape.phase_distribution == {
        "Early Phase 1": 5,
        "Phase 1": 18,
        "Phase 1/Phase 2": 10,
        "Phase 2": 8,
        "Phase 2/Phase 3": 3,
        "Phase 3": 3,
        "Phase 4": 1,
    }

    assert len(output.landscape.competitors) == 10
    for i, (sponsor, drug, max_phase, enrollment) in enumerate(_EXPECTED_COMPETITORS):
        c = output.landscape.competitors[i]
        assert c.sponsor == sponsor, f"competitors[{i}].sponsor"
        assert c.drug_name == drug, f"competitors[{i}].drug_name"
        assert c.max_phase == max_phase, f"competitors[{i}].max_phase"
        assert c.total_enrollment == enrollment, f"competitors[{i}].total_enrollment"

    assert len(output.landscape.recent_starts) >= 1
    novartis_rs = next(
        (rs for rs in output.landscape.recent_starts if rs.nct_id == "NCT06643481"), None
    )
    assert novartis_rs is not None, "NCT06643481 (Novartis VHB937) not found in recent_starts"
    assert novartis_rs.sponsor == "Novartis Pharmaceuticals"
    assert novartis_rs.drug == "VHB937"
    assert novartis_rs.phase == "Phase 2"

    # --- terminated ---
    assert isinstance(output.terminated, list)

    # --- summary ---
    assert len(output.summary) > 100
    assert "riluzole" in output.summary.lower()
    assert "als" in output.summary.lower()
