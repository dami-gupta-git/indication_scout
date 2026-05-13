"""Integration tests for ClinicalTrialsClient."""

import pytest

from indication_scout.agents.clinical_trials.clinical_trials_tools import (
    _classify_stop_reason,
)
from indication_scout.data_sources.base_client import DataSourceError
from indication_scout.models.model_clinical_trials import (
    MeshTerm,
    TerminatedTrialsResult,
)

# --- Main functionality ---


async def test_get_landscape(clinical_trials_client):
    """Test get_landscape returns competitive landscape for an indication.

    get_landscape filters to Drug/Biological interventions, excludes vaccines,
    groups by sponsor + drug, and ranks by phase (desc) then most_recent_start (desc).
    """
    landscape = await clinical_trials_client.get_landscape("gastroparesis", top_n=10)

    # Verify IndicationLandscape.total_trial_count - gastroparesis has ~300+ trials
    assert landscape.total_trial_count > 200

    # Verify IndicationLandscape.competitors - requested top_n=10
    assert len(landscape.competitors) == 10

    # Verify IndicationLandscape.phase_distribution
    assert 10 < landscape.phase_distribution["Phase 2"] < 100
    assert 5 < landscape.phase_distribution["Phase 3"] < 50
    assert 1 < landscape.phase_distribution["Phase 4"] < 10

    # Verify IndicationLandscape.recent_starts - find a known 2024+ trial
    assert len(landscape.recent_starts) >= 1
    [tradipitant] = [rs for rs in landscape.recent_starts if rs.nct_id == "NCT06836557"]
    assert tradipitant.nct_id == "NCT06836557"
    assert tradipitant.sponsor == "Vanda Pharmaceuticals"
    assert tradipitant.drug == "Tradipitant"
    assert tradipitant.phase == "Phase 3"

    # --- Ranking: phase first, then most_recent_start descending ---
    # Phase 4 competitors must all appear before Phase 3 competitors
    phases = [c.max_phase for c in landscape.competitors]
    phase_4_indices = [i for i, p in enumerate(phases) if p == "Phase 4"]
    phase_3_indices = [i for i, p in enumerate(phases) if p == "Phase 3"]
    assert phase_4_indices, "Expected at least one Phase 4 competitor in top 10"
    assert phase_3_indices, "Expected at least one Phase 3 competitor in top 10"
    assert max(phase_4_indices) < min(phase_3_indices)

    # Within Phase 4, competitors are ordered by most_recent_start descending
    phase_4_competitors = [c for c in landscape.competitors if c.max_phase == "Phase 4"]
    starts = [c.most_recent_start for c in phase_4_competitors]
    assert starts == sorted(starts, reverse=True)

    # --- Vaccine exclusion ---
    # No competitor name should match vaccine keywords
    for c in landscape.competitors:
        name_lower = c.drug_name.lower()
        assert "vaccine" not in name_lower
        assert "vax" not in name_lower
        assert "immuniz" not in name_lower

    # --- most_recent_start field is populated ---
    # Find Vanda Pharmaceuticals / Tradipitant (Phase 3, two trials, most recent 2024-01-09)
    [tradipitant_comp] = [
        c
        for c in landscape.competitors
        if c.sponsor == "Vanda Pharmaceuticals" and c.drug_name == "Tradipitant"
    ]
    assert tradipitant_comp.sponsor == "Vanda Pharmaceuticals"
    assert tradipitant_comp.drug_name == "Tradipitant"
    assert tradipitant_comp.drug_type == "Drug"
    assert tradipitant_comp.max_phase == "Phase 3"
    assert tradipitant_comp.trial_count == 2
    assert tradipitant_comp.statuses == {"COMPLETED", "RECRUITING"}
    assert tradipitant_comp.total_enrollment == 1092
    assert tradipitant_comp.most_recent_start == "2024-01-09"


async def test_get_trial(clinical_trials_client):
    """Test get_trial returns a single trial by NCT ID."""
    # Fetch NCT00127933 - XeNA Study (Roche breast cancer trial)
    trial = await clinical_trials_client.get_trial("NCT00127933")

    # Verify all Trial fields with exact values
    assert trial.nct_id == "NCT00127933"
    assert (
        trial.title
        == "XeNA Study - A Study of Xeloda (Capecitabine) in Patients With Invasive Breast Cancer"
    )
    assert trial.phase == "Phase 4"
    assert trial.overall_status == "COMPLETED"
    assert trial.why_stopped is None
    assert trial.indications == ["Breast Cancer"]
    assert trial.sponsor == "Hoffmann-La Roche"
    assert trial.enrollment == 157
    assert trial.start_date == "2005-08"
    assert trial.completion_date == "2009-04"
    assert trial.references == []

    # Verify interventions - trial has 5 drug interventions
    assert len(trial.interventions) == 5
    [herceptin] = [
        i
        for i in trial.interventions
        if i.intervention_name == "Herceptin (HER2-neu positive patients only)"
    ]
    assert herceptin.intervention_type == "Drug"

    # Verify primary_outcomes
    assert len(trial.primary_outcomes) == 1
    assert (
        trial.primary_outcomes[0].measure
        == "Percentage of Participants Assessed for Pathological Complete Response (pCR) Plus Near Complete (npCR) in Primary Breast Tumor at Time of Definitive Surgery"
    )
    assert (
        trial.primary_outcomes[0].time_frame
        == "at the time of definitive surgery; after four 3-week cycles (3-4 months)"
    )


async def test_search_trials_nash_trial_fields(clinical_trials_client):
    """Verify all Trial fields (including MeSH ancestors) for NCT04971785.

    Covers both full-field parsing and conditionBrowseModule.ancestors
    extraction in a single live call.
    """
    result = await clinical_trials_client.search_trials(
        drug="semaglutide",
        mesh_term="Non-alcoholic Fatty Liver Disease",
    )

    [nash_trial] = [t for t in result.trials if t.nct_id == "NCT04971785"]

    assert nash_trial.nct_id == "NCT04971785"
    assert (
        nash_trial.title
        == "Study of Semaglutide, and Cilofexor/Firsocostat, Alone and in Combination, in Adults With Cirrhosis Due to Nonalcoholic Steatohepatitis (NASH)"
    )
    assert nash_trial.phase == "Phase 2"
    assert nash_trial.overall_status == "COMPLETED"
    assert nash_trial.why_stopped is None
    assert nash_trial.indications == ["Nonalcoholic Steatohepatitis"]
    assert nash_trial.sponsor == "Gilead Sciences"
    assert nash_trial.enrollment == 457
    assert nash_trial.start_date == "2021-08-09"
    assert nash_trial.completion_date == "2024-11-12"
    assert nash_trial.references == []

    assert len(nash_trial.interventions) == 4
    [sema] = [
        i
        for i in nash_trial.interventions
        if i.intervention_name == "Semaglutide (SEMA)"
    ]
    assert sema.intervention_type == "Drug"

    assert len(nash_trial.primary_outcomes) == 1
    assert nash_trial.primary_outcomes[0].measure.startswith(
        "Percentage of Participants Who Achieved"
    )

    # MeSH conditions + ancestors from conditionBrowseModule
    assert nash_trial.mesh_conditions == [
        MeshTerm(id="D065626", term="Non-alcoholic Fatty Liver Disease"),
    ]
    ancestor_pairs = [(m.id, m.term) for m in nash_trial.mesh_ancestors]
    assert ancestor_pairs == [
        ("D005234", "Fatty Liver"),
        ("D008107", "Liver Diseases"),
        ("D004066", "Digestive System Diseases"),
    ]


# --- Edge cases and weird inputs ---


async def test_get_trial_nonexistent_nct_id_raises_error(clinical_trials_client):
    """Test that a nonexistent NCT ID raises DataSourceError."""
    with pytest.raises(DataSourceError, match="404") as exc_info:
        await clinical_trials_client.get_trial("NCT99999999")

    assert "NCT99999999" in str(exc_info.value)


@pytest.mark.parametrize(
    "drug, mesh_term",
    [
        ("xyzzy_not_a_real_drug_12345", "Diabetes Mellitus"),
        ("semaglutide", "xyzzy_fake_mesh_term_99999"),
    ],
    ids=["nonexistent_drug", "nonexistent_mesh_term"],
)
async def test_search_trials_invalid_input_returns_empty(
    clinical_trials_client, drug, mesh_term
):
    """search_trials returns an empty SearchTrialsResult (no error) for unresolvable terms."""
    result = await clinical_trials_client.search_trials(
        drug=drug,
        mesh_term=mesh_term,
    )
    assert result.total_count == 0
    assert result.trials == []


async def test_get_landscape_nonexistent_indication_returns_empty(
    clinical_trials_client,
):
    """Test that nonexistent indication returns empty landscape."""
    landscape = await clinical_trials_client.get_landscape(
        "xyzzy_fake_indication_99999", top_n=10
    )

    assert landscape.total_trial_count == 0
    assert landscape.competitors == []
    assert landscape.phase_distribution == {}


async def test_get_terminated_trials_nonexistent_query_returns_empty(
    clinical_trials_client,
):
    """Nonexistent drug and MeSH term returns empty TerminatedTrialsResult."""
    outcomes = await clinical_trials_client.get_terminated_trials(
        "xyzzy_not_a_real_term_12345",
        "xyzzy_not_a_real_mesh_term_12345",
    )

    assert outcomes == TerminatedTrialsResult()


@pytest.mark.parametrize(
    "nct_id, why_stopped_fragment, expected_category",
    [
        (
            "NCT00109577",
            "no adverse events",  # negated safety phrase
            "Unable to recruit",
        ),
        (
            "NCT06134661",
            "unrelated to safety",  # negated safety phrase
            "enrollment",
        ),
    ],
)
async def test_classify_stop_reason_negation_on_live_data(
    clinical_trials_client, nct_id, why_stopped_fragment, expected_category
):
    """_classify_stop_reason does not misclassify negated safety phrases as 'safety'.

    Fetches real terminated trials whose why_stopped text contains a negated
    safety phrase and asserts the category is not 'safety'.
    """

    trial = await clinical_trials_client.get_trial(nct_id)

    assert trial.why_stopped is not None
    assert why_stopped_fragment in trial.why_stopped.lower()
    assert expected_category in _classify_stop_reason(trial.why_stopped)
    assert _classify_stop_reason(trial.why_stopped) != "safety"


async def test_get_landscape_total_count_exceeds_fetch_cap(clinical_trials_client):
    """total_trial_count reflects the real API count, not the number of fetched trials.

    CLINICAL_TRIALS_LANDSCAPE_MAX_TRIALS caps the fetch at 50 trials, but the API
    reports the true total. For type 2 diabetes, thousands of trials exist.
    """
    from indication_scout.config import get_settings

    landscape = await clinical_trials_client.get_landscape(
        "Diabetes Mellitus, Type 2", top_n=5
    )

    assert (
        landscape.total_trial_count
        > get_settings().clinical_trials_landscape_max_trials
    )
