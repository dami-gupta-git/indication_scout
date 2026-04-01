"""Integration tests for ClinicalTrialsClient."""

import logging

import pytest

from indication_scout.data_sources.base_client import DataSourceError
from indication_scout.markers import no_review

logger = logging.getLogger(__name__)


# --- Main functionality ---


async def test_get_landscape(clinical_trials_client):
    """Test get_landscape returns competitive landscape for an indication.

    get_landscape filters to intervention_type in ("Drug", "Biological") only,
    groups by sponsor + drug, and ranks by phase (desc) then enrollment (desc).
    """
    # Use a smaller, more stable indication for predictable results
    landscape = await clinical_trials_client.get_landscape("gastroparesis", top_n=10)

    # Verify IndicationLandscape.total_trial_count - gastroparesis has ~300+ trials
    assert landscape.total_trial_count > 200

    # Verify IndicationLandscape.competitors - requested top_n=10
    assert len(landscape.competitors) == 10

    # Verify IndicationLandscape.phase_distribution
    assert 10 < landscape.phase_distribution["Phase 2"] < 100
    assert 5 < landscape.phase_distribution["Phase 3"] < 50
    assert 1 < landscape.phase_distribution["Phase 4"] < 30

    # Verify IndicationLandscape.recent_starts - find a known 2024+ trial
    assert len(landscape.recent_starts) >= 1
    [tradipitant] = [rs for rs in landscape.recent_starts if rs.nct_id == "NCT06836557"]
    assert tradipitant.nct_id == "NCT06836557"
    assert tradipitant.sponsor == "Vanda Pharmaceuticals"
    assert tradipitant.drug == "Tradipitant"
    assert tradipitant.phase == "Phase 3"

    # Find Chinese University of Hong Kong with Esomeprazole - top ranked competitor
    [cuhk] = [
        c
        for c in landscape.competitors
        if c.sponsor == "Chinese University of Hong Kong"
        and c.drug_name == "Esomeprazole"
    ]

    # Verify all CompetitorEntry fields
    assert cuhk.sponsor == "Chinese University of Hong Kong"
    assert cuhk.drug_name == "Esomeprazole"
    assert cuhk.drug_type == "Drug"
    assert cuhk.max_phase == "Phase 4"
    assert cuhk.trial_count == 1
    assert cuhk.statuses == {"COMPLETED"}
    assert cuhk.total_enrollment == 155


# TODO remove
@no_review
async def test_get_trial_flow(clinical_trials_client):
    """Test get_trial returns a single trial by NCT ID."""
    # Fetch NCT03819153
    trial = await clinical_trials_client.get_trial("NCT03819153")
    logger.info(trial)


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


async def test_search_trials(clinical_trials_client):
    """Test search_trials returns trials for a drug-indication pair."""
    # Search for trastuzumab + breast cancer, find a specific known trial
    trials = await clinical_trials_client.search_trials(
        drug="trastuzumab",
        indication="breast cancer",
        max_results=50,
        phase_filter="PHASE4",
    )

    # Find NCT00127933 - XeNA Study (Roche, includes Herceptin)
    [xena] = [t for t in trials if t.nct_id == "NCT00127933"]

    # Verify all Trial fields with exact values
    assert xena.nct_id == "NCT00127933"
    assert (
        xena.title
        == "XeNA Study - A Study of Xeloda (Capecitabine) in Patients With Invasive Breast Cancer"
    )
    assert xena.phase == "Phase 4"
    assert xena.overall_status == "COMPLETED"
    assert xena.why_stopped is None
    assert xena.indications == ["Breast Cancer"]
    assert xena.sponsor == "Hoffmann-La Roche"
    assert xena.enrollment == 157
    assert xena.start_date == "2005-08"
    assert xena.completion_date == "2009-04"
    assert xena.references == []

    # Verify interventions - trial has 5 interventions including Herceptin
    assert len(xena.interventions) == 5
    [herceptin] = [
        i
        for i in xena.interventions
        if i.intervention_name == "Herceptin (HER2-neu positive patients only)"
    ]
    assert herceptin.intervention_type == "Drug"

    # Verify primary_outcomes
    assert len(xena.primary_outcomes) == 1
    assert (
        xena.primary_outcomes[0].measure
        == "Percentage of Participants Assessed for Pathological Complete Response (pCR) Plus Near Complete (npCR) in Primary Breast Tumor at Time of Definitive Surgery"
    )
    assert (
        xena.primary_outcomes[0].time_frame
        == "at the time of definitive surgery; after four 3-week cycles (3-4 months)"
    )


async def test_search_trials_drug_only(clinical_trials_client):
    """Test search_trials returns trials for a drug without specifying indication."""
    trials = await clinical_trials_client.search_trials(
        drug="semaglutide",
        max_results=300,
    )

    # Semaglutide has many trials across diabetes, obesity, NASH, etc.
    assert len(trials) >= 20

    # Verify trials span multiple indications
    all_indications = set()
    for trial in trials:
        all_indications.update(trial.indications)
    assert len(all_indications) >= 10


async def test_search_trials_nash_trial_fields(clinical_trials_client):
    """Verify all Trial fields for NCT04971785 (Gilead NASH/semaglutide trial)."""
    trials = await clinical_trials_client.search_trials(
        drug="semaglutide",
        indication="nonalcoholic steatohepatitis",
        max_results=100,
    )

    [nash_trial] = [t for t in trials if t.nct_id == "NCT04971785"]

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


async def test_search_trials_indication_only(clinical_trials_client):
    """Test search_trials returns trials for an indication without specifying drug."""
    # Search for gastroparesis trials across all drugs
    trials = await clinical_trials_client.search_trials(
        drug="",
        indication="gastroparesis",
        max_results=50,
        phase_filter="PHASE4",
    )

    # Gastroparesis has multiple Phase 4 trials
    assert len(trials) >= 5

    # Find NCT00492622 - University of Louisville omeprazole PK study
    [pk_trial] = [t for t in trials if t.nct_id == "NCT00492622"]

    # Verify all Trial fields with exact values
    assert pk_trial.nct_id == "NCT00492622"
    assert (
        pk_trial.title
        == "Pharmacokinetics of Immediate-Release vs. Delayed-Release Omeprazole in Gastroparesis"
    )
    assert pk_trial.phase == "Phase 4"
    assert pk_trial.overall_status == "COMPLETED"
    assert pk_trial.why_stopped is None
    assert pk_trial.indications == [
        "Gastroparesis",
        "Gastroesophageal Reflux Disease",
    ]
    assert pk_trial.sponsor == "University of Louisville"
    assert pk_trial.enrollment == 12
    assert pk_trial.start_date == "2007-06"
    assert pk_trial.completion_date == "2008-12"
    assert pk_trial.references == ["19925497"]

    # Verify interventions - trial has 2 drug interventions
    assert len(pk_trial.interventions) == 2
    intervention_names = [i.intervention_name for i in pk_trial.interventions]
    assert "Immediate-release omeprazole" in intervention_names
    assert "Delayed-release omeprazole" in intervention_names

    # Verify primary_outcomes
    assert len(pk_trial.primary_outcomes) == 3
    assert (
        pk_trial.primary_outcomes[0].measure
        == "Time to Maximal Omeprazole Concentration (Tmax)"
    )
    assert (
        pk_trial.primary_outcomes[0].time_frame
        == "10, 20, 30, 45, 60, 90, 120, 150, 180, 210, 240 and 300 min after the study drug was ingested on day 7 of treatment"
    )


async def test_search_trials_phase_filter(clinical_trials_client):
    """Test that phase_filter only returns trials matching the specified phase."""
    # Search for Phase 3 trials only
    trials = await clinical_trials_client.search_trials(
        drug="semaglutide",
        indication="diabetes",
        max_results=20,
        phase_filter="PHASE3",
    )

    # All returned trials must be Phase 3
    assert len(trials) >= 5  # semaglutide + diabetes has many Phase 3 trials
    for trial in trials:
        assert "Phase 3" in trial.phase


async def test_get_terminated(clinical_trials_client):
    """Test get_terminated returns terminated trials for a drug and indication.

    Runs two queries:
      - Drug query (semaglutide): only safety/efficacy terminations
      - Indication query (overweight): all terminations in this space
    Returns union deduped by nct_id.
    """
    trials = await clinical_trials_client.get_terminated(
        "semaglutide", "overweight", max_results=30
    )

    assert len(trials) >= 1

    # NCT02499705 — safety termination from overweight indication query
    # "2 out of 3 participants had clinically elevated fasting insulin"
    [safety_trial] = [t for t in trials if t.nct_id == "NCT02499705"]
    assert safety_trial.stop_category == "safety"


async def test_detect_whitespace(clinical_trials_client):
    """Test detect_whitespace identifies unexplored drug-indication pairs.

    When is_whitespace=True, returns indication_drugs (other drugs being tested
    for this indication) ranked by phase (desc) then active status, deduplicated
    by drug_name.
    When is_whitespace=False, indication_drugs is empty.
    """
    # Tirzepatide + Huntington disease = whitespace (no exact matches)
    result = await clinical_trials_client.detect_whitespace(
        "tirzepatide", "Huntington disease"
    )

    # Verify WhitespaceResult fields
    assert result.is_whitespace is True
    assert result.exact_match_count == 0
    assert 150 < result.drug_only_trials < 300
    assert 200 < result.indication_only_trials < 400

    # Verify indication_drugs
    assert 40 < len(result.indication_drugs) < 60

    # Verify deduplication: all drug_names should be unique
    drug_names = [cd.drug_name for cd in result.indication_drugs]
    assert len(drug_names) == len(set(drug_names))

    # Verify known drugs are found (Memantine and Tetrabenazine are Phase 4 HD drugs)
    assert "Memantine" in drug_names
    assert "Tetrabenazine" in drug_names

    # Verify ranking: first drugs should be Phase 4 (highest phase)
    assert result.indication_drugs[0].phase == "Phase 4"
    assert result.indication_drugs[1].phase == "Phase 4"
    assert result.indication_drugs[2].phase == "Phase 4"

    # Find Memantine (Phase 4 completed trial) and verify all IndicationDrug fields
    [memantine] = [cd for cd in result.indication_drugs if cd.drug_name == "Memantine"]
    assert memantine.nct_id == "NCT00652457"
    assert memantine.drug_name == "Memantine"
    assert memantine.indication == "Huntington's Disease"
    assert memantine.phase == "Phase 4"
    assert memantine.status == "COMPLETED"

    # Verify Phase 2+ filter: no Phase 1 or Early Phase 1 trials
    for cd in result.indication_drugs:
        assert "Phase 1" not in cd.phase
        assert "Early Phase 1" not in cd.phase


async def test_detect_whitespace_not_whitespace(clinical_trials_client):
    """Test detect_whitespace when exact matches exist (is_whitespace=False).

    When trials exist for the drug-indication pair, is_whitespace=False
    and indication_drugs is empty (no need to show competitors).
    """
    # Semaglutide + diabetes = NOT whitespace (many exact matches)
    result = await clinical_trials_client.detect_whitespace("semaglutide", "diabetes")

    # Verify WhitespaceResult fields for non-whitespace case
    assert result.is_whitespace is False
    assert result.exact_match_count >= 10  # semaglutide + diabetes has many trials
    assert 400 < result.drug_only_trials < 800
    assert 10000 < result.indication_only_trials < 100000

    # indication_drugs should be empty when not whitespace
    assert result.indication_drugs == []


# --- Edge cases and weird inputs ---


async def test_get_trial_nonexistent_nct_id_raises_error(clinical_trials_client):
    """Test that a nonexistent NCT ID raises DataSourceError."""
    with pytest.raises(DataSourceError, match="404") as exc_info:
        await clinical_trials_client.get_trial("NCT99999999")

    assert "NCT99999999" in str(exc_info.value)


async def test_search_trials_nonexistent_drug_returns_empty(clinical_trials_client):
    """Test that a nonexistent drug returns empty list (not an error)."""
    trials = await clinical_trials_client.search_trials(
        drug="xyzzy_not_a_real_drug_12345",
        indication="diabetes",
        max_results=10,
    )

    assert trials == []


async def test_search_trials_nonexistent_indication_returns_empty(
    clinical_trials_client,
):
    """Test that a nonexistent indication returns empty list."""
    trials = await clinical_trials_client.search_trials(
        drug="semaglutide",
        indication="xyzzy_fake_disease_99999",
        max_results=10,
    )

    assert trials == []


async def test_search_trials_empty_drug_returns_empty(clinical_trials_client):
    """Test that empty drug string returns empty list."""
    trials = await clinical_trials_client.search_trials(
        drug="",
        indication="diabetes",
        max_results=10,
    )

    # Empty drug acts as no filter, so may return results
    # Just verify it doesn't raise an error
    assert isinstance(trials, list)


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


async def test_get_terminated_nonexistent_query_returns_empty(clinical_trials_client):
    """Test that nonexistent drug and indication returns empty list."""
    trials = await clinical_trials_client.get_terminated(
        "xyzzy_not_a_real_term_12345", "xyzzy_not_a_real_indication_12345", max_results=10
    )

    assert trials == []


async def test_get_terminated_business_trial_excluded_from_drug_query(
    clinical_trials_client,
):
    """Drug-side safety/efficacy filter excludes business-terminated trials end-to-end.

    NCT04012255 (semaglutide, Overweight, Phase 1) was terminated for strategic/business
    reasons. It must not appear in results: the drug query drops it (wrong stop_category),
    and the indication query ("type 2 diabetes") never includes an overweight trial.
    """
    trials = await clinical_trials_client.get_terminated(
        "semaglutide", "type 2 diabetes", max_results=30
    )

    nct_ids = {t.nct_id for t in trials}
    assert "NCT04012255" not in nct_ids


@pytest.mark.parametrize(
    "nct_id, why_stopped_fragment, expected_category",
    [
        (
            "NCT00109577",
            "no adverse events",  # negated safety phrase
            "other",
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
    from indication_scout.data_sources.clinical_trials import _classify_stop_reason

    trial = await clinical_trials_client.get_trial(nct_id)

    assert trial.why_stopped is not None
    assert why_stopped_fragment in trial.why_stopped.lower()
    assert _classify_stop_reason(trial.why_stopped) == expected_category
    assert _classify_stop_reason(trial.why_stopped) != "safety"


async def test_get_landscape_total_count_exceeds_fetch_cap(clinical_trials_client):
    """total_trial_count reflects the real API count, not the number of fetched trials.

    CLINICAL_TRIALS_LANDSCAPE_MAX_TRIALS caps the fetch at 50 trials, but the API
    reports the true total. For type 2 diabetes, thousands of trials exist.
    """
    from indication_scout.constants import CLINICAL_TRIALS_LANDSCAPE_MAX_TRIALS

    landscape = await clinical_trials_client.get_landscape("type 2 diabetes", top_n=5)

    assert landscape.total_trial_count > CLINICAL_TRIALS_LANDSCAPE_MAX_TRIALS


async def test_detect_whitespace_nonexistent_drug_is_whitespace(clinical_trials_client):
    """Test that nonexistent drug returns is_whitespace=True."""
    result = await clinical_trials_client.detect_whitespace(
        "xyzzy_fake_drug_99999", "diabetes"
    )

    assert result.is_whitespace is True
    assert result.exact_match_count == 0
    assert result.drug_only_trials == 0
    # Indication-only trials should still exist for a real indication
    assert result.indication_only_trials > 0


async def test_detect_whitespace_nonexistent_indication_is_whitespace(
    clinical_trials_client,
):
    """Test that nonexistent indication returns is_whitespace=True."""
    result = await clinical_trials_client.detect_whitespace(
        "semaglutide", "xyzzy_fake_disease_99999"
    )

    assert result.is_whitespace is True
    assert result.exact_match_count == 0
    # Drug-only trials should still exist for a real drug
    assert result.drug_only_trials > 0
    assert result.indication_only_trials == 0
    assert result.indication_drugs == []
