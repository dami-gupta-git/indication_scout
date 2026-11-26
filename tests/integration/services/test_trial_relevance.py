"""Integration test for services/trial_relevance — the per-trial relevance checks on real registry records and the real
model (PLAN_per_trial_relevance). Each case is a labelled drug x disease x trial with the check that decides it."""

import pytest

from indication_scout.data_sources.clinical_trials import ClinicalTrialsClient
from indication_scout.services.trial_relevance import classify_trials

SEMAGLUTIDE_APPROVED = [
    "type 2 diabetes mellitus",
    "major adverse cardiovascular events",
    "obesity",
    "MASH",
    "chronic kidney disease",
]


@pytest.mark.parametrize(
    "drug, disease, approved, nct, expected",
    [
        # Batch flips of 2026-11-25/26: excluded in some runs although the trial treats the candidate with the drug.
        ("metformin", "type 1 diabetes mellitus", ["type 2 diabetes mellitus"], "NCT04879511", "relevant"),
        ("bupropion", "cocaine dependence", ["major depressive disorder", "smoking cessation"], "NCT00227812", "relevant"),
        # Semaglutide is given only in the Klinefelter metabolic cohort; the fertility cohort gets other drugs.
        ("semaglutide", "infertility", SEMAGLUTIDE_APPROVED, "NCT05586802", "excluded"),
        # Drug role: metformin is the comparator arm; DPP studies metformin.
        ("metformin", "prediabetes syndrome", ["type 2 diabetes mellitus"], "NCT03968224", "excluded"),
        ("metformin", "prediabetes syndrome", ["type 2 diabetes mellitus"], "NCT00038727", "relevant"),
    ],
)
async def test_classify_trials_relevance_cases(drug, disease, approved, nct, expected, tmp_path):
    async with ClinicalTrialsClient(cache_dir=tmp_path) as client:
        trial = await client.get_trial(nct)

    result = await classify_trials(drug, disease, [trial], approved, tmp_path)

    assert result.verdicts == {nct: expected}


@pytest.mark.parametrize(
    "drug, disease, approved, nct, expected",
    [
        # Approved subtype: a MASH trial is already-approved evidence for semaglutide.
        ("semaglutide", "non-alcoholic fatty liver disease", SEMAGLUTIDE_APPROVED, "NCT04822181", "excluded"),
        # Distinct disease: a pulmonary arterial hypertension trial under systemic hypertension.
        ("sildenafil", "hypertensive disorder", ["erectile dysfunction", "pulmonary arterial hypertension"], "NCT00644605", "excluded"),
        # Combination with no arm isolating the drug (decided 2026-11-26).
        ("metformin", "duchenne muscular dystrophy", ["type 2 diabetes mellitus"], "NCT01995032", "combination"),
        ("bupropion", "binge eating disorder", ["major depressive disorder", "smoking cessation"], "NCT03539900", "combination"),
        ("bupropion", "binge eating disorder", ["major depressive disorder", "smoking cessation"], "NCT00414167", "relevant"),
    ],
)
async def test_classify_trials_subtype_disease_and_combination_cases(
    drug, disease, approved, nct, expected, tmp_path
):
    async with ClinicalTrialsClient(cache_dir=tmp_path) as client:
        trial = await client.get_trial(nct)

    result = await classify_trials(drug, disease, [trial], approved, tmp_path)

    assert result.verdicts == {nct: expected}
