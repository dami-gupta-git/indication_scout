"""Integration test for services/trial_drug_isolation — real ClinicalTrials.gov records and the real model."""

from indication_scout.data_sources.clinical_trials import ClinicalTrialsClient
from indication_scout.services.trial_drug_isolation import find_combination_only_trials


async def test_bupropion_binge_eating_trials_split_combination_from_single_drug(tmp_path):
    """NCT03539900 gives a naltrexone/bupropion combination against placebo: combination-only. NCT00414167 gives
    bupropion against placebo: not flagged. Both checks agreed on these in the 2026-11-26 validation sample."""
    async with ClinicalTrialsClient(cache_dir=tmp_path) as client:
        trials = [
            await client.get_trial("NCT03539900"),
            await client.get_trial("NCT00414167"),
        ]

    assert await find_combination_only_trials("bupropion", trials, tmp_path) == {"NCT03539900"}
