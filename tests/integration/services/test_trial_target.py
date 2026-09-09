"""Integration test for the per-trial therapeutic-target gate — the LIVE LLM that decides whether a
registered trial set out to TREAT the candidate disease. Hits real ClinicalTrials.gov (so the fields
the gate reads are the ones the registry actually returns) and real Anthropic.

The three rejections are the sildenafil report's own errors, each a trial that studies sildenafil in
patients who have the candidate disease while aiming at something else:
  - NCT01168908 treats the cardiomyopathy of Duchenne muscular dystrophy, not the dystrophy.
  - NCT02136329 characterises pharmacokinetics, safety and tolerability in cardiac surgery; it was
    counted as an acute-kidney-injury programme.
  - NCT01441934 (SPHERIC-1) treats pulmonary hypertension arising in COPD patients, and pulmonary
    hypertension is an approved sildenafil indication; it set the COPD development stage to Phase 3.
NCT00104637 is the counterweight: sildenafil given for COPD itself must stay in, so the gate rejects
the aim rather than every trial whose patients carry a second diagnosis.
"""

import logging

import pytest

from indication_scout.services.trial_target import judge_trials_treat_disease

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "nct, disease, expected",
    [
        ("NCT01168908", "duchenne muscular dystrophy", False),
        ("NCT02136329", "acute kidney injury", False),
        ("NCT01441934", "chronic obstructive pulmonary disease", False),
        ("NCT00104637", "chronic obstructive pulmonary disease", True),
    ],
)
async def test_trial_target_gate_rejects_trials_aimed_elsewhere(
    clinical_trials_client, tmp_path, nct, disease, expected
):
    trial = await clinical_trials_client.get_trial(nct)
    assert trial.nct_id == nct
    assert trial.title

    verdicts = await judge_trials_treat_disease(
        "sildenafil", disease, [trial], tmp_path
    )

    assert verdicts == {nct: expected}
