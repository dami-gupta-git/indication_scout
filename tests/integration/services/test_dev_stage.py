"""Integration test for services/dev_stage.judge_dev_stage — the LIVE LLM stage judgment.

Hits real Anthropic. Mirrors the crux cases from scratch/dev_stage_judgment_harness.py: the
Phase-4-ranks-above-Phase-3 trap, completed Phase 2/Phase 3, the recurring T1DM-shape (completed
P2/3 + active P3s + Phase 4), withdrawn/unknown status, and a pure Phase-4 portfolio. These are
the judgments that recurred as bugs when the stage was derived by deterministic phase-rank code.

Uses test_cache_dir so the real cache is untouched. Each case asserts the model returns the
correct tier; cases with a defensible second answer accept either.
"""

import logging

import pytest

from indication_scout.models.model_clinical_trials import Trial
from indication_scout.services.dev_stage import DEV_STAGE_TIERS, judge_dev_stage

logger = logging.getLogger(__name__)


def _t(nct, phase, status):
    return Trial(nct_id=nct, phase=phase, overall_status=status)


# (label, trials, accepted_tiers). Multiple accepted tiers where two readings are defensible
# from phase+status alone.
_CASES = [
    (
        "phase4_trap_with_completed_p2p3",  # the recurring T1DM bug
        [
            _t("NCT_A", "Phase 2", "COMPLETED"),
            _t("NCT_B", "Phase 4", "COMPLETED"),
            _t("NCT_C", "Phase 2/Phase 3", "COMPLETED"),
        ],
        {"completed_phase3"},
    ),
    (
        "phase4_plus_phase2_no_phase3",  # must NOT read Phase 4 as past-Phase-3
        [
            _t("NCT_A", "Phase 2", "COMPLETED"),
            _t("NCT_B", "Phase 4", "COMPLETED"),
        ],
        {"completed_phase2"},
    ),
    (
        "only_completed_phase4",
        [_t("NCT_A", "Phase 4", "COMPLETED")],
        {"exploratory_phase4_only"},
    ),
    (
        "large_mixed_t1dm_portfolio",  # completed P2/3 + active P3s + P4
        [
            _t("NCT_1", "Phase 2", "COMPLETED"),
            _t("NCT_2", "Phase 4", "COMPLETED"),
            _t("NCT_3", "Phase 2/Phase 3", "COMPLETED"),
            _t("NCT_4", "Phase 3", "Recruiting"),
            _t("NCT_5", "Phase 3", "Recruiting"),
            _t("NCT_6", "Phase 3", "Not yet recruiting"),
        ],
        {"completed_phase3"},  # completed wins over the additional active P3s
    ),
    (
        "withdrawn_phase3_never_ran",
        [_t("NCT_A", "Phase 3", "Withdrawn")],
        {"early_phase", "untested"},
    ),
    (
        "unknown_status_phase3",
        [_t("NCT_A", "Phase 3", "UNKNOWN")],
        {"phase3_unknown_status"},
    ),
    (
        "recruiting_phase3_none_completed",
        [
            _t("NCT_A", "Phase 3", "Recruiting"),
            _t("NCT_B", "Phase 2", "COMPLETED"),
        ],
        {"active_phase3", "completed_phase2"},
    ),
]


@pytest.mark.parametrize("label,trials,accepted", _CASES, ids=[c[0] for c in _CASES])
async def test_judge_dev_stage_live(label, trials, accepted, test_cache_dir):
    """The live model must return one of the accepted tiers for each crux case."""
    j = await judge_dev_stage(trials, test_cache_dir, drug="testdrug", indication=label)
    assert j.tier in DEV_STAGE_TIERS, f"{label}: returned non-tier {j.tier!r}"
    assert (
        j.tier in accepted
    ), f"{label}: model judged {j.tier!r}, expected one of {sorted(accepted)}"
    assert isinstance(j.active_programs, str) and j.active_programs


async def test_judge_active_programs_never_lists_a_completed_trial(test_cache_dir):
    """active_programs must describe what is MOVING — a portfolio of only COMPLETED trials
    must yield 'None active', never list a completed trial as an active program."""
    trials = [
        _t("NCT_A", "Phase 2/Phase 3", "COMPLETED"),
        _t("NCT_B", "Phase 4", "COMPLETED"),
    ]
    j = await judge_dev_stage(
        trials, test_cache_dir, drug="testdrug", indication="all-completed"
    )
    assert j.tier == "completed_phase3"
    assert (
        j.active_programs.strip().lower() == "none active"
    ), f"completed-only portfolio listed an active program: {j.active_programs!r}"


async def test_judge_active_programs_lists_recruiting_phase3(test_cache_dir):
    """A recruiting Phase 3 must appear in active_programs (the T1DM-shape the blurb LLM kept
    mis-filling with a completed trial)."""
    trials = [
        _t("NCT_DONE", "Phase 2/Phase 3", "COMPLETED"),
        _t("NCT06082063", "Phase 3", "Recruiting"),
        _t("NCT05819138", "Phase 3", "Recruiting"),
    ]
    j = await judge_dev_stage(
        trials, test_cache_dir, drug="testdrug", indication="t1dm-shape"
    )
    assert j.tier == "completed_phase3"
    ap = j.active_programs
    assert (
        "NCT06082063" in ap or "NCT05819138" in ap
    ), f"recruiting Phase 3 not surfaced in active_programs: {ap!r}"
    assert "NCT_DONE" not in ap, f"completed trial leaked into active_programs: {ap!r}"


async def test_judge_dev_stage_empty_is_floor(test_cache_dir):
    """No trials → untested / None active without an LLM call (cheap floor)."""
    j = await judge_dev_stage([], test_cache_dir, drug="d", indication="none")
    assert j.tier == "untested"
    assert j.active_programs == "None active"
