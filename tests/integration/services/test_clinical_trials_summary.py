"""Integration test for services/clinical_trials_summary.judge_ct_summary — the LIVE LLM
trial-section synthesis fed the already-resolved development stage.

Hits real Anthropic. Mirrors the crux shapes from scratch/ct_summary_harness.py:
- The recurring T1DM bug: stage="Phase 3 completed", a completed Phase 2/Phase 3 + active P3s.
  The fed-the-stage prose must NOT contradict the stage (no "no completed Phase 3" etc.), and
  closure must be "live".
- A relevant Phase 3 terminated for safety → closure "closed".
- An old generic with no approval / an operational stop → NOT closed on that alone.

Uses test_cache_dir so the real cache is untouched.
"""

import logging

from indication_scout.models.model_clinical_trials import Trial
from indication_scout.services.clinical_trials_summary import judge_ct_summary

logger = logging.getLogger(__name__)

# Phrases that re-judge the tier DOWN — a contradiction when the fed stage asserts a completed
# Phase 3. Mirrors the harness's contradiction set.
_CONTRADICTION_PHRASES = [
    "no completed phase 3",
    "no completed phase iii",
    "no dedicated phase 3",
    "no pivotal trial",
    "no pivotal phase 3",
    "no phase 3 program",
    "exploratory only",
    "phase 4 only",
    "no phase 3 on record",
    "lacks a completed phase 3",
    "without a completed phase 3",
]


def _t(nct, phase, status, title="", why_stopped=None):
    return Trial(
        nct_id=nct,
        phase=phase,
        overall_status=status,
        title=title,
        why_stopped=why_stopped,
    )


async def test_fed_stage_prose_does_not_contradict_completed_phase3(test_cache_dir):
    """The recurring T1DM bug: fed stage='Phase 3 completed' with a completed Phase 2/Phase 3
    + active P3s, the prose must NOT write a tier-contradiction and closure must be 'live'.
    """
    trials = [
        _t("NCT_A", "Phase 2", "COMPLETED", "Phase 2 study"),
        _t("NCT_B", "Phase 4", "COMPLETED", "Phase 4 post-approval study"),
        _t("NCT_C", "Phase 2/Phase 3", "COMPLETED", "Pivotal Phase 2/3"),
        _t("NCT_D", "Phase 3", "Recruiting", "Active Phase 3"),
        _t("NCT_E", "Phase 3", "Recruiting", "Active Phase 3"),
    ]
    s = await judge_ct_summary(
        trials,
        stage="Phase 3 completed for this indication",
        active_programs="2 Phase 3 recruiting (NCT_D, NCT_E)",
        first_approval=1923,  # old generic — no-approval must not read as closure
        cache_dir=test_cache_dir,
        drug="testdrug",
        indication="t1dm-shape",
    )
    assert s is not None
    low = s.prose.lower()
    hits = [p for p in _CONTRADICTION_PHRASES if p in low]
    assert not hits, f"prose contradicts the fed stage: {hits} in {s.prose!r}"
    assert (
        s.closure == "live"
    ), f"expected live, got {s.closure!r}: {s.closure_reason!r}"


async def test_relevant_phase3_terminated_for_safety_is_closed(test_cache_dir):
    """A relevant Phase 3 terminated for SAFETY → closure 'closed'."""
    trials = [
        _t(
            "NCT_A",
            "Phase 3",
            "Terminated",
            "Pivotal Phase 3",
            why_stopped="halted for serious adverse events",
        ),
        _t("NCT_B", "Phase 2", "COMPLETED", "Phase 2 study"),
    ]
    s = await judge_ct_summary(
        trials,
        stage="Phase 3 terminated for cause (safety/efficacy stop)",
        active_programs="None active",
        first_approval=2010,
        cache_dir=test_cache_dir,
        drug="testdrug",
        indication="safety-terminated",
    )
    assert s is not None
    assert (
        s.closure == "closed"
    ), f"expected closed, got {s.closure!r}: {s.closure_reason!r}"


async def test_old_generic_no_approval_is_not_closed(test_cache_dir):
    """An old generic with a completed Phase 2 and no approval must NOT be closed on
    no-approval alone — closure 'live' (or 'unknown'), never 'closed'."""
    trials = [_t("NCT_A", "Phase 2", "COMPLETED", "Phase 2 readout")]
    s = await judge_ct_summary(
        trials,
        stage="Phase 2 completed for this indication, no Phase 3",
        active_programs="None active",
        first_approval=1957,  # metformin-era
        cache_dir=test_cache_dir,
        drug="testdrug",
        indication="old-generic",
    )
    assert s is not None
    assert s.closure in {
        "live",
        "unknown",
    }, f"old generic wrongly closed: {s.closure_reason!r}"


async def test_operational_termination_is_not_closure(test_cache_dir):
    """A Phase 3 terminated for LOW ENROLLMENT (operational) is NOT closure — never 'closed'."""
    trials = [
        _t(
            "NCT_A",
            "Phase 3",
            "Terminated",
            "Phase 3",
            why_stopped="terminated due to low enrollment",
        ),
    ]
    s = await judge_ct_summary(
        trials,
        stage="Early-phase only, no completed pivotal readout",
        active_programs="None active",
        first_approval=2012,
        cache_dir=test_cache_dir,
        drug="testdrug",
        indication="operational-stop",
    )
    assert s is not None
    assert s.closure in {
        "live",
        "unknown",
    }, f"operational stop wrongly closed: {s.closure_reason!r}"


async def test_empty_trials_returns_none(test_cache_dir):
    """No trials → None (the caller leaves the summary empty)."""
    s = await judge_ct_summary(
        [],
        stage="No trials on record for this indication",
        active_programs="None active",
        first_approval=2000,
        cache_dir=test_cache_dir,
        drug="testdrug",
        indication="none",
    )
    assert s is None
