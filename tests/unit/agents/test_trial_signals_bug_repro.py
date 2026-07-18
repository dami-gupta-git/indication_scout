"""Reproduces the original A1 bug in derive_trial_signals (findings.md, 2026-06-13/14).

Before the fix, the Phase-3-terminated-for-cause check used a `>=` floor on `_PHASE_RANK`
instead of the bounded pivotal band. Since Phase 4 (rank 8) ranks ABOVE Phase 3 (rank 6),
`rank >= Phase 3's rank` is satisfied by a Phase 4 trial too — so a terminated Phase 4 trial
was mislabeled "Phase 3 terminated for cause," a confidently-wrong closure signal.

Real case: NCT00736385 (metformin x NAFLD), a terminated Phase 4 trial with no stated
safety/efficacy reason (see gold_standard/metformin_2026-07-13_16-16-37.md — "stopped due
to poor enrollment, funding, and administrative challenges — neither for safety or
efficacy concerns").
"""

from indication_scout.agents._trial_formatting import _PHASE_RANK, _classify_stop_reason
from indication_scout.agents._trial_signals import derive_trial_signals
from indication_scout.agents.clinical_trials.clinical_trials_output import (
    ClinicalTrialsOutput,
)
from indication_scout.models.model_clinical_trials import (
    CompletedTrialsResult,
    SearchTrialsResult,
    TerminatedTrialsResult,
    Trial,
)

_CAUSE_STOP_CATEGORIES = {"safety", "efficacy"}


def _buggy_is_pivotal_phase3_floor(trial: Trial) -> bool:
    """The ORIGINAL (buggy) predicate: `>=` floor on Phase 2/Phase 3, no upper bound.

    This is what `_is_pivotal_phase3` looked like before it was bounded to
    `Phase 2/Phase 3 <= rank <= Phase 3` (see _trial_signals.py:69-77 for the fixed version).
    """
    rank = _PHASE_RANK.get(trial.phase or "", -1)
    return rank >= _PHASE_RANK["Phase 2/Phase 3"]


def test_buggy_floor_mislabels_terminated_phase4_as_phase3_cause():
    """Demonstrates the original defect in isolation: an unbounded `>=` floor treats a
    terminated Phase 4 trial as pivotal-Phase-3 band membership, purely because Phase 4
    outranks Phase 3 in _PHASE_RANK.
    """
    trial = Trial(
        nct_id="NCT00736385",
        phase="Phase 4",
        why_stopped="Terminated for safety concerns.",  # hypothetical safety stop
    )

    # The buggy predicate wrongly admits this Phase 4 trial into the "Phase 3 band".
    assert _buggy_is_pivotal_phase3_floor(trial) is True

    # Combined with a safety/efficacy stop reason, the original logic would have flagged
    # this as a Phase-3-terminated-for-cause closure signal — even though it's Phase 4.
    would_be_flagged_cause = (
        _buggy_is_pivotal_phase3_floor(trial)
        and _classify_stop_reason(trial.why_stopped) in _CAUSE_STOP_CATEGORIES
    )
    assert would_be_flagged_cause is True  # the original bug: Phase 4 read as "Phase 3 for cause"


def test_fixed_derive_trial_signals_does_not_reproduce_bug_on_real_case():
    """Same real trial (NCT00736385, Phase 4, actually terminated for non-cause reasons in
    the gold data) run through the CURRENT `derive_trial_signals` must NOT set
    phase3_terminated_for_cause — confirming the fix holds even if the stop reason had been
    a safety/efficacy phrase, because Phase 4 is excluded from the pivotal band outright.
    """
    ct = ClinicalTrialsOutput(
        completed=CompletedTrialsResult(total_count=0, trials=[]),
        terminated=TerminatedTrialsResult(
            total_count=1,
            trials=[
                Trial(
                    nct_id="NCT00736385",
                    phase="Phase 4",
                    why_stopped="Terminated for safety concerns.",
                )
            ],
        ),
        search=SearchTrialsResult(total_count=0, trials=[]),
    )
    sig = derive_trial_signals(ct)
    assert sig.phase3_terminated_for_cause is False
    assert sig.terminated_phase3_nct_ids == []
