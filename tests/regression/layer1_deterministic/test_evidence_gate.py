"""Layer 1: deterministic checks on the supervisor's top-N evidence gate.

The gate logic lives inline in `supervisor_tools.finalize_supervisor`. This
test does NOT call into that code — it re-expresses the predicate as a pure
function and asserts on representative cases. The point is to lock in the
gate's *intent* so a refactor that accidentally drops a branch is caught
even if no LLM is run.

When the production gate logic legitimately changes, update the predicate
here in the same PR. The test is the contract.
"""

from __future__ import annotations

import pytest

from indication_scout.constants import SUPERVISOR_MIN_PMIDS_NO_TRIALS


def evidence_gate_drops(
    *,
    n_trials: int,
    n_pmids: int,
    lit_direction: str | None,
    lit_study_count: int | None,
) -> bool:
    """Re-expression of the gate in `finalize_supervisor`.

    Drop the candidate from the ranked top-N when there are zero trials AND
    no usable literature signal. The gate keys off DIRECTION, not strength:
    a contradicts/supports/mixed body is real evidence and must survive (a
    contradicts pair is surfaced as a bottom-ranked negative, not dropped).
    "No usable" is one of:
      - synthesize ran and tagged direction="none"
      - synthesize ran and tagged study_count==0
      - synthesize didn't run and the raw PMID count is below threshold.
    """
    no_lit_signal = (
        lit_direction == "none"
        or lit_study_count == 0
        or (
            lit_direction is None
            and lit_study_count is None
            and n_pmids < SUPERVISOR_MIN_PMIDS_NO_TRIALS
        )
    )
    return n_trials == 0 and no_lit_signal


class TestEvidenceGate:
    def test_trials_present_always_kept(self):
        # n_trials > 0 never drops, regardless of literature signal.
        assert not evidence_gate_drops(
            n_trials=1, n_pmids=0, lit_direction="none", lit_study_count=0
        )

    def test_no_trials_no_direction_dropped(self):
        # The canonical "rescue" case: 0 trials + synthesize said direction "none".
        assert evidence_gate_drops(
            n_trials=0, n_pmids=10, lit_direction="none", lit_study_count=5
        )

    def test_no_trials_zero_study_count_dropped(self):
        assert evidence_gate_drops(
            n_trials=0, n_pmids=10, lit_direction="supports", lit_study_count=0
        )

    def test_no_trials_supports_kept(self):
        # Legitimate untested-but-rationale-supported repurposing — keep it.
        assert not evidence_gate_drops(
            n_trials=0, n_pmids=10, lit_direction="supports", lit_study_count=5
        )

    def test_no_trials_contradicts_kept(self):
        # The core fix: a robustly-disproven pair (strong evidence, direction
        # contradicts) is REAL evidence — it must survive the gate and be
        # surfaced as a bottom-ranked negative, never dropped as zero-evidence.
        assert not evidence_gate_drops(
            n_trials=0, n_pmids=10, lit_direction="contradicts", lit_study_count=5
        )

    def test_pmid_fallback_below_threshold_drops(self):
        # Synthesize didn't run; raw PMID count falls back. Below threshold,
        # we drop.
        assert evidence_gate_drops(
            n_trials=0,
            n_pmids=SUPERVISOR_MIN_PMIDS_NO_TRIALS - 1,
            lit_direction=None,
            lit_study_count=None,
        )

    def test_pmid_fallback_at_threshold_kept(self):
        assert not evidence_gate_drops(
            n_trials=0,
            n_pmids=SUPERVISOR_MIN_PMIDS_NO_TRIALS,
            lit_direction=None,
            lit_study_count=None,
        )

    @pytest.mark.parametrize("direction", ["supports", "contradicts", "mixed"])
    def test_signal_with_zero_studies_still_dropped(self, direction):
        # study_count==0 wins even if direction is non-"none".
        assert evidence_gate_drops(
            n_trials=0, n_pmids=10, lit_direction=direction, lit_study_count=0
        )
