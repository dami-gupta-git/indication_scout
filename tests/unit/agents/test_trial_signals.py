"""Unit tests for derive_trial_signals — deterministic trial FACTS (no verdict).

The helper computes facts only (highest completed phase, completed-Phase-3, a
relevant-Phase-3-terminated-for-cause signal). It does NOT decide closure. When a relevant
NCT set is supplied, facts are computed over RELEVANT trials only so a contaminating
Phase 3 pulled in by the recall-first search does not inflate the phase or trip the
cause-termination signal.
"""

from indication_scout.agents._trial_signals import (
    derive_trial_signals,
    format_derived_signals,
)
from indication_scout.agents.clinical_trials.clinical_trials_output import (
    ClinicalTrialsOutput,
)
from indication_scout.models.model_clinical_trials import (
    CompletedTrialsResult,
    TerminatedTrialsResult,
    Trial,
)

# Real SLE benefit:risk stop text — does NOT match a STOP_KEYWORDS keyword, so the helper
# must treat any non-operational Phase 3 termination as a cause-termination.
_BENEFIT_RISK_STOP = (
    "Study terminated due to insufficient evidence to support a positive "
    "benefit: risk profile in systemic lupus erythematosus patients."
)


def _ct(completed=None, terminated=None) -> ClinicalTrialsOutput:
    return ClinicalTrialsOutput(
        completed=CompletedTrialsResult(
            total_count=len(completed or []), trials=completed or []
        ),
        terminated=TerminatedTrialsResult(
            total_count=len(terminated or []), trials=terminated or []
        ),
    )


def test_benefit_risk_phase3_termination_is_cause():
    """Real SLE shape: completed Phase 3 + benefit:risk Phase 3 termination."""
    ct = _ct(
        completed=[
            Trial(nct_id="NCT03616964", phase="Phase 3"),
            Trial(nct_id="NCT02708095", phase="Phase 2"),
        ],
        terminated=[
            Trial(
                nct_id="NCT03616912", phase="Phase 3", why_stopped=_BENEFIT_RISK_STOP
            ),
        ],
    )
    sig = derive_trial_signals(ct)
    assert sig.highest_completed_phase == "Phase 3"
    assert sig.has_completed_phase3 is True
    assert sig.completed_phase3_nct_ids == ["NCT03616964"]
    assert sig.phase3_terminated_for_cause is True
    assert sig.terminated_phase3_nct_ids == ["NCT03616912"]


def test_completed_phase3_no_termination_has_no_cause_signal():
    """ADHD/bupropion shape: genuine completed Phase 3, no termination. No-approval is not
    a closure signal and the helper surfaces no cause-termination."""
    ct = _ct(
        completed=[
            Trial(nct_id="NCT00048360", phase="Phase 3"),
            Trial(nct_id="NCT00061087", phase="Phase 2/Phase 3"),
        ],
        terminated=[],
    )
    sig = derive_trial_signals(ct)
    assert sig.highest_completed_phase == "Phase 3"
    assert sig.has_completed_phase3 is True
    assert sig.phase3_terminated_for_cause is False
    assert sig.terminated_phase3_nct_ids == []


def test_phase2_phase3_terminated_is_not_cause():
    """A 'Phase 2/Phase 3' terminated trial must NOT trip the cause signal — only a true
    Phase 3."""
    ct = _ct(
        completed=[Trial(nct_id="NCT11111111", phase="Phase 2")],
        terminated=[
            Trial(
                nct_id="NCT22222222",
                phase="Phase 2/Phase 3",
                why_stopped="Study terminated: lack of efficacy.",
            )
        ],
    )
    sig = derive_trial_signals(ct)
    assert sig.phase3_terminated_for_cause is False
    assert sig.terminated_phase3_nct_ids == []


def test_phase3_terminated_for_enrollment_is_not_cause():
    """An operational (enrollment) Phase 3 stop is NOT a cause-termination."""
    ct = _ct(
        terminated=[
            Trial(
                nct_id="NCT00000001",
                phase="Phase 3",
                why_stopped="Study halted due to low enrollment and slow accrual.",
            )
        ],
    )
    sig = derive_trial_signals(ct)
    assert sig.phase3_terminated_for_cause is False


def test_relevance_filter_excludes_contaminating_phase3():
    """A contaminating completed Phase 3 (different disease) is excluded when a relevant set
    is supplied — phase reflects only the relevant Phase 2."""
    ct = _ct(
        completed=[
            Trial(nct_id="NCT_REL", phase="Phase 2"),
            Trial(nct_id="NCT_CONTAM", phase="Phase 3"),
        ],
    )
    # Without the filter: contaminating Phase 3 inflates the phase.
    unfiltered = derive_trial_signals(ct)
    assert unfiltered.highest_completed_phase == "Phase 3"
    assert unfiltered.has_completed_phase3 is True

    # With only NCT_REL relevant: the Phase 3 is dropped.
    filtered = derive_trial_signals(ct, relevant_nct_ids={"NCT_REL"})
    assert filtered.highest_completed_phase == "Phase 2"
    assert filtered.has_completed_phase3 is False
    assert filtered.completed_phase3_nct_ids == []


def test_relevance_filter_excludes_contaminating_termination():
    """A contaminating Phase 3 termination must not trip the cause signal once filtered out."""
    ct = _ct(
        terminated=[
            Trial(
                nct_id="NCT_CONTAM",
                phase="Phase 3",
                why_stopped="Terminated for safety concerns.",
            ),
        ],
    )
    unfiltered = derive_trial_signals(ct)
    assert unfiltered.phase3_terminated_for_cause is True

    filtered = derive_trial_signals(ct, relevant_nct_ids={"NCT_REL_OTHER"})
    assert filtered.phase3_terminated_for_cause is False
    assert filtered.terminated_phase3_nct_ids == []


def test_relevance_filter_keeps_relevant_termination():
    """A RELEVANT Phase 3 safety termination survives the filter (genuine closing signal)."""
    ct = _ct(
        terminated=[
            Trial(
                nct_id="NCT_SLE_1",
                phase="Phase 3",
                why_stopped=_BENEFIT_RISK_STOP,
            ),
            Trial(
                nct_id="NCT_SLE_2",
                phase="Phase 3",
                why_stopped="Terminated for a safety signal.",
            ),
        ],
    )
    sig = derive_trial_signals(ct, relevant_nct_ids={"NCT_SLE_1", "NCT_SLE_2"})
    assert sig.phase3_terminated_for_cause is True
    assert set(sig.terminated_phase3_nct_ids) == {"NCT_SLE_1", "NCT_SLE_2"}


def test_empty_relevant_set_filters_everything():
    """An explicit empty relevant set computes facts over zero trials (no inflation)."""
    ct = _ct(completed=[Trial(nct_id="NCT_X", phase="Phase 3")])
    sig = derive_trial_signals(ct, relevant_nct_ids=set())
    assert sig.highest_completed_phase is None
    assert sig.has_completed_phase3 is False


def test_none_artifact_returns_empty_signals():
    sig = derive_trial_signals(None)
    assert sig.highest_completed_phase is None
    assert sig.has_completed_phase3 is False
    assert sig.phase3_terminated_for_cause is False


def test_format_block_facts_only_no_closure_hint():
    """The rendered block is facts only — no closure verdict or 'closed_candidate' hint."""
    sig = derive_trial_signals(_ct(completed=[Trial(nct_id="NCT1", phase="Phase 2")]))
    block = format_derived_signals(sig)
    assert block.startswith("DERIVED SIGNALS (authoritative facts")
    assert "completed_phase_3: no" in block
    assert "relevant_phase3_terminated_for_cause: no" in block
    assert "closed_candidate" not in block
