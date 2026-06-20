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
    SearchTrialsResult,
    TerminatedTrialsResult,
    Trial,
)

# Real SLE benefit:risk stop text — does NOT match a STOP_KEYWORDS keyword, so the helper
# must treat any non-operational Phase 3 termination as a cause-termination.
_BENEFIT_RISK_STOP = (
    "Study terminated due to insufficient evidence to support a positive "
    "benefit: risk profile in systemic lupus erythematosus patients."
)


def _ct(completed=None, terminated=None, search=None) -> ClinicalTrialsOutput:
    return ClinicalTrialsOutput(
        completed=CompletedTrialsResult(
            total_count=len(completed or []), trials=completed or []
        ),
        terminated=TerminatedTrialsResult(
            total_count=len(terminated or []), trials=terminated or []
        ),
        search=SearchTrialsResult(total_count=len(search or []), trials=search or []),
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


def test_completed_pure_phase3_excludes_combined_phase2_3():
    """has_completed_pure_phase3 is True only for a completed PURE Phase 3 (or Phase 3/Phase 4),
    NOT a completed combined 'Phase 2/Phase 3'. has_completed_phase3 stays True for both — the
    distinction lets the dev-stage judge avoid reading a completed Phase 2/3 as a pivotal Phase 3
    when the real Phase 3 is still recruiting (T1DM/semaglutide shape)."""
    # Only a completed Phase 2/Phase 3 → has_completed_phase3 True, but pure is False.
    ct_combined = _ct(completed=[Trial(nct_id="NCT05205928", phase="Phase 2/Phase 3")])
    sig_c = derive_trial_signals(ct_combined)
    assert sig_c.has_completed_phase3 is True
    assert sig_c.has_completed_pure_phase3 is False
    assert sig_c.completed_pure_phase3_nct_ids == []

    # A completed pure Phase 3 (and Phase 3/Phase 4) → pure True.
    ct_pure = _ct(
        completed=[
            Trial(nct_id="NCT00048360", phase="Phase 3"),
            Trial(nct_id="NCT00061087", phase="Phase 3/Phase 4"),
        ]
    )
    sig_p = derive_trial_signals(ct_pure)
    assert sig_p.has_completed_pure_phase3 is True
    assert sig_p.completed_pure_phase3_nct_ids == ["NCT00048360", "NCT00061087"]


def test_subphase_terminated_is_not_cause():
    """A terminated trial BELOW the pivotal band (e.g. Phase 2) must NOT trip the cause
    signal — only the pivotal 'Phase 2/Phase 3'..'Phase 3' band counts."""
    ct = _ct(
        completed=[Trial(nct_id="NCT11111111", phase="Phase 2")],
        terminated=[
            Trial(
                nct_id="NCT22222222",
                phase="Phase 2",
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


def test_phase4_blank_reason_is_not_cause():
    """Hepatic regression (NCT00736385): a Phase 4 trial with a BLANK stop reason must NOT
    set phase3_terminated_for_cause. Phase 4 is post-approval (excluded from the pivotal
    band) and a blank reason is not evidence of a safety/efficacy stop."""
    ct = _ct(
        terminated=[Trial(nct_id="NCT00736385", phase="Phase 4", why_stopped=None)],
    )
    sig = derive_trial_signals(ct)
    assert sig.phase3_terminated_for_cause is False
    assert sig.terminated_phase3_nct_ids == []


def test_phase4_safety_stop_is_not_phase3_cause():
    """A Phase 4 trial stopped for a real safety reason is excluded from this Phase-3 signal —
    the flag must stay honest about phase (Phase 4 is not a Phase 3 termination)."""
    ct = _ct(
        terminated=[
            Trial(
                nct_id="NCT99999999",
                phase="Phase 4",
                why_stopped="Terminated for safety concerns.",
            )
        ],
    )
    sig = derive_trial_signals(ct)
    assert sig.phase3_terminated_for_cause is False
    assert sig.terminated_phase3_nct_ids == []


def test_phase2_phase3_safety_stop_is_cause():
    """The pivotal band includes 'Phase 2/Phase 3': a safety stop there IS a cause-termination."""
    ct = _ct(
        terminated=[
            Trial(
                nct_id="NCT12340000",
                phase="Phase 2/Phase 3",
                why_stopped="Terminated for safety concerns.",
            )
        ],
    )
    sig = derive_trial_signals(ct)
    assert sig.phase3_terminated_for_cause is True
    assert sig.terminated_phase3_nct_ids == ["NCT12340000"]


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
    assert "active_phase_3: no" in block
    assert "relevant_phase3_terminated_for_cause: no" in block
    assert "closed_candidate" not in block


def test_active_phase3_from_recruiting_search_trial():
    """semaglutide × T1D shape: a recruiting Phase 3 in the all-status search set sets
    has_active_phase3, even with no completed Phase 3 on record."""
    ct = _ct(
        completed=[Trial(nct_id="NCT05537233", phase="Phase 2")],
        search=[
            Trial(nct_id="NCT06909006", phase="Phase 3", overall_status="Recruiting"),
            Trial(
                nct_id="NCT06082063",
                phase="Phase 3",
                overall_status="Active, not recruiting",
            ),
        ],
    )
    sig = derive_trial_signals(ct, relevant_nct_ids={"NCT05537233"})
    assert sig.has_completed_phase3 is False
    assert sig.has_active_phase3 is True
    assert sig.active_phase3_nct_ids == ["NCT06909006", "NCT06082063"]


def test_active_phase3_excludes_contaminated():
    """A recruiting Phase 3 flagged as contamination is dropped from the active signal."""
    ct = _ct(
        search=[
            Trial(nct_id="NCT_REL", phase="Phase 3", overall_status="Recruiting"),
            Trial(nct_id="NCT_CONTAM", phase="Phase 3", overall_status="Recruiting"),
        ],
    )
    sig = derive_trial_signals(ct, contaminated_nct_ids={"NCT_CONTAM"})
    assert sig.has_active_phase3 is True
    assert sig.active_phase3_nct_ids == ["NCT_REL"]


def test_active_phase23_counts_but_phase4_and_inactive_do_not():
    """Floor is >= Phase 2/Phase 3: a recruiting Phase 2/3 counts; a recruiting Phase 4,
    a recruiting Phase 1, and a completed Phase 3 (not active) do not."""
    ct = _ct(
        search=[
            Trial(
                nct_id="NCT_P23", phase="Phase 2/Phase 3", overall_status="Recruiting"
            ),
            Trial(
                nct_id="NCT_P34", phase="Phase 3/Phase 4", overall_status="Recruiting"
            ),
            Trial(nct_id="NCT_P4", phase="Phase 4", overall_status="Recruiting"),
            Trial(nct_id="NCT_P1", phase="Phase 1", overall_status="Recruiting"),
            Trial(nct_id="NCT_DONE", phase="Phase 3", overall_status="Completed"),
        ],
    )
    sig = derive_trial_signals(ct)
    assert sig.has_active_phase3 is True
    assert sig.active_phase3_nct_ids == ["NCT_P23", "NCT_P34"]


def test_no_active_phase3_when_search_empty():
    """No search trials -> has_active_phase3 stays False."""
    ct = _ct(completed=[Trial(nct_id="NCT1", phase="Phase 3")])
    sig = derive_trial_signals(ct)
    assert sig.has_active_phase3 is False
    assert sig.active_phase3_nct_ids == []


def test_format_block_renders_active_phase3_yes():
    ct = _ct(
        search=[Trial(nct_id="NCT_A", phase="Phase 3", overall_status="Recruiting")]
    )
    block = format_derived_signals(derive_trial_signals(ct))
    assert "active_phase_3: yes (NCT_A)" in block



# ------------------------------------------------------------------
# status-classification edge cases (E1-E4)
# ------------------------------------------------------------------


def test_not_yet_recruiting_phase3_is_active_via_exact_match():
    """'Not yet recruiting' is a planned program and counts as active — matched by exact set
    membership, not the old substring accident (RECRUITING in NOT_YET_RECRUITING)."""
    ct = _ct(
        search=[
            Trial(
                nct_id="NCT_NYR", phase="Phase 3", overall_status="Not yet recruiting"
            )
        ]
    )
    sig = derive_trial_signals(ct)
    assert sig.has_active_phase3 is True


def test_withdrawn_phase3_is_not_active():
    """A withdrawn Phase 3 never enrolled — it must NOT count as an active program."""
    ct = _ct(
        search=[Trial(nct_id="NCT_W", phase="Phase 3", overall_status="Withdrawn")]
    )
    sig = derive_trial_signals(ct)
    assert sig.has_active_phase3 is False


def test_suspended_phase3_counts_as_active():
    """A suspended trial is a pause (often resumes), so it counts as an ongoing program."""
    ct = _ct(
        search=[Trial(nct_id="NCT_S", phase="Phase 3", overall_status="Suspended")]
    )
    assert derive_trial_signals(ct).has_active_phase3 is True


def test_active_phase3_drops_classified_nonrelevant_trial():
    """E4: a Phase 3 the agent classified (it appears in completed/terminated) but judged NOT
    relevant must not be re-admitted via the unfiltered search read."""
    ct = _ct(
        completed=[
            Trial(nct_id="REL", phase="Phase 2"),
            Trial(nct_id="NOTREL", phase="Phase 3", overall_status="Recruiting"),
        ],
        search=[Trial(nct_id="NOTREL", phase="Phase 3", overall_status="Recruiting")],
    )
    sig = derive_trial_signals(ct, relevant_nct_ids={"REL"})
    assert sig.has_active_phase3 is False


def test_active_phase3_kept_when_relevant():
    """The E4 guard must not over-drop: a relevant active Phase 3 still counts."""
    ct = _ct(
        completed=[Trial(nct_id="REL2", phase="Phase 2")],
        search=[Trial(nct_id="RELP3", phase="Phase 3", overall_status="Recruiting")],
    )
    # RELP3 only in search (never classified into completed/terminated) → not excluded.
    sig = derive_trial_signals(ct, relevant_nct_ids={"REL2", "RELP3"})
    assert sig.has_active_phase3 is True
    assert sig.active_phase3_nct_ids == ["RELP3"]
