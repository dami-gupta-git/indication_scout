"""Unit tests for services/dev_stage — LLM stage-judgment parsing, caching, and empty-set
handling. The live LLM call is mocked; the live judgment is exercised in the integration test.
"""

import re
from unittest.mock import AsyncMock, patch

from indication_scout.models.model_clinical_trials import Trial
from indication_scout.services.dev_stage import (
    DEV_STAGE_TIERS,
    StageJudgment,
    _enforce_tier_floor,
    _has_active_phase3_band,
    _has_completed_phase3_band,
    _parse_tier,
    _render_active_programs,
    judge_dev_stage,
)

# A completed Phase 2/3 trial: tier completed_phase3, but NOT an active trial → "None active".
_T = [Trial(nct_id="NCT1", phase="Phase 2/Phase 3", overall_status="COMPLETED")]

_OK = '{"tier": "completed_phase3", "reason": "has a Phase 3 arm"}'


def test_parse_tier_plain_json():
    assert _parse_tier(_OK) == "completed_phase3"


def test_parse_tier_fenced_json():
    assert _parse_tier('```json\n{"tier": "active_phase3"}\n```') == "active_phase3"


def test_parse_tier_unknown_tier_is_none():
    assert _parse_tier('{"tier": "phase_seven"}') is None


def test_parse_tier_garbage_is_none():
    assert _parse_tier("not json at all") is None


async def test_judge_empty_trials_returns_floor_without_llm(tmp_path):
    """No trials → untested / None active, and the LLM is never called."""
    with patch(
        "indication_scout.services.dev_stage.query_llm", new=AsyncMock()
    ) as mock_llm:
        j = await judge_dev_stage([], tmp_path, drug="d", indication="i")
    assert j == StageJudgment(tier="untested", active_programs="None active")
    mock_llm.assert_not_awaited()


async def test_judge_returns_tier_with_deterministic_active_programs(tmp_path):
    """The LLM supplies the tier; active_programs is rendered deterministically (here the only
    trial is COMPLETED, so nothing is active)."""
    with patch(
        "indication_scout.services.dev_stage.query_llm",
        new=AsyncMock(return_value=_OK),
    ):
        j = await judge_dev_stage(_T, tmp_path, drug="d", indication="i")
    assert j.tier == "completed_phase3"
    assert j.tier in DEV_STAGE_TIERS
    assert j.active_programs == "None active"


async def test_judge_active_programs_lists_recruiting_phase3_deterministically(tmp_path):
    """active_programs is rendered from the trials, not the LLM — a recruiting Phase 3 is named
    with a count equal to the listed ids (the semaglutide miscount fix)."""
    trials = [
        Trial(nct_id="NCTD", phase="Phase 2/Phase 3", overall_status="COMPLETED"),
        Trial(nct_id="NCT1", phase="Phase 3", overall_status="Recruiting"),
        Trial(nct_id="NCT2", phase="Phase 3", overall_status="Recruiting"),
    ]
    with patch(
        "indication_scout.services.dev_stage.query_llm",
        new=AsyncMock(return_value='{"tier": "active_phase3", "reason": "x"}'),
    ):
        j = await judge_dev_stage(trials, tmp_path, drug="d", indication="i")
    assert j.active_programs == "2 Phase 3 active/recruiting (NCT1, NCT2)"
    assert "NCTD" not in j.active_programs  # completed trial never listed as active


async def test_judge_semaglutide_t1d_count_is_four_not_five(tmp_path):
    """Regression for the semaglutide T1D miscount: the LLM said '5 Phase 3' but listed 4 NCTs.
    judge_dev_stage now renders active_programs deterministically — exactly the 4 recruiting pure
    Phase 3 NCTs, count == ids, with the Phase 2/3 trials reported separately (not folded into 5).
    This is the mid-pipeline hop (real trials in, propagated StageJudgment out)."""
    # Statuses use the real CT.gov underscored forms (NOT_YET_RECRUITING etc).
    trials = [
        Trial(nct_id="NCT06082063", phase="Phase 3", overall_status="RECRUITING"),
        Trial(nct_id="NCT06909006", phase="Phase 3", overall_status="NOT_YET_RECRUITING"),
        Trial(nct_id="NCT05819138", phase="Phase 3", overall_status="RECRUITING"),
        Trial(nct_id="NCT06894784", phase="Phase 3", overall_status="RECRUITING"),
        Trial(nct_id="NCT03899402", phase="Phase 2/Phase 3", overall_status="ACTIVE_NOT_RECRUITING"),
        Trial(nct_id="NCT06387199", phase="Phase 2/Phase 3", overall_status="RECRUITING"),
        Trial(nct_id="NCT05537233", phase="Phase 2", overall_status="COMPLETED"),
    ]
    with patch(
        "indication_scout.services.dev_stage.query_llm",
        new=AsyncMock(return_value='{"tier": "active_phase3", "reason": "x"}'),
    ):
        j = await judge_dev_stage(trials, tmp_path, drug="semaglutide", indication="t1d")
    # The pure-Phase-3 group count must be 4 and list exactly the 4 pure Phase 3 NCTs.
    assert "4 Phase 3 active/recruiting" in j.active_programs
    assert "5 Phase 3" not in j.active_programs
    pure3 = {"NCT06082063", "NCT06909006", "NCT05819138", "NCT06894784"}
    listed_in_pure_group = set(
        re.findall(r"4 Phase 3 active/recruiting \(([^)]*)\)", j.active_programs)[0].split(", ")
    )
    assert listed_in_pure_group == pure3


async def test_judge_parse_failure_falls_back_to_floor(tmp_path):
    """An unparseable response defaults to untested, then the deterministic tier-floor lifts it
    to 'early_phase' because trials are on record (a completed Phase 2/3 is not pure Phase 3)."""
    with patch(
        "indication_scout.services.dev_stage.query_llm",
        new=AsyncMock(return_value="the model rambled without JSON"),
    ):
        j = await judge_dev_stage(_T, tmp_path, drug="d", indication="i")
    assert j == StageJudgment(tier="early_phase", active_programs="None active")


async def test_judge_caches_tier_and_does_not_recall_llm(tmp_path):
    """Second call with the same trial set hits the cache — the LLM is called once; the tier
    round-trips and active_programs is re-rendered."""
    mock = AsyncMock(return_value=_OK)
    with patch("indication_scout.services.dev_stage.query_llm", new=mock):
        first = await judge_dev_stage(_T, tmp_path, drug="d", indication="i")
        second = await judge_dev_stage(_T, tmp_path, drug="d", indication="i")
    assert first == second
    assert second.tier == "completed_phase3"
    assert mock.await_count == 1


async def test_judge_cache_key_is_order_independent(tmp_path):
    """Two orderings of the same trial set share one cache entry."""
    a = Trial(nct_id="NCT_A", phase="Phase 3", overall_status="COMPLETED")
    b = Trial(nct_id="NCT_B", phase="Phase 2", overall_status="COMPLETED")
    mock = AsyncMock(return_value=_OK)
    with patch("indication_scout.services.dev_stage.query_llm", new=mock):
        await judge_dev_stage([a, b], tmp_path, drug="d", indication="i")
        await judge_dev_stage([b, a], tmp_path, drug="d", indication="i")
    assert mock.await_count == 1


# --- deterministic tier-floor invariants (pure functions, no LLM) ---


def test_has_completed_phase3_band_pure_phase3():
    trials = [Trial(nct_id="N", phase="Phase 3", overall_status="COMPLETED")]
    assert _has_completed_phase3_band(trials) is True


def test_has_completed_phase3_band_phase3_phase4():
    trials = [Trial(nct_id="N", phase="Phase 3/Phase 4", overall_status="Completed")]
    assert _has_completed_phase3_band(trials) is True


def test_has_completed_phase3_band_excludes_phase2_phase3():
    """The combined Phase 2/Phase 3 designation is NOT an unambiguous completed Phase 3 — its
    resolution stays the LLM's call (active-pure-Phase-3 rule)."""
    trials = [Trial(nct_id="N", phase="Phase 2/Phase 3", overall_status="COMPLETED")]
    assert _has_completed_phase3_band(trials) is False


def test_has_completed_phase3_band_excludes_non_completed_phase3():
    trials = [Trial(nct_id="N", phase="Phase 3", overall_status="Recruiting")]
    assert _has_completed_phase3_band(trials) is False


def test_floor_untested_with_trials_becomes_early_phase():
    """Trials on record can never read 'untested' (the Mood Disorder / Not-Applicable bug)."""
    trials = [Trial(nct_id="N", phase="Not Applicable", overall_status="COMPLETED")]
    assert _enforce_tier_floor("untested", trials) == "early_phase"


def test_floor_untested_with_no_trials_stays_untested():
    assert _enforce_tier_floor("untested", []) == "untested"


def test_floor_completed_phase3_band_lifts_untested():
    """A completed pure Phase 3 in the set lifts a wrong 'untested' to completed_phase3
    (the imatinib x Leukemia bug — LLM returned untested over completed Phase 3 ALL trials)."""
    trials = [
        Trial(nct_id="N1", phase="Phase 3", overall_status="COMPLETED"),
        Trial(nct_id="N2", phase="Phase 2", overall_status="COMPLETED"),
    ]
    assert _enforce_tier_floor("untested", trials) == "completed_phase3"


def test_floor_completed_phase3_band_lifts_early_phase():
    trials = [Trial(nct_id="N1", phase="Phase 3", overall_status="COMPLETED")]
    assert _enforce_tier_floor("early_phase", trials) == "completed_phase3"


def test_floor_does_not_override_terminated_for_cause():
    """The terminated-for-cause closure signal is a deliberate LLM judgment — not overridden
    even when a completed Phase 3 also exists."""
    trials = [
        Trial(nct_id="N1", phase="Phase 3", overall_status="COMPLETED"),
        Trial(nct_id="N2", phase="Phase 3", overall_status="Terminated"),
    ]
    assert (
        _enforce_tier_floor("phase3_terminated_for_cause", trials)
        == "phase3_terminated_for_cause"
    )


def test_floor_does_not_lower_a_higher_tier():
    """When no completed pure Phase 3 exists, a valid higher tier is left untouched."""
    trials = [Trial(nct_id="N", phase="Phase 3", overall_status="Recruiting")]
    assert _enforce_tier_floor("active_phase3", trials) == "active_phase3"


def test_has_active_phase3_band_recruiting():
    trials = [Trial(nct_id="N", phase="Phase 3", overall_status="Recruiting")]
    assert _has_active_phase3_band(trials) is True


def test_has_active_phase3_band_excludes_completed():
    trials = [Trial(nct_id="N", phase="Phase 3", overall_status="COMPLETED")]
    assert _has_active_phase3_band(trials) is False


def test_floor_active_phase3_lifts_early_phase():
    """The semaglutide × T1D regression: LLM judged early_phase over 4 recruiting Phase 3 trials.
    The floor lifts it to active_phase3."""
    trials = [
        Trial(nct_id="N1", phase="Phase 3", overall_status="Recruiting"),
        Trial(nct_id="N2", phase="Phase 3", overall_status="Not yet recruiting"),
        Trial(nct_id="N3", phase="Phase 2", overall_status="COMPLETED"),
    ]
    assert _enforce_tier_floor("early_phase", trials) == "active_phase3"


def test_floor_completed_phase3_wins_over_active():
    """A completed pure Phase 3 outranks active ones — floor lands on completed_phase3."""
    trials = [
        Trial(nct_id="N1", phase="Phase 3", overall_status="COMPLETED"),
        Trial(nct_id="N2", phase="Phase 3", overall_status="Recruiting"),
    ]
    assert _enforce_tier_floor("early_phase", trials) == "completed_phase3"


# --- deterministic active_programs rendering (counts must equal listed ids) ---


def test_render_active_programs_count_equals_listed_phase3_ids():
    """The semaglutide miscount: count MUST equal the number of NCTs listed. 4 recruiting pure
    Phase 3 → '4 Phase 3 active/recruiting (4 ids)', completed/Phase-2 excluded."""
    trials = [
        Trial(nct_id="NCT06082063", phase="Phase 3", overall_status="Recruiting"),
        Trial(nct_id="NCT06909006", phase="Phase 3", overall_status="Not yet recruiting"),
        Trial(nct_id="NCT05819138", phase="Phase 3", overall_status="Recruiting"),
        Trial(nct_id="NCT06894784", phase="Phase 3", overall_status="Recruiting"),
        Trial(nct_id="NCT05537233", phase="Phase 2", overall_status="COMPLETED"),
        Trial(nct_id="NCT05205928", phase="Phase 2/Phase 3", overall_status="COMPLETED"),
    ]
    line = _render_active_programs(trials)
    listed = set(re.findall(r"NCT\d+", line))
    assert line.startswith("4 Phase 3 active/recruiting")
    assert len(listed) == 4
    assert "NCT05537233" not in listed and "NCT05205928" not in listed


def test_render_active_programs_splits_pure_and_phase2_phase3_counts():
    """Pure Phase 3 and active Phase 2/3 are reported as separate groups, each count == its ids."""
    trials = [
        Trial(nct_id="NCT1", phase="Phase 3", overall_status="Recruiting"),
        Trial(nct_id="NCT2", phase="Phase 2/Phase 3", overall_status="Active, not recruiting"),
        Trial(nct_id="NCT3", phase="Phase 2/Phase 3", overall_status="Recruiting"),
    ]
    line = _render_active_programs(trials)
    assert "1 Phase 3 active/recruiting (NCT1)" in line
    assert "2 Phase 2/Phase 3 active (NCT2, NCT3)" in line


def test_render_active_programs_non_pivotal_only():
    """No active pivotal trial, but earlier-phase activity is moving → disclosed, not hidden."""
    trials = [
        Trial(nct_id="NCT1", phase="Phase 1", overall_status="Recruiting"),
        Trial(nct_id="NCT2", phase="Phase 3", overall_status="COMPLETED"),
    ]
    line = _render_active_programs(trials)
    assert line == "No pivotal program active; 1 non-pivotal active (NCT1)"


def test_render_active_programs_handles_ctgov_underscored_status():
    """CT.gov returns NOT_YET_RECRUITING / ACTIVE_NOT_RECRUITING (underscored, uppercased) — the
    semaglutide NCT06909006 bug where a not-yet-recruiting Phase 3 was dropped from the count."""
    trials = [
        Trial(nct_id="NCT1", phase="Phase 3", overall_status="RECRUITING"),
        Trial(nct_id="NCT2", phase="Phase 3", overall_status="NOT_YET_RECRUITING"),
        Trial(nct_id="NCT3", phase="Phase 3", overall_status="ACTIVE_NOT_RECRUITING"),
    ]
    line = _render_active_programs(trials)
    assert line == "3 Phase 3 active/recruiting (NCT1, NCT2, NCT3)"


def test_render_active_programs_none_active():
    """All completed/terminated → None active."""
    trials = [
        Trial(nct_id="NCT1", phase="Phase 3", overall_status="COMPLETED"),
        Trial(nct_id="NCT2", phase="Phase 2", overall_status="Terminated"),
    ]
    assert _render_active_programs(trials) == "None active"
