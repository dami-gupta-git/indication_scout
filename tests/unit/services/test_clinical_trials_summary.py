"""Unit tests for services/clinical_trials_summary — trial-section prose parsing, typed-closure
validation, caching, and empty-set / parse-failure handling. The live LLM call is mocked; the
live judgment (fed-the-stage prose never contradicts the stage) is exercised in the integration
test.
"""

from unittest.mock import AsyncMock, patch

from indication_scout.models.model_clinical_trials import Trial
from indication_scout.services.clinical_trials_summary import (
    CTSummary,
    _parse_summary,
    judge_ct_summary,
)

_T = [Trial(nct_id="NCT1", phase="Phase 2/Phase 3", overall_status="COMPLETED")]

_OK = (
    '{"prose": "A completed Phase 2/Phase 3 pivotal trial (NCT1) is on record.", '
    '"closure": "live", "closure_reason": "no negative readout; trials ongoing"}'
)


def test_parse_summary_plain_json():
    s = _parse_summary(_OK)
    assert s == CTSummary(
        prose="A completed Phase 2/Phase 3 pivotal trial (NCT1) is on record.",
        closure="live",
        closure_reason="no negative readout; trials ongoing",
    )


def test_parse_summary_fenced_json():
    s = _parse_summary(
        '```json\n{"prose": "Phase 3 terminated for safety (NCT_A).", '
        '"closure": "closed", "closure_reason": "safety stop"}\n```'
    )
    assert s.prose == "Phase 3 terminated for safety (NCT_A)."
    assert s.closure == "closed"
    assert s.closure_reason == "safety stop"


def test_parse_summary_missing_reason_defaults_empty():
    s = _parse_summary('{"prose": "Some prose.", "closure": "unknown"}')
    assert s.prose == "Some prose."
    assert s.closure == "unknown"
    assert s.closure_reason == ""


def test_parse_summary_empty_prose_is_none():
    assert _parse_summary('{"prose": "   ", "closure": "live"}') is None


def test_parse_summary_unknown_closure_is_none():
    assert _parse_summary('{"prose": "x", "closure": "halted"}') is None


def test_parse_summary_garbage_is_none():
    assert _parse_summary("not json at all") is None


async def test_judge_empty_trials_returns_none_without_llm(tmp_path):
    """No trials → None, and the LLM is never called (caller leaves summary empty)."""
    with patch(
        "indication_scout.services.clinical_trials_summary.query_llm", new=AsyncMock()
    ) as mock_llm:
        s = await judge_ct_summary(
            [],
            stage="Phase 3 completed for this indication",
            active_programs="None active",
            first_approval=2000,
            cache_dir=tmp_path,
            drug="d",
            indication="i",
        )
    assert s is None
    mock_llm.assert_not_awaited()


async def test_judge_returns_parsed_summary(tmp_path):
    with patch(
        "indication_scout.services.clinical_trials_summary.query_llm",
        new=AsyncMock(return_value=_OK),
    ):
        s = await judge_ct_summary(
            _T,
            stage="Phase 3 completed for this indication",
            active_programs="None active",
            first_approval=1923,
            cache_dir=tmp_path,
            drug="d",
            indication="i",
        )
    assert s.prose == "A completed Phase 2/Phase 3 pivotal trial (NCT1) is on record."
    assert s.closure == "live"
    assert s.closure_reason == "no negative readout; trials ongoing"


async def test_judge_parse_failure_returns_none(tmp_path):
    """An unparseable response returns None (caller leaves summary empty — never fabricates)."""
    with patch(
        "indication_scout.services.clinical_trials_summary.query_llm",
        new=AsyncMock(return_value="the model rambled without JSON"),
    ):
        s = await judge_ct_summary(
            _T,
            stage="Phase 3 completed for this indication",
            active_programs="None active",
            first_approval=1923,
            cache_dir=tmp_path,
            drug="d",
            indication="i",
        )
    assert s is None


async def test_judge_caches_and_does_not_recall_llm(tmp_path):
    """Second call with the same fact-tuple hits the cache — the LLM is called once and all
    three fields round-trip."""
    mock = AsyncMock(return_value=_OK)
    with patch("indication_scout.services.clinical_trials_summary.query_llm", new=mock):
        first = await judge_ct_summary(
            _T,
            stage="Phase 3 completed for this indication",
            active_programs="None active",
            first_approval=1923,
            cache_dir=tmp_path,
            drug="d",
            indication="i",
        )
        second = await judge_ct_summary(
            _T,
            stage="Phase 3 completed for this indication",
            active_programs="None active",
            first_approval=1923,
            cache_dir=tmp_path,
            drug="d",
            indication="i",
        )
    assert first == second
    assert second.closure == "live"
    assert (
        second.prose == "A completed Phase 2/Phase 3 pivotal trial (NCT1) is on record."
    )
    assert mock.await_count == 1


async def test_judge_different_stage_is_separate_cache_entry(tmp_path):
    """The fed stage is part of the cache key — a different stage re-calls the LLM (the prose
    must be re-authored against the new ground truth)."""
    mock = AsyncMock(return_value=_OK)
    with patch("indication_scout.services.clinical_trials_summary.query_llm", new=mock):
        await judge_ct_summary(
            _T,
            stage="Phase 3 completed for this indication",
            active_programs="None active",
            first_approval=1923,
            cache_dir=tmp_path,
            drug="d",
            indication="i",
        )
        await judge_ct_summary(
            _T,
            stage="Active Phase 3 development on record for this indication",
            active_programs="None active",
            first_approval=1923,
            cache_dir=tmp_path,
            drug="d",
            indication="i",
        )
    assert mock.await_count == 2
