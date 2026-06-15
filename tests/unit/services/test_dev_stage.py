"""Unit tests for services/dev_stage — LLM stage-judgment parsing, caching, and empty-set
handling. The live LLM call is mocked; the live judgment is exercised in the integration test.
"""

from unittest.mock import AsyncMock, patch

from indication_scout.models.model_clinical_trials import Trial
from indication_scout.services.dev_stage import (
    DEV_STAGE_TIERS,
    StageJudgment,
    _parse_judgment,
    judge_dev_stage,
)

_T = [Trial(nct_id="NCT1", phase="Phase 2/Phase 3", overall_status="COMPLETED")]

_OK = (
    '{"tier": "completed_phase3", "active_programs": "2 Phase 3 recruiting (NCT_X, NCT_Y)", '
    '"reason": "has a Phase 3 arm"}'
)


def test_parse_judgment_plain_json():
    j = _parse_judgment(_OK)
    assert j == StageJudgment(
        tier="completed_phase3", active_programs="2 Phase 3 recruiting (NCT_X, NCT_Y)"
    )


def test_parse_judgment_fenced_json():
    j = _parse_judgment(
        '```json\n{"tier": "active_phase3", "active_programs": "1 Phase 3 recruiting"}\n```'
    )
    assert j.tier == "active_phase3"
    assert j.active_programs == "1 Phase 3 recruiting"


def test_parse_judgment_missing_active_defaults_to_none_active():
    j = _parse_judgment('{"tier": "completed_phase2"}')
    assert j.tier == "completed_phase2"
    assert j.active_programs == "None active"


def test_parse_judgment_unknown_tier_is_none():
    assert _parse_judgment('{"tier": "phase_seven", "active_programs": "x"}') is None


def test_parse_judgment_garbage_is_none():
    assert _parse_judgment("not json at all") is None


async def test_judge_empty_trials_returns_floor_without_llm(tmp_path):
    """No trials → untested / None active, and the LLM is never called."""
    with patch(
        "indication_scout.services.dev_stage.query_llm", new=AsyncMock()
    ) as mock_llm:
        j = await judge_dev_stage([], tmp_path, drug="d", indication="i")
    assert j == StageJudgment(tier="untested", active_programs="None active")
    mock_llm.assert_not_awaited()


async def test_judge_returns_parsed_judgment(tmp_path):
    with patch(
        "indication_scout.services.dev_stage.query_llm",
        new=AsyncMock(return_value=_OK),
    ):
        j = await judge_dev_stage(_T, tmp_path, drug="d", indication="i")
    assert j.tier == "completed_phase3"
    assert j.tier in DEV_STAGE_TIERS
    assert j.active_programs == "2 Phase 3 recruiting (NCT_X, NCT_Y)"


async def test_judge_parse_failure_falls_back_to_floor(tmp_path):
    """An unparseable response defaults to the safe floor (never fabricates a higher tier)."""
    with patch(
        "indication_scout.services.dev_stage.query_llm",
        new=AsyncMock(return_value="the model rambled without JSON"),
    ):
        j = await judge_dev_stage(_T, tmp_path, drug="d", indication="i")
    assert j == StageJudgment(tier="untested", active_programs="None active")


async def test_judge_caches_and_does_not_recall_llm(tmp_path):
    """Second call with the same trial set hits the cache — the LLM is called once, and the
    cached judgment round-trips both fields."""
    mock = AsyncMock(return_value=_OK)
    with patch("indication_scout.services.dev_stage.query_llm", new=mock):
        first = await judge_dev_stage(_T, tmp_path, drug="d", indication="i")
        second = await judge_dev_stage(_T, tmp_path, drug="d", indication="i")
    assert first == second
    assert second.tier == "completed_phase3"
    assert second.active_programs == "2 Phase 3 recruiting (NCT_X, NCT_Y)"
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
