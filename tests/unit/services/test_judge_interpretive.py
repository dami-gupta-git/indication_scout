"""Unit tests for services/judge_interpretive — parsing, key-mapping, caching, and the
parse-failure floor. The live LLM call is mocked; live judgment is in the integration test.
"""

from unittest.mock import AsyncMock, patch

from indication_scout.services.judge_interpretive import (
    InterpretiveJudgment,
    _parse_interpretive,
    judge_interpretive,
)

_OK = (
    '{"constraint": "Regulatory differentiation burden", "key_risk": "May not beat '
    'approved family member", "assessment": "Live but bottlenecked", "prose": "Two '
    'sentences here. And a second one."}'
)

_FACTS = dict(
    stage="Phase 3 completed for this indication",
    active_programs="Phase 3 recruiting (NCT_A)",
    literature="Moderate, supports, RCT-backed",
    relationship="related_family",
    approved_indication="Type 2 Diabetes",
)


def test_parse_maps_json_keys_to_blurb_fields():
    j = _parse_interpretive(_OK)
    assert j == InterpretiveJudgment(
        blocker="Regulatory differentiation burden",
        key_risk="May not beat approved family member",
        verdict="Live but bottlenecked",
        prose="Two sentences here. And a second one.",
    )


def test_parse_fenced_json():
    j = _parse_interpretive(
        '```json\n{"constraint":"c","key_risk":"k","assessment":"a","prose":"p"}\n```'
    )
    assert (j.blocker, j.key_risk, j.verdict, j.prose) == ("c", "k", "a", "p")


def test_parse_missing_keys_default_empty():
    j = _parse_interpretive('{"constraint": "only this"}')
    assert j.blocker == "only this"
    assert j.key_risk == "" and j.verdict == "" and j.prose == ""


def test_parse_garbage_is_none():
    assert _parse_interpretive("not json at all") is None


def test_parse_non_object_is_none():
    assert _parse_interpretive('["a", "list"]') is None


async def test_judge_returns_parsed_judgment(tmp_path):
    with patch(
        "indication_scout.services.judge_interpretive.query_llm",
        new=AsyncMock(return_value=_OK),
    ):
        j = await judge_interpretive(
            **_FACTS, cache_dir=tmp_path, drug="d", indication="i"
        )
    assert j.verdict == "Live but bottlenecked"
    assert j.prose.startswith("Two sentences")


async def test_judge_parse_failure_returns_none(tmp_path):
    """A parse failure leaves the caller to keep fields empty — never fabricated."""
    with patch(
        "indication_scout.services.judge_interpretive.query_llm",
        new=AsyncMock(return_value="rambling, no json"),
    ):
        j = await judge_interpretive(
            **_FACTS, cache_dir=tmp_path, drug="d", indication="i"
        )
    assert j is None


async def test_judge_caches_and_does_not_recall_llm(tmp_path):
    mock = AsyncMock(return_value=_OK)
    with patch("indication_scout.services.judge_interpretive.query_llm", new=mock):
        first = await judge_interpretive(
            **_FACTS, cache_dir=tmp_path, drug="d", indication="i"
        )
        second = await judge_interpretive(
            **_FACTS, cache_dir=tmp_path, drug="d", indication="i"
        )
    assert first == second
    assert mock.await_count == 1


async def test_judge_cache_key_varies_by_facts(tmp_path):
    """A different stage is a different cache entry — the LLM is called again."""
    mock = AsyncMock(return_value=_OK)
    with patch("indication_scout.services.judge_interpretive.query_llm", new=mock):
        await judge_interpretive(**_FACTS, cache_dir=tmp_path, drug="d", indication="i")
        other = {
            **_FACTS,
            "stage": "Active Phase 3 development on record for this indication",
        }
        await judge_interpretive(**other, cache_dir=tmp_path, drug="d", indication="i")
    assert mock.await_count == 2


async def test_judge_none_approved_indication_ok(tmp_path):
    """approved_indication=None is accepted (rendered as 'none' in the prompt/cache key)."""
    facts = {**_FACTS, "approved_indication": None}
    with patch(
        "indication_scout.services.judge_interpretive.query_llm",
        new=AsyncMock(return_value=_OK),
    ):
        j = await judge_interpretive(
            **facts, cache_dir=tmp_path, drug="d", indication="i"
        )
    assert j is not None
