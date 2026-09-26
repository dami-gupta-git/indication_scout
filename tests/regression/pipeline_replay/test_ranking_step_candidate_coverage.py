"""Opt-in live replay of the supervisor turn that drafts ranked blurbs.

The frozen ranking-step recording (`cassettes/semaglutide/ranking_step.yaml`) supplies one real
semaglutide request: system prompt, tool schemas, and the complete
conversation through investigation of 16 candidates. The test resends that request, and a copy
edited to what the current code sends (the finalize blurbs wording and the must-include list that
ends the investigation result), without running data sources, sub-agents, critique, finalization,
or report rendering.

Run explicitly because this test makes live Anthropic calls and measures stochastic behaviour:

    RUN_LIVE_RANKING_STEP_TEST=1 RANKING_STEP_REPEATS=5 \
        pytest -m regression \
        tests/regression/pipeline_replay/test_ranking_step_candidate_coverage.py
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections import Counter

import pytest
from anthropic import AsyncAnthropic

from scripts.replay_ranking_step import (
    _current_blurbs_desc,
    expected_candidates,
    load_ranking_requests,
    make_variant,
    one_call,
)

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.regression

_DRUG = "semaglutide"
_RECORDING_INDEX = 0
_EXPECTED_RANKED = [
    "type 1 diabetes mellitus",
    "metabolic dysfunction-associated steatohepatitis",
    "myocardial infarction",
    "parkinson disease",
    "stroke",
    "nicotine dependence",
    "heart failure",
    "eye disorder",
    "liver disorder",
    "coronary artery disorder",
    "polycystic ovary syndrome",
    "alzheimer disease",
    "alcohol abuse",
]
_EXPECTED_GATE_EXCLUDED = ["hypoglycemia", "schizophrenia"]
_EXPECTED_UNRESOLVED = ["inherited lipid metabolism disorder"]
_RECORDED_BLURBS_DESC = (
    "- blurbs: a list of structured per-candidate entries for the TOP 3\n"
    "  ranked candidates in your summary, in rank order."
)
_LIST_LINE = (
    "\n\nYour ranking (summary and blurbs) must include every one of these 13 candidates: "
    + ", ".join(_EXPECTED_RANKED)
    + "."
)


def _recorded_request() -> dict:
    requests = load_ranking_requests(_DRUG)
    assert len(requests) > _RECORDING_INDEX
    return requests[_RECORDING_INDEX]


def _finalize_description(request: dict) -> str:
    return next(
        tool["description"]
        for tool in request["tools"]
        if tool["name"] == "finalize_supervisor"
    )


def _investigation_result(request: dict) -> dict:
    return next(
        part
        for message in request["messages"]
        if isinstance(message["content"], list)
        for part in message["content"]
        if part.get("type") == "tool_result"
        and isinstance(part.get("content"), str)
        and part["content"].startswith("Auto-investigated")
    )


def _coverage(
    results: list[list[str] | None], expected: list[str]
) -> tuple[int, Counter[str]]:
    """Return run-level drop count and per-candidate inclusion count."""
    drop_runs = 0
    inclusion: Counter[str] = Counter()
    for blurbs in results:
        if blurbs is None:
            continue
        included = set(blurbs)
        inclusion.update(disease for disease in expected if disease in included)
        if any(disease not in included for disease in expected):
            drop_runs += 1
    return drop_runs, inclusion


def test_semaglutide_recording_and_current_variant_are_exact() -> None:
    """Pin the 13/2/1 partition and prove the current variant changes only the blurbs wording and the list line."""
    base = _recorded_request()
    ranked, gated, unresolved = expected_candidates(base)

    assert ranked == _EXPECTED_RANKED
    assert gated == _EXPECTED_GATE_EXCLUDED
    assert unresolved == _EXPECTED_UNRESOLVED

    recorded = make_variant(base, "recorded", ranked)
    current = make_variant(base, "current_with_list", ranked)
    recorded_description = _finalize_description(recorded)
    current_description = _finalize_description(current)

    assert _RECORDED_BLURBS_DESC in recorded_description
    assert _RECORDED_BLURBS_DESC not in current_description
    assert _current_blurbs_desc() in current_description
    assert not _investigation_result(recorded)["content"].endswith(_LIST_LINE)
    assert _investigation_result(current)["content"].endswith(_LIST_LINE)

    # Undo the two intended edits; every other request field must be identical.
    next(t for t in current["tools"] if t["name"] == "finalize_supervisor")["description"] = recorded_description
    current_result = _investigation_result(current)
    current_result["content"] = current_result["content"][: -len(_LIST_LINE)]
    assert current == recorded


@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_RANKING_STEP_TEST") != "1",
    reason="set RUN_LIVE_RANKING_STEP_TEST=1 to make live Anthropic replay calls",
)
async def test_current_request_keeps_every_rankable_semaglutide_candidate() -> None:
    """Compare the recorded request with what the current code sends; current must retain all 13 candidates every run."""
    repeats = int(os.environ.get("RANKING_STEP_REPEATS", "5"))
    assert repeats > 0
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    assert api_key, "ANTHROPIC_API_KEY is required for the live ranking-step replay"

    base = _recorded_request()
    expected, gated, unresolved = expected_candidates(base)
    assert expected == _EXPECTED_RANKED
    assert gated == _EXPECTED_GATE_EXCLUDED
    assert unresolved == _EXPECTED_UNRESOLVED

    client = AsyncAnthropic(api_key=api_key)
    results_by_variant: dict[str, list[list[str] | None]] = {}
    for variant in ("recorded", "current_with_list"):
        request = make_variant(base, variant, expected)
        calls = await asyncio.gather(*(one_call(client, request) for _ in range(repeats)))
        results_by_variant[variant] = [blurbs for blurbs, _text_lines in calls]

    stats: dict[str, tuple[int, Counter[str]]] = {}
    for variant, results in results_by_variant.items():
        non_critique_runs = sum(result is None for result in results)
        drop_runs, inclusion = _coverage(results, expected)
        stats[variant] = (drop_runs, inclusion)
        logger.info(
            "%s: runs=%d, non-critique=%d, runs-with-drop=%d, inclusion=%s",
            variant,
            repeats,
            non_critique_runs,
            drop_runs,
            {disease: inclusion[disease] for disease in expected},
        )
        assert non_critique_runs == 0, (
            f"{variant}: {non_critique_runs}/{repeats} responses did not call "
            "critique_ranking"
        )

    recorded_drops, _ = stats["recorded"]
    current_drops, current_inclusion = stats["current_with_list"]
    assert current_drops == 0, (
        f"current request dropped at least one eligible candidate in "
        f"{current_drops}/{repeats} runs; recorded request dropped in "
        f"{recorded_drops}/{repeats}"
    )
    assert current_inclusion == Counter(dict.fromkeys(expected, repeats))
