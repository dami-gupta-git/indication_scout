"""Unit tests for the hierarchical-dedup LLM helper.

These tests cover the post-LLM-response cleaning logic: idempotence guards,
unknown-name rejection, and failure-mode "keep everything" behavior. The LLM
itself is mocked.
"""

from unittest.mock import AsyncMock, patch

import pytest

from indication_scout.agents.supervisor.candidate_dedup import (
    HierarchyDecision,
    HierarchyDedupOutput,
    run_hierarchical_dedup,
)


CANDIDATES_UC_IBD = [
    ("inflammatory bowel disease", "competitor", "EFO_0003767"),
    ("ulcerative colitis", "mechanism", "EFO_0000729"),
    ("type 2 diabetes mellitus", "mechanism", "EFO_0001360"),
    ("diabetes mellitus", "mechanism", "EFO_0000400"),
]


@pytest.mark.asyncio
async def test_short_circuits_when_fewer_than_two_candidates():
    """Single-candidate list cannot have hierarchical overlap — skip LLM call."""
    with patch(
        "indication_scout.agents.supervisor.candidate_dedup.query_llm",
        new=AsyncMock(),
    ) as mock_llm:
        result = await run_hierarchical_dedup(
            drug_name="rosiglitazone",
            mechanism_targets=[("PPARG", "AGONIST")],
            candidates=[("type 2 diabetes mellitus", "mechanism", "EFO_0001360")],
        )
    assert result == HierarchyDedupOutput(decisions=[])
    assert mock_llm.call_count == 0


@pytest.mark.asyncio
async def test_well_formed_response_returns_cleaned_decisions():
    """Two valid overlap decisions are passed through with reasons preserved."""
    raw = (
        '{"decisions": ['
        '{"survivor": "ulcerative colitis", '
        '"dropped": ["inflammatory bowel disease"], '
        '"reason": "PPARG MoA most actionable in UC subtype"}, '
        '{"survivor": "type 2 diabetes mellitus", '
        '"dropped": ["diabetes mellitus"], '
        '"reason": "rosiglitazone approved for T2DM specifically"}'
        "]}"
    )
    with patch(
        "indication_scout.agents.supervisor.candidate_dedup.query_llm",
        new=AsyncMock(return_value=raw),
    ):
        result = await run_hierarchical_dedup(
            drug_name="rosiglitazone",
            mechanism_targets=[("PPARG", "AGONIST")],
            candidates=CANDIDATES_UC_IBD,
        )

    assert len(result.decisions) == 2
    d0 = result.decisions[0]
    assert d0.survivor == "ulcerative colitis"
    assert d0.dropped == ["inflammatory bowel disease"]
    assert d0.reason == "PPARG MoA most actionable in UC subtype"
    d1 = result.decisions[1]
    assert d1.survivor == "type 2 diabetes mellitus"
    assert d1.dropped == ["diabetes mellitus"]
    assert d1.reason == "rosiglitazone approved for T2DM specifically"


@pytest.mark.asyncio
async def test_unknown_survivor_decision_is_dropped():
    """A decision naming a survivor not in the candidate list is rejected."""
    raw = (
        '{"decisions": ['
        '{"survivor": "crohn disease", '
        '"dropped": ["inflammatory bowel disease"], "reason": "..."}, '
        '{"survivor": "type 2 diabetes mellitus", '
        '"dropped": ["diabetes mellitus"], "reason": "ok"}'
        "]}"
    )
    with patch(
        "indication_scout.agents.supervisor.candidate_dedup.query_llm",
        new=AsyncMock(return_value=raw),
    ):
        result = await run_hierarchical_dedup(
            drug_name="rosiglitazone",
            mechanism_targets=[],
            candidates=CANDIDATES_UC_IBD,
        )
    assert len(result.decisions) == 1
    assert result.decisions[0].survivor == "type 2 diabetes mellitus"


@pytest.mark.asyncio
async def test_unknown_names_inside_dropped_are_filtered_per_decision():
    """`dropped` entries not in the input candidate list get filtered out;
    the decision is kept only if at least one valid drop remains."""
    raw = (
        '{"decisions": ['
        '{"survivor": "ulcerative colitis", '
        '"dropped": ["inflammatory bowel disease", "crohn disease"], '
        '"reason": "valid + unknown — keep the valid drop"}, '
        '{"survivor": "type 2 diabetes mellitus", '
        '"dropped": ["unknown disease 1", "unknown disease 2"], '
        '"reason": "all unknown — drop the whole decision"}'
        "]}"
    )
    with patch(
        "indication_scout.agents.supervisor.candidate_dedup.query_llm",
        new=AsyncMock(return_value=raw),
    ):
        result = await run_hierarchical_dedup(
            drug_name="rosiglitazone",
            mechanism_targets=[],
            candidates=CANDIDATES_UC_IBD,
        )
    assert len(result.decisions) == 1
    assert result.decisions[0].survivor == "ulcerative colitis"
    assert result.decisions[0].dropped == ["inflammatory bowel disease"]


@pytest.mark.asyncio
async def test_survivor_appearing_in_own_dropped_list_is_filtered():
    """A self-referential `dropped` entry (survivor == dropped) is removed; if
    that leaves no valid drops the decision is rejected."""
    raw = (
        '{"decisions": ['
        '{"survivor": "ulcerative colitis", '
        '"dropped": ["ulcerative colitis"], "reason": "self-ref only"}'
        "]}"
    )
    with patch(
        "indication_scout.agents.supervisor.candidate_dedup.query_llm",
        new=AsyncMock(return_value=raw),
    ):
        result = await run_hierarchical_dedup(
            drug_name="rosiglitazone",
            mechanism_targets=[],
            candidates=CANDIDATES_UC_IBD,
        )
    assert result.decisions == []


@pytest.mark.asyncio
async def test_llm_call_failure_returns_empty_decisions():
    """An exception in the LLM call yields an empty HierarchyDedupOutput, not a raise."""
    with patch(
        "indication_scout.agents.supervisor.candidate_dedup.query_llm",
        new=AsyncMock(side_effect=RuntimeError("Anthropic 529")),
    ):
        result = await run_hierarchical_dedup(
            drug_name="rosiglitazone",
            mechanism_targets=[],
            candidates=CANDIDATES_UC_IBD,
        )
    assert result == HierarchyDedupOutput(decisions=[])


@pytest.mark.asyncio
async def test_unparseable_json_returns_empty_decisions():
    """Garbage in the LLM response yields empty decisions, with a log warning."""
    with patch(
        "indication_scout.agents.supervisor.candidate_dedup.query_llm",
        new=AsyncMock(return_value="this is not JSON at all"),
    ):
        result = await run_hierarchical_dedup(
            drug_name="rosiglitazone",
            mechanism_targets=[],
            candidates=CANDIDATES_UC_IBD,
        )
    assert result == HierarchyDedupOutput(decisions=[])


@pytest.mark.asyncio
async def test_missing_decisions_key_returns_empty():
    """JSON object missing the 'decisions' list is treated as failure."""
    with patch(
        "indication_scout.agents.supervisor.candidate_dedup.query_llm",
        new=AsyncMock(return_value='{"foo": "bar"}'),
    ):
        result = await run_hierarchical_dedup(
            drug_name="rosiglitazone",
            mechanism_targets=[],
            candidates=CANDIDATES_UC_IBD,
        )
    assert result == HierarchyDedupOutput(decisions=[])


@pytest.mark.asyncio
async def test_no_overlaps_returns_empty_decisions():
    """LLM returning empty decisions list (no overlaps found) is valid output."""
    with patch(
        "indication_scout.agents.supervisor.candidate_dedup.query_llm",
        new=AsyncMock(return_value='{"decisions": []}'),
    ):
        result = await run_hierarchical_dedup(
            drug_name="rosiglitazone",
            mechanism_targets=[],
            candidates=CANDIDATES_UC_IBD,
        )
    assert result == HierarchyDedupOutput(decisions=[])


def test_hierarchy_decision_coerce_nones():
    """coerce_nones validator replaces None with defaults for str/list fields."""
    d = HierarchyDecision(survivor=None, dropped=None, reason=None)
    assert d.survivor == ""
    assert d.dropped == []
    assert d.reason == ""


def test_hierarchy_dedup_output_coerce_nones():
    """coerce_nones validator replaces None decisions with empty list."""
    o = HierarchyDedupOutput(decisions=None)
    assert o.decisions == []
