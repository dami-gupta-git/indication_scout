"""Unit tests for services/approval_check — no network, no LLM calls."""

import json
from datetime import date
from unittest.mock import AsyncMock, patch

import pytest

from indication_scout.services.approval_check import (
    _load_drug_approvals_table,
    extract_approved_from_labels,
    get_approved_indications,
    list_approved_indications_at,
    list_approved_indications_from_labels,
)

# --- extract_approved_from_labels ---


async def test_extract_approved_returns_matched_candidates(tmp_path):
    llm_response = json.dumps(["obesity", "type 2 diabetes mellitus"])

    with patch(
        "indication_scout.services.approval_check.query_small_llm",
        new=AsyncMock(return_value=llm_response),
    ):
        result = await extract_approved_from_labels(
            label_texts=["Indicated for obesity and type 2 diabetes."],
            candidate_diseases=["obesity", "type 2 diabetes mellitus", "heart failure"],
            cache_dir=tmp_path,
        )

    assert result == {"obesity", "type 2 diabetes mellitus"}


async def test_extract_approved_returns_empty_on_no_matches(tmp_path):
    llm_response = json.dumps([])

    with patch(
        "indication_scout.services.approval_check.query_small_llm",
        new=AsyncMock(return_value=llm_response),
    ):
        result = await extract_approved_from_labels(
            label_texts=["Indicated for condition X."],
            candidate_diseases=["obesity", "heart failure"],
            cache_dir=tmp_path,
        )

    assert result == set()


async def test_extract_approved_empty_label_texts_no_llm_call(tmp_path):
    mock_llm = AsyncMock()
    with patch(
        "indication_scout.services.approval_check.query_small_llm",
        new=mock_llm,
    ):
        result = await extract_approved_from_labels(
            label_texts=[],
            candidate_diseases=["obesity"],
            cache_dir=tmp_path,
        )

    assert result == set()
    mock_llm.assert_not_awaited()


async def test_extract_approved_empty_candidates_no_llm_call(tmp_path):
    mock_llm = AsyncMock()
    with patch(
        "indication_scout.services.approval_check.query_small_llm",
        new=mock_llm,
    ):
        result = await extract_approved_from_labels(
            label_texts=["Some label text."],
            candidate_diseases=[],
            cache_dir=tmp_path,
        )

    assert result == set()
    mock_llm.assert_not_awaited()


async def test_extract_approved_strips_markdown_fences(tmp_path):
    llm_response = '```json\n["obesity"]\n```'

    with patch(
        "indication_scout.services.approval_check.query_small_llm",
        new=AsyncMock(return_value=llm_response),
    ):
        result = await extract_approved_from_labels(
            label_texts=["Indicated for obesity."],
            candidate_diseases=["obesity", "heart failure"],
            cache_dir=tmp_path,
        )

    assert result == {"obesity"}


async def test_extract_approved_rejects_hallucinated_disease(tmp_path):
    llm_response = json.dumps(["obesity", "made up disease"])

    with patch(
        "indication_scout.services.approval_check.query_small_llm",
        new=AsyncMock(return_value=llm_response),
    ):
        result = await extract_approved_from_labels(
            label_texts=["Indicated for obesity."],
            candidate_diseases=["obesity", "heart failure"],
            cache_dir=tmp_path,
        )

    assert result == {"obesity"}


async def test_extract_approved_invalid_json_returns_empty(tmp_path):
    llm_response = "This is not valid JSON at all"

    with patch(
        "indication_scout.services.approval_check.query_small_llm",
        new=AsyncMock(return_value=llm_response),
    ):
        result = await extract_approved_from_labels(
            label_texts=["Indicated for obesity."],
            candidate_diseases=["obesity"],
            cache_dir=tmp_path,
        )

    assert result == set()


async def test_extract_approved_case_insensitive_matching(tmp_path):
    llm_response = json.dumps(["Obesity"])

    with patch(
        "indication_scout.services.approval_check.query_small_llm",
        new=AsyncMock(return_value=llm_response),
    ):
        result = await extract_approved_from_labels(
            label_texts=["Indicated for obesity."],
            candidate_diseases=["obesity", "heart failure"],
            cache_dir=tmp_path,
        )

    assert result == {"obesity"}


async def test_extract_approved_caches_result(tmp_path):
    llm_response = json.dumps(["obesity"])
    mock_llm = AsyncMock(return_value=llm_response)

    with patch(
        "indication_scout.services.approval_check.query_small_llm",
        new=mock_llm,
    ):
        first = await extract_approved_from_labels(
            label_texts=["Indicated for obesity."],
            candidate_diseases=["obesity"],
            cache_dir=tmp_path,
        )
        second = await extract_approved_from_labels(
            label_texts=["Indicated for obesity."],
            candidate_diseases=["obesity"],
            cache_dir=tmp_path,
        )

    assert first == {"obesity"}
    assert second == {"obesity"}
    mock_llm.assert_awaited_once()


# --- list_approved_indications_from_labels ---


async def test_list_approved_indications_returns_parsed_list(tmp_path):
    llm_response = json.dumps(["type 2 diabetes mellitus", "chronic weight management"])

    with patch(
        "indication_scout.services.approval_check.query_llm",
        new=AsyncMock(return_value=llm_response),
    ):
        result = await list_approved_indications_from_labels(
            label_texts=[
                "OZEMPIC is indicated as an adjunct to diet and exercise to "
                "improve glycemic control in adults with type 2 diabetes mellitus."
            ],
            cache_dir=tmp_path,
        )

    assert result == ["type 2 diabetes mellitus", "chronic weight management"]


async def test_list_approved_indications_empty_label_texts_no_llm_call(tmp_path):
    mock_llm = AsyncMock()
    with patch(
        "indication_scout.services.approval_check.query_llm",
        new=mock_llm,
    ):
        result = await list_approved_indications_from_labels(
            label_texts=[],
            cache_dir=tmp_path,
        )

    assert result == []
    mock_llm.assert_not_awaited()


async def test_list_approved_indications_strips_markdown_fences(tmp_path):
    llm_response = '```json\n["MASH"]\n```'

    with patch(
        "indication_scout.services.approval_check.query_llm",
        new=AsyncMock(return_value=llm_response),
    ):
        result = await list_approved_indications_from_labels(
            label_texts=[
                "Indicated for MASH with moderate to advanced liver fibrosis."
            ],
            cache_dir=tmp_path,
        )

    assert result == ["MASH"]


async def test_list_approved_indications_invalid_json_returns_empty(tmp_path):
    llm_response = "not valid json"

    with patch(
        "indication_scout.services.approval_check.query_llm",
        new=AsyncMock(return_value=llm_response),
    ):
        result = await list_approved_indications_from_labels(
            label_texts=["Some label text."],
            cache_dir=tmp_path,
        )

    assert result == []


async def test_list_approved_indications_extracts_array_from_wrapped_object(tmp_path):
    llm_response = json.dumps({"approved": ["obesity"]})

    with patch(
        "indication_scout.services.approval_check.query_llm",
        new=AsyncMock(return_value=llm_response),
    ):
        result = await list_approved_indications_from_labels(
            label_texts=["Indicated for obesity."],
            cache_dir=tmp_path,
        )

    assert result == ["obesity"]


async def test_list_approved_indications_dedupes_case_insensitively(tmp_path):
    llm_response = json.dumps(
        ["Obesity", "obesity", "  type 2 diabetes mellitus  ", ""]
    )

    with patch(
        "indication_scout.services.approval_check.query_llm",
        new=AsyncMock(return_value=llm_response),
    ):
        result = await list_approved_indications_from_labels(
            label_texts=["Some label."],
            cache_dir=tmp_path,
        )

    assert result == ["Obesity", "type 2 diabetes mellitus"]


async def test_list_approved_indications_skips_non_string_items(tmp_path):
    llm_response = json.dumps(["obesity", 42, None, "type 2 diabetes mellitus"])

    with patch(
        "indication_scout.services.approval_check.query_llm",
        new=AsyncMock(return_value=llm_response),
    ):
        result = await list_approved_indications_from_labels(
            label_texts=["Some label."],
            cache_dir=tmp_path,
        )

    assert result == ["obesity", "type 2 diabetes mellitus"]


async def test_list_approved_indications_caches_result(tmp_path):
    llm_response = json.dumps(["obesity"])
    mock_llm = AsyncMock(return_value=llm_response)

    with patch(
        "indication_scout.services.approval_check.query_llm",
        new=mock_llm,
    ):
        first = await list_approved_indications_from_labels(
            label_texts=["Indicated for obesity."],
            cache_dir=tmp_path,
        )
        second = await list_approved_indications_from_labels(
            label_texts=["Indicated for obesity."],
            cache_dir=tmp_path,
        )

    assert first == ["obesity"]
    assert second == ["obesity"]
    mock_llm.assert_awaited_once()


# --- get_approved_indications (hardcoded table lookup) ---------------------
#
# Reads the real drug_approvals.json file. The table is loaded via
# functools.lru_cache, so each test calls cache_clear() to ensure the file
# is re-read in the current test's process state.


@pytest.fixture(autouse=True)
def _clear_approvals_cache():
    """Reset the lru_cache on the table loader between tests."""
    _load_drug_approvals_table.cache_clear()
    yield
    _load_drug_approvals_table.cache_clear()


@pytest.mark.parametrize(
    "as_of, candidate, expected_in_set",
    [
        # 2025-01-01 cutoff: MASH approval is 2025-08-15, NOT yet approved
        # → MASH stays a candidate (this is the holdout-recoverable case).
        (date(2025, 1, 1), "MASH", False),
        (date(2025, 1, 1), "non-alcoholic steatohepatitis (MASH)", False),
        # 2025-08-15 is the approval date itself; the lookup uses strict
        # less-than, so on the day of approval it's still NOT yet approved.
        (date(2025, 8, 15), "MASH", False),
        # 2025-08-16: one day after approval → MASH is approved → dropped.
        (date(2025, 8, 16), "MASH", True),
        # 2026-01-01: well after the MASH approval → dropped.
        (date(2026, 1, 1), "MASH", True),
        # 2026-01-01: substring match — table entry "MASH" should match a
        # candidate that contains it. Verifies the substring-matching rule.
        (date(2026, 1, 1), "non-alcoholic steatohepatitis (MASH)", True),
    ],
)
def test_get_approved_indications_semaglutide_mash(as_of, candidate, expected_in_set):
    """Verify the cutoff semantics for semaglutide × MASH against the real table.

    MASH was approved 2025-08-15. The lookup uses strict less-than on the
    cutoff, so a holdout dated on or before that day must NOT see MASH as
    approved (and therefore must NOT drop it from the candidate allowlist).
    """
    result = get_approved_indications(
        drug_name="semaglutide",
        candidate_diseases=[candidate],
        as_of=as_of,
    )
    if expected_in_set:
        assert result == {candidate}
    else:
        assert result == set()


def test_get_approved_indications_semaglutide_pre_2017_returns_empty():
    """Cutoff before semaglutide's first approval (2017-12-05) → empty set."""
    result = get_approved_indications(
        drug_name="semaglutide",
        candidate_diseases=[
            "type 2 diabetes mellitus",
            "MASH",
            "chronic weight management",
        ],
        as_of=date(2017, 1, 1),
    )
    assert result == set()


def test_get_approved_indications_semaglutide_2022_returns_three():
    """Cutoff 2022-01-01 → T2DM (2017), CV risk (2020), and obesity (2021)
    are all approved; CKD (2025) and MASH (2025) are not.
    """
    candidates = [
        "type 2 diabetes mellitus",
        "cardiovascular risk reduction",
        "obesity",
        "chronic kidney disease",
        "MASH",
    ]
    result = get_approved_indications(
        drug_name="semaglutide",
        candidate_diseases=candidates,
        as_of=date(2022, 1, 1),
    )
    assert result == {
        "type 2 diabetes mellitus",
        "cardiovascular risk reduction",
        "obesity",
    }


def test_get_approved_indications_returns_empty_when_as_of_is_none():
    """as_of=None → callers should use the live FDA path; lookup returns empty."""
    result = get_approved_indications(
        drug_name="semaglutide",
        candidate_diseases=["MASH"],
        as_of=None,
    )
    assert result == set()


def test_get_approved_indications_uncurated_drug_returns_empty(caplog):
    """Drug not in the table → empty set + warning logged."""
    with caplog.at_level("WARNING"):
        result = get_approved_indications(
            drug_name="not-a-real-drug",
            candidate_diseases=["obesity"],
            as_of=date(2025, 1, 1),
        )
    assert result == set()
    assert any(
        "not in hardcoded approvals table" in rec.message for rec in caplog.records
    )


# --- list_approved_indications_at ------------------------------------------


def test_list_approved_indications_at_semaglutide_2022():
    """Cutoff 2022-01-01 → only the three pre-2022 approvals returned, in
    table order (the loader preserves JSON file order).
    """
    result = list_approved_indications_at("semaglutide", date(2022, 1, 1))
    assert result == [
        "type 2 diabetes mellitus",
        "cardiovascular risk reduction",
        "obesity",
    ]


def test_list_approved_indications_at_semaglutide_pre_2017_returns_empty():
    """Cutoff before any approval → empty list."""
    result = list_approved_indications_at("semaglutide", date(2017, 1, 1))
    assert result == []


def test_list_approved_indications_at_returns_empty_when_as_of_is_none():
    """as_of=None → empty list (callers should use the live FDA path)."""
    result = list_approved_indications_at("semaglutide", None)
    assert result == []
