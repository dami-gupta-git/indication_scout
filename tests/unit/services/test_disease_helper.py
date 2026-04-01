"""Unit tests for disease_helper blocklist logic."""

import json
import logging
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from indication_scout.constants import BROADENING_BLOCKLIST
from indication_scout.services.disease_helper import (
    llm_normalize_disease,
    llm_normalize_disease_batch,
    merge_duplicate_diseases,
    normalize_for_pubmed,
)

logger = logging.getLogger(__name__)


# ── normalize_for_pubmed happy path ──────────────────────────────────────────


@pytest.mark.parametrize(
    "raw_term, llm_output, expected",
    [
        ("narcolepsy-cataplexy syndrome", "narcolepsy", "narcolepsy"),
        ("atopic eczema", "eczema OR dermatitis", "eczema OR dermatitis"),
        (
            "non-small cell lung carcinoma",
            "lung cancer OR lung neoplasm",
            "lung cancer OR lung neoplasm",
        ),
        (
            "hepatocellular carcinoma",
            "liver cancer OR hepatocellular carcinoma",
            "liver cancer OR hepatocellular carcinoma",
        ),
        (
            "myocardial infarction",
            "heart attack OR myocardial infarction",
            "heart attack OR myocardial infarction",
        ),
    ],
)
async def test_normalize_for_pubmed_returns_llm_output(raw_term, llm_output, expected):
    """normalize_for_pubmed passes LLM output through when it is not blocklisted."""
    with patch(
        "indication_scout.services.disease_helper.llm_normalize_disease",
        new=AsyncMock(return_value=llm_output),
    ):
        result = await normalize_for_pubmed(raw_term, drug_name=None)
        assert result == expected


# ── Blocklist guard on initial normalization ──────────────────────────────────


@pytest.mark.parametrize(
    "raw_term, llm_output",
    [
        ("neoplasm", "cancer"),
        ("malignant neoplasm", "neoplasm"),
        ("tumor", "tumor"),
        ("malignancy", "malignancy"),
        ("cancer OR neoplasm", "cancer OR neoplasm"),
    ],
)
async def test_blocklist_rejects_overly_broad_initial_normalization(
    raw_term, llm_output
):
    """If LLM returns only blocklisted terms, raw_term is returned unchanged."""
    with patch(
        "indication_scout.services.disease_helper.llm_normalize_disease",
        new=AsyncMock(return_value=llm_output),
    ):
        result = await normalize_for_pubmed(raw_term, drug_name=None)
        assert result == raw_term


# ── Organ specificity preserved ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw_term, llm_output, forbidden_exact",
    [
        ("non-small cell lung carcinoma", "lung cancer OR lung neoplasm", "cancer"),
        (
            "hepatocellular carcinoma",
            "liver cancer OR hepatocellular carcinoma",
            "cancer",
        ),
    ],
)
async def test_organ_specificity_preserved(raw_term, llm_output, forbidden_exact):
    """Result must retain organ context and not collapse to a bare blocklisted term."""
    with patch(
        "indication_scout.services.disease_helper.llm_normalize_disease",
        new=AsyncMock(return_value=llm_output),
    ):
        result = await normalize_for_pubmed(raw_term, drug_name=None)
        terms = [t.strip().lower() for t in result.split("OR")]
        # Must not be a bare blocklisted term
        assert not (set(terms) <= BROADENING_BLOCKLIST)
        # Must not have collapsed to only the forbidden term
        assert forbidden_exact not in terms or len(terms) > 1


# ── Fallback blocklist guard ──────────────────────────────────────────────────


async def test_fallback_blocklist_rejects_overly_broad_broader_term():
    """If the fallback LLM generalizes to a blocklisted term, it is discarded."""
    call_count = 0

    async def mock_llm_normalize(term: str) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return "renal tubular dysgenesis"  # first call: specific
        return "disease"  # second call (fallback): blocklisted

    with (
        patch(
            "indication_scout.services.disease_helper.llm_normalize_disease",
            new=mock_llm_normalize,
        ),
        patch(
            "indication_scout.services.disease_helper.pubmed_count",
            new=AsyncMock(return_value=0),  # force fallback path
        ),
    ):
        result = await normalize_for_pubmed(
            "renal tubular dysgenesis", drug_name="metformin"
        )
        # Fallback was blocked; should keep the first LLM result
        assert result == "renal tubular dysgenesis"


async def test_fallback_accepted_when_not_blocklisted():
    """If the fallback term is specific enough, it is accepted."""
    call_count = 0

    async def mock_llm_normalize(term: str) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return "renal tubular dysgenesis"
        return "kidney disease"  # not in blocklist

    with (
        patch(
            "indication_scout.services.disease_helper.llm_normalize_disease",
            new=mock_llm_normalize,
        ),
        patch(
            "indication_scout.services.disease_helper.pubmed_count",
            new=AsyncMock(side_effect=[0, 10]),  # first: too few; second: enough
        ),
    ):
        result = await normalize_for_pubmed(
            "renal tubular dysgenesis", drug_name="metformin"
        )
        assert result == "kidney disease"


# ── llm_normalize_disease unit tests ─────────────────────────────────────────


@pytest.mark.parametrize(
    "raw_term, llm_response, expected",
    [
        ("atopic eczema", "eczema OR dermatitis", "eczema OR dermatitis"),
        ("narcolepsy-cataplexy syndrome", "narcolepsy", "narcolepsy"),
        (
            "non-small cell lung carcinoma",
            "lung cancer OR lung neoplasm",
            "lung cancer OR lung neoplasm",
        ),
        (
            "myocardial infarction",
            "heart attack OR myocardial infarction",
            "heart attack OR myocardial infarction",
        ),
        ('"CML"', "chronic myeloid leukemia", "chronic myeloid leukemia"),
    ],
)
async def test_llm_normalize_disease_returns_cleaned_llm_output(
    raw_term, llm_response, expected
):
    """llm_normalize_disease strips quotes/whitespace and returns the LLM output."""
    with patch(
        "indication_scout.services.disease_helper.query_small_llm",
        new=AsyncMock(return_value=llm_response),
    ):
        result = await llm_normalize_disease(raw_term)
        assert result == expected


@pytest.mark.parametrize(
    "raw_term, llm_response, expected",
    [
        ("narcolepsy-cataplexy syndrome", '"narcolepsy"', "narcolepsy"),
        ("atopic eczema", "  eczema OR dermatitis  ", "eczema OR dermatitis"),
    ],
)
async def test_llm_normalize_disease_strips_llm_response(
    raw_term, llm_response, expected
):
    """Leading/trailing quotes and whitespace in the LLM response are stripped."""
    with patch(
        "indication_scout.services.disease_helper.query_small_llm",
        new=AsyncMock(return_value=llm_response),
    ):
        result = await llm_normalize_disease(raw_term)
        assert result == expected


# ── merge_duplicate_diseases unit tests ──────────────────────────────────────


@pytest.mark.parametrize(
    "llm_response",
    [
        '{"merge": {"narcolepsy": ["narcolepsy-cataplexy syndrome"]}, "remove": []}',
        '```json\n{"merge": {"narcolepsy": ["narcolepsy-cataplexy syndrome"]}, "remove": []}\n```',
        '```\n{"merge": {"narcolepsy": ["narcolepsy-cataplexy syndrome"]}, "remove": []}\n```',
    ],
)
async def test_merge_duplicate_diseases_parses_response_formats(llm_response):
    """merge_duplicate_diseases handles raw JSON and markdown code fences."""
    with patch(
        "indication_scout.services.disease_helper.query_small_llm",
        new=AsyncMock(return_value=llm_response),
    ):
        result = await merge_duplicate_diseases(
            ["narcolepsy", "narcolepsy-cataplexy syndrome"], []
        )
        assert result == {
            "merge": {"narcolepsy": ["narcolepsy-cataplexy syndrome"]},
            "remove": [],
        }


async def test_merge_duplicate_diseases_returns_fallback_on_invalid_json():
    """merge_duplicate_diseases returns empty structure when LLM returns invalid JSON."""
    with patch(
        "indication_scout.services.disease_helper.query_small_llm",
        new=AsyncMock(return_value="not valid json at all"),
    ):
        result = await merge_duplicate_diseases(["narcolepsy"], [])
        assert result == {"merge": {}, "remove": []}


# ── llm_normalize_disease_batch unit tests ───────────────────────────────────


async def test_batch_all_cached_no_llm_call():
    """If all terms are cached, no LLM call is made and cached values are returned."""
    cache_values = {
        "narcolepsy-cataplexy syndrome": "narcolepsy",
        "type 2 diabetes mellitus": "type 2 diabetes",
    }

    def fake_cache_get(namespace, params, cache_dir):
        return cache_values.get(params["raw_term"])

    with patch(
        "indication_scout.services.disease_helper.cache_get",
        side_effect=fake_cache_get,
    ), patch(
        "indication_scout.services.disease_helper.query_small_llm",
    ) as mock_llm:
        result = await llm_normalize_disease_batch(
            ["narcolepsy-cataplexy syndrome", "type 2 diabetes mellitus"]
        )

    mock_llm.assert_not_called()
    assert result == {
        "narcolepsy-cataplexy syndrome": "narcolepsy",
        "type 2 diabetes mellitus": "type 2 diabetes",
    }


async def test_batch_only_cache_misses_sent_to_llm():
    """Only the terms not in cache are included in the LLM batch call."""
    cache_values = {"type 2 diabetes mellitus": "type 2 diabetes"}
    llm_response = json.dumps({"narcolepsy-cataplexy syndrome": "narcolepsy"})

    def fake_cache_get(namespace, params, cache_dir):
        return cache_values.get(params["raw_term"])

    with patch(
        "indication_scout.services.disease_helper.cache_get",
        side_effect=fake_cache_get,
    ), patch(
        "indication_scout.services.disease_helper.cache_set",
    ), patch(
        "indication_scout.services.disease_helper.query_small_llm",
        new=AsyncMock(return_value=llm_response),
    ) as mock_llm:
        result = await llm_normalize_disease_batch(
            ["narcolepsy-cataplexy syndrome", "type 2 diabetes mellitus"]
        )

    # LLM was called once with only the uncached term
    mock_llm.assert_awaited_once()
    prompt_arg = mock_llm.call_args[0][0]
    assert "narcolepsy-cataplexy syndrome" in prompt_arg
    assert "type 2 diabetes mellitus" not in prompt_arg

    assert result == {
        "narcolepsy-cataplexy syndrome": "narcolepsy",
        "type 2 diabetes mellitus": "type 2 diabetes",
    }


async def test_batch_results_cached_individually():
    """After the LLM batch call, each result is written to cache individually."""
    llm_response = json.dumps({
        "narcolepsy-cataplexy syndrome": "narcolepsy",
        "type 2 diabetes mellitus": "type 2 diabetes",
    })

    with patch(
        "indication_scout.services.disease_helper.cache_get",
        return_value=None,
    ), patch(
        "indication_scout.services.disease_helper.cache_set",
    ) as mock_cache_set, patch(
        "indication_scout.services.disease_helper.query_small_llm",
        new=AsyncMock(return_value=llm_response),
    ):
        await llm_normalize_disease_batch(
            ["narcolepsy-cataplexy syndrome", "type 2 diabetes mellitus"]
        )

    assert mock_cache_set.call_count == 2
    cached_keys = {c.args[1]["raw_term"] for c in mock_cache_set.call_args_list}
    assert cached_keys == {"narcolepsy-cataplexy syndrome", "type 2 diabetes mellitus"}


async def test_batch_returned_keys_match_all_input_terms():
    """Returned dict contains every input term as a key, whether cached or not."""
    cache_values = {"type 2 diabetes mellitus": "type 2 diabetes"}
    llm_response = json.dumps({"narcolepsy-cataplexy syndrome": "narcolepsy"})

    def fake_cache_get(namespace, params, cache_dir):
        return cache_values.get(params["raw_term"])

    with patch(
        "indication_scout.services.disease_helper.cache_get",
        side_effect=fake_cache_get,
    ), patch(
        "indication_scout.services.disease_helper.cache_set",
    ), patch(
        "indication_scout.services.disease_helper.query_small_llm",
        new=AsyncMock(return_value=llm_response),
    ):
        result = await llm_normalize_disease_batch(
            ["narcolepsy-cataplexy syndrome", "type 2 diabetes mellitus"]
        )

    assert set(result.keys()) == {"narcolepsy-cataplexy syndrome", "type 2 diabetes mellitus"}
