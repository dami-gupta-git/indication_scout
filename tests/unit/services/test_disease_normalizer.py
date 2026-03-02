"""Unit tests for disease_normalizer blocklist logic."""

import logging
from unittest.mock import AsyncMock, patch

import pytest

from indication_scout.services.disease_normalizer import (
    BROADENING_BLOCKLIST,
    llm_normalize_disease,
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
        "indication_scout.services.disease_normalizer.llm_normalize_disease",
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
        "indication_scout.services.disease_normalizer.llm_normalize_disease",
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
        "indication_scout.services.disease_normalizer.llm_normalize_disease",
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
            "indication_scout.services.disease_normalizer.llm_normalize_disease",
            new=mock_llm_normalize,
        ),
        patch(
            "indication_scout.services.disease_normalizer.pubmed_count",
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
            "indication_scout.services.disease_normalizer.llm_normalize_disease",
            new=mock_llm_normalize,
        ),
        patch(
            "indication_scout.services.disease_normalizer.pubmed_count",
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
        "indication_scout.services.disease_normalizer.query_small_llm",
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
        "indication_scout.services.disease_normalizer.query_small_llm",
        new=AsyncMock(return_value=llm_response),
    ):
        result = await llm_normalize_disease(raw_term)
        assert result == expected
