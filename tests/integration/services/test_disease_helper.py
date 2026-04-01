"""Integration tests for services/disease_normalizer."""

import logging
from unittest.mock import patch, AsyncMock

import pytest

from indication_scout.markers import no_review
from indication_scout.services.disease_helper import (
    BROADENING_BLOCKLIST,
    llm_normalize_disease_batch,
    merge_duplicate_diseases,
    normalize_batch,
    normalize_for_pubmed,
)

logger = logging.getLogger(__name__)


@no_review
# Exclude from testing rules, TODO delete
async def test_single_disease_normalizer():
    disease = "hepatocellular carcinoma"
    drug = ""
    result = await normalize_for_pubmed(disease, drug)
    assert result


@no_review
# Exclude from testing rules, TODO delete
async def test_single_drug_disease_normalizer():
    disease = "colorectal neoplasm"
    drug = "metformin"
    result = await normalize_for_pubmed(disease, drug)
    assert result


async def test_normalize_returns_multiple_terms():
    # "atopic eczema" should normalize to two terms joined by OR (e.g. "eczema OR dermatitis")
    result = await normalize_for_pubmed("atopic eczema", None)
    terms = [t.strip().lower() for t in result.split("OR")]
    assert len(terms) == 2
    assert terms[0] == "eczema"
    assert terms[1] == "dermatitis"


@pytest.mark.parametrize(
    "raw_term",
    [
        "neoplasm",
        "cancer",
        "malignancy",
    ],
)
async def test_blocklist_terms_return_raw_term(raw_term):
    """Bare blocklisted terms should be returned unchanged (not further generalized)."""
    result = await normalize_for_pubmed(raw_term, drug_name=None)
    result_terms = {t.strip().lower() for t in result.split("OR")}
    assert raw_term.lower() in result_terms


async def test_organ_specificity_not_lost_for_cancer_terms():
    """Organ-specific cancer terms must retain organ context, not collapse to bare 'cancer'."""
    result = await normalize_for_pubmed("non-small cell lung carcinoma", drug_name=None)
    terms = [t.strip().lower() for t in result.split("OR")]
    assert any("lung" in t for t in terms)


@pytest.mark.parametrize(
    "disease, drug, required_keyword",
    [
        ("atopic dermatitis", "baricitinib", "dermatitis"),
        ("obesity", "bupropion", "obesity"),
        ("narcolepsy-cataplexy syndrome", "modafinil", "narcolepsy"),
        ("diabetic nephropathy", "sildenafil", "nephropathy"),
        ("myelofibrosis", "baricitinib", "myelofibrosis"),
    ],
)
async def test_multiple_drug_disease_normalizer(disease, drug, required_keyword):
    """normalize_for_pubmed with a drug name returns a non-empty, specific result."""
    result = await normalize_for_pubmed(disease, drug_name=drug)
    assert result, f"Expected a non-empty result for {drug} + {disease}"
    result_terms = {t.strip().lower() for t in result.split("OR")}
    assert not (
        result_terms <= BROADENING_BLOCKLIST
    ), f"Result '{result}' collapsed to over-generic terms for {drug} + {disease}"
    assert any(
        required_keyword in t for t in result_terms
    ), f"Expected '{required_keyword}' in result terms {result_terms} for {drug} + {disease}"


async def test_merge_duplicate_diseases():
    result = await merge_duplicate_diseases(
        [
            "narcolepsy",
            "narcolepsy-cataplexy syndrome",
            "obesity",
            "overweight body mass index status",
        ],
        ["type 2 diabetes mellitus"],
    )

    assert "merge" in result
    assert "remove" in result

    all_merged = []
    for canonical, aliases in result["merge"].items():
        all_merged.append({canonical} | set(aliases))

    assert any(
        {"narcolepsy", "narcolepsy-cataplexy syndrome"}.issubset(group)
        for group in all_merged
    ), f"Expected narcolepsy variants to be merged, got: {result['merge']}"

    assert any(
        {"obesity", "overweight body mass index status"}.issubset(group)
        for group in all_merged
    ), f"Expected obesity variants to be merged, got: {result['merge']}"

    assert "narcolepsy" not in result["remove"]
    assert "obesity" in result["remove"]


async def test_llm_normalize_disease_batch_returns_correct_forms(tmp_path):
    """llm_normalize_disease_batch returns correct normalised forms for known disease terms.

    Uses a tmp_path cache so this test is isolated from production cache state.
    Run once, observe the output, then fill in the expected values below.
    """
    with patch("indication_scout.services.disease_helper.DEFAULT_CACHE_DIR", tmp_path):
        result = await llm_normalize_disease_batch(
            ["type 2 diabetes mellitus", "narcolepsy-cataplexy syndrome"]
        )

    assert set(result.keys()) == {"type 2 diabetes mellitus", "narcolepsy-cataplexy syndrome"}
    assert result["type 2 diabetes mellitus"] == "type 2 diabetes OR diabetes mellitus"
    assert result["narcolepsy-cataplexy syndrome"] == "narcolepsy"


async def test_llm_normalize_disease_batch_second_call_uses_cache(tmp_path):
    """Second call for the same terms returns from cache with no LLM call."""
    with patch("indication_scout.services.disease_helper.DEFAULT_CACHE_DIR", tmp_path):
        # Prime the cache
        await llm_normalize_disease_batch(
            ["type 2 diabetes mellitus", "narcolepsy-cataplexy syndrome"]
        )

        # Second call — LLM must not be invoked
        with patch(
            "indication_scout.services.disease_helper.query_small_llm",
            new=AsyncMock(side_effect=AssertionError("LLM called on second request")),
        ):
            result = await llm_normalize_disease_batch(
                ["type 2 diabetes mellitus", "narcolepsy-cataplexy syndrome"]
            )

    assert set(result.keys()) == {"type 2 diabetes mellitus", "narcolepsy-cataplexy syndrome"}


async def test_normalize_batch_returns_pubmed_friendly_term(test_cache_dir):
    """normalize_batch returns a PubMed-friendly term for a known disease.

    Uses metformin + type 2 diabetes — a well-studied pair with thousands of
    PubMed results — to confirm the normalised term yields at least MIN_RESULTS hits.
    """
    from indication_scout.services.disease_helper import MIN_RESULTS, pubmed_count

    result = await normalize_batch(["type 2 diabetes mellitus"], drug_name="metformin")

    assert "type 2 diabetes mellitus" in result
    normalized = result["type 2 diabetes mellitus"]
    assert normalized  # non-empty

    count = await pubmed_count(f"metformin AND ({normalized})")
    assert count >= MIN_RESULTS
