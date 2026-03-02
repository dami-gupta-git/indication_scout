"""Integration tests for services/disease_normalizer."""

import logging

import pytest

from indication_scout.markers import no_review
from indication_scout.services.disease_normalizer import (
    BROADENING_BLOCKLIST,
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
