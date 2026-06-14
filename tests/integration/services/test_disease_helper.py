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
    resolve_mesh_id,
)

logger = logging.getLogger(__name__)


async def test_resolve_mesh_id_covid_picks_disease_not_testing():
    """B1 regression: bare "covid-19" free-text-matched sub-concepts and resolved to
    "COVID-19 Testing" (D000086742), causing silent false-zero trial counts. The
    "[MeSH Terms]" qualifier must return the disease descriptor "COVID-19" (D000086382)."""
    result = await resolve_mesh_id("covid-19")
    assert result == ("D000086382", "COVID-19")


@pytest.mark.parametrize(
    "disease, expected_id",
    [
        ("hypertension", "D006973"),
        ("raynaud disease", "D011928"),
        ("obesity", "D009765"),
    ],
)
async def test_resolve_mesh_id_known_diseases_unchanged(disease, expected_id):
    """The [MeSH Terms] qualifier must not regress diseases that already resolved right."""
    result = await resolve_mesh_id(disease)
    assert result is not None
    assert result[0] == expected_id


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

@pytest.mark.parametrize(
    "d1, d2, d3, d4",
    [
        (
            {
                "narcolepsy",
                "narcolepsy-cataplexy syndrome",
                "obesity",
                "overweight body mass index status",
            },
            {"type 2 diabetes mellitus"},
            {
                frozenset({"narcolepsy", "narcolepsy-cataplexy syndrome"}),
                frozenset({"obesity", "overweight body mass index status"}),
            },
            set(),
        ),
        (
            {"obesity", "overweight body mass index status"},
            set(),
            {frozenset({"obesity", "overweight body mass index status"})},
            set(),
        ),
        # Distinct substance-dependence disorders must NOT be collapsed into each
        # other or the "drug dependence" / "substance abuse" umbrella. Only the two
        # umbrella synonyms may merge together.
        (
            {
                "cocaine dependence",
                "methamphetamine dependence",
                "nicotine dependence",
                "alcohol dependence",
                "cannabis dependence",
                "substance abuse",
            },
            set(),
            set(),
            {
                "cocaine dependence",
                "methamphetamine dependence",
                "nicotine dependence",
                "alcohol dependence",
                "cannabis dependence",
            },
        ),
    ],
)
async def test_merge_duplicate_diseases(d1, d2, d3, d4):
    # d1: input diseases, d2: drug indications, d3: expected merge groups (each a
    # frozenset, so canonical-name choice and alias ordering don't matter),
    # d4: diseases that must each stay in their own group (never merged together).
    result = await merge_duplicate_diseases(
        list(d1),
        list(d2),
    )

    assert "merge" in result
    assert "remove" in result

    actual_groups = {
        frozenset({canonical} | set(aliases))
        for canonical, aliases in result["merge"].items()
    }

    for group in d3:
        assert any(
            group <= merged for merged in actual_groups
        ), f"Expected {set(group)} to be merged, got: {result['merge']}"

    # No two diseases in d4 may share a merge group (they are clinically distinct).
    for merged in actual_groups:
        shared = d4 & merged
        assert len(shared) <= 1, (
            f"Expected {sorted(d4)} to stay separate, but {sorted(shared)} were "
            f"merged together, got: {result['merge']}"
        )

    # `remove` is for diseases equivalent to the drug's EXISTING indications
    # (here type 2 diabetes mellitus) — not for merge aliases. None of the input
    # diseases matches that indication, so nothing should be removed.
    assert result["remove"] == [], (
        f"Expected no removals (no input matches an existing indication), "
        f"got: {result['remove']}"
    )


async def test_llm_normalize_disease_batch_returns_correct_forms(tmp_path):
    """llm_normalize_disease_batch returns correct normalised forms for known disease terms.

    Uses a tmp_path cache so this test is isolated from production cache state.
    Run once, observe the output, then fill in the expected values below.
    """
    with patch("indication_scout.services.disease_helper.DEFAULT_CACHE_DIR", tmp_path):
        result = await llm_normalize_disease_batch(
            ["type 2 diabetes mellitus", "narcolepsy-cataplexy syndrome"]
        )

    assert set(result.keys()) == {
        "type 2 diabetes mellitus",
        "narcolepsy-cataplexy syndrome",
    }
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

    assert set(result.keys()) == {
        "type 2 diabetes mellitus",
        "narcolepsy-cataplexy syndrome",
    }


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


# ── resolve_mesh_id integration tests ───────────────────────────────────────


@pytest.mark.parametrize(
    "indication, expected_mesh_id, expected_preferred_term",
    [
        ("hypertension", "D006973", "Hypertension"),
        ("type 2 diabetes", "D003924", "Diabetes Mellitus, Type 2"),
        ("asthma", "D001249", "Asthma"),
        ("parkinson disease", "D010300", "Parkinson Disease"),
        ("rheumatoid arthritis", "D001172", "Arthritis, Rheumatoid"),
        ("alzheimer disease", "D000544", "Alzheimer Disease"),
        ("breast neoplasms", "D001943", "Breast Neoplasms"),
        ("multiple sclerosis", "D009103", "Multiple Sclerosis"),
        ("psoriasis", "D011565", "Psoriasis"),
        ("epilepsy", "D004827", "Epilepsy"),
    ],
)
async def test_resolve_mesh_id_real_ncbi(
    indication, expected_mesh_id, expected_preferred_term
):
    """Real NCBI esearch+esummary resolves common indications to (D-number, preferred term)."""
    result = await resolve_mesh_id(indication)
    assert result == (expected_mesh_id, expected_preferred_term)


@pytest.mark.parametrize(
    "indication, expected_mesh_id, expected_preferred_term",
    [
        # "skin melanoma" is not itself a MeSH heading; NCBI ATM translates it
        # to "cutaneous malignant melanoma"[MeSH Terms] (D000096142).
        ("skin melanoma", "D000096142", "Cutaneous Malignant Melanoma"),
        ("high blood pressure", "D006973", "Hypertension"),
        ("heart attack", "D009203", "Myocardial Infarction"),
        ("stroke", "D020521", "Stroke"),
        ("lou gehrig disease", "D000690", "Amyotrophic Lateral Sclerosis"),
    ],
)
async def test_resolve_mesh_id_atm_fallback(
    tmp_path, indication, expected_mesh_id, expected_preferred_term
):
    """Descriptive phrases with no exact MeSH heading resolve via NCBI ATM fallback."""
    with patch("indication_scout.services.disease_helper.DEFAULT_CACHE_DIR", tmp_path):
        result = await resolve_mesh_id(indication)
    assert result == (expected_mesh_id, expected_preferred_term)
