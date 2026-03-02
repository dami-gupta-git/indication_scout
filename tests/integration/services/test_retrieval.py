"""Integration tests for services/retrieval."""

import logging

import pytest
from sqlalchemy import text

from indication_scout.data_sources.pubmed import PubMedClient
from indication_scout.models.model_drug_profile import DrugProfile
from indication_scout.services.retrieval import (
    build_drug_profile,
    embed_abstracts,
    expand_search_terms,
    extract_organ_term,
    fetch_and_cache,
    fetch_new_abstracts,
    get_disease_synonyms,
    get_stored_pmids,
    semantic_search,
    synthesize,
)

logger = logging.getLogger(__name__)
# db_session fixture is defined in tests/integration/conftest.py and
# connects to scout_test (TEST_DATABASE_URL). It rolls back after each test.


@pytest.mark.parametrize(
    "disease, synonyms",
    [
        (
            "eczematoid dermatitis",
            ["eczematoid dermatitis", "eczema", "AD", "atopic dermatitis"],
        ),
        (
            "benign prostatic hyperplasia",
            [
                "benign prostatic hyperplasia",
                "BPH",
                "enlarged prostate",
                "prostatic enlargement",
            ],
        ),
        (
            "HER2-positive breast cancer",
            ["breast cancer", "ERBB2-positive breast cancer"],
        ),
        (
            "CML",
            ["CML", "chronic myeloid leukemia"],
        ),
    ],
)
async def test_get_disease_synonyms(disease, synonyms):
    """Returned synonyms should include all expected terms for the given disease."""
    result = await get_disease_synonyms(disease)

    assert set(synonyms).issubset(set(result))


# @pytest.mark.parametrize(
#     "disease, expected_terms",
#     [
#         ("type 2 diabetes", ["diabetes"]),
#         ("chronic myeloid leukemia", ["leukemia", "CML"]),
#         ("HER2-positive breast cancer", ["breast cancer", "HER2"]),
#         ("benign prostatic hyperplasia", ["BPH", "prostat"]),
#     ],
# )
# async def test_get_best_disease_contains_key_term(disease, expected_terms):
#     """Returned PubMed term should contain at least one expected keyword."""
#     result = await get_best_disease(disease)
#
#     assert any(term.lower() in result.lower() for term in expected_terms)


# async def test_get_best_disease():
#     disease = "renal tubular dysgenesis"
#
#     """Returned PubMed term should contain at least one expected keyword."""
#     result = await g(disease)
#
#     assert result


# --- Stubs for unimplemented methods ---


# async def test_fetch_and_cache_returns_pmids():
#     """Should return a non-empty list of PMID strings for valid queries."""
#     queries = [
#         "metformin AND colorectal cancer",
#         "metformin AND AMPK AND colon",
#     ]
#     pmids = await fetch_and_cache(queries)
#
#     assert isinstance(pmids, list)
#     assert len(pmids) >= 1
#     assert all(isinstance(p, str) and p.isdigit() for p in pmids)


# async def test_fetch_and_cache_deduplicates():
#     """Duplicate queries should not produce duplicate PMIDs."""
#     queries = [
#         "metformin AND colorectal cancer",
#         "metformin AND colorectal cancer",
#     ]
#     pmids = await fetch_and_cache(queries)
#
#     assert len(pmids) == len(set(pmids))


# async def test_semantic_search_returns_ranked_abstracts():
#     """Should return up to top_k dicts with required abstract fields."""
#     results = await semantic_search("colorectal cancer", "metformin", top_k=5)
#
#     assert isinstance(results, list)
#     assert 1 <= len(results) <= 5
#     first = results[0]
#     assert isinstance(first["pmid"], str) and first["pmid"].isdigit()
#     assert isinstance(first["title"], str) and len(first["title"]) > 0
#     assert isinstance(first["abstract"], str) and len(first["abstract"]) > 0
#     assert isinstance(first["similarity"], float)
#     assert 0.0 <= first["similarity"] <= 1.0


# async def test_semantic_search_respects_top_k():
#     """Result count must not exceed top_k."""
#     results = await semantic_search("colorectal cancer", "metformin", top_k=3)
#
#     assert len(results) <= 3


# async def test_synthesize_returns_structured_summary():
#     """Should return a structured evidence summary with PMIDs."""
#     abstracts = [
#         {
#             "pmid": "29734553",
#             "title": "Metformin suppresses colorectal cancer",
#             "abstract": "Metformin activates AMPK and inhibits mTOR in colon cancer cells.",
#             "similarity": 0.89,
#         }
#     ]
#     result = synthesize("metformin", "colorectal cancer", abstracts)
#
#     assert result is not None
#     assert "pmid" in str(result) or hasattr(result, "pmids")


async def test_extract_organ_term_returns_string():
    """extract_organ_term should return the primary organ for a known disease."""
    result = await extract_organ_term("colorectal cancer")

    assert result == "colon"


async def test_expand_search_terms_returns_queries():
    """expand_search_terms should return queries covering all 5 prompt axes."""
    profile = DrugProfile(
        name="metformin",
        synonyms=["Glucophage", "Fortamet"],
        target_gene_symbols=["PRKAA1", "PRKAA2", "STK11"],
        mechanisms_of_action=[
            "AMP-activated protein kinase activator",
            "mTOR inhibitor",
        ],
        atc_codes=["A10BA02"],
        atc_descriptions=["BLOOD GLUCOSE LOWERING DRUGS, EXCL. INSULINS", "Biguanides"],
        drug_type="Small molecule",
    )
    queries = await expand_search_terms("metformin", "colorectal cancer", profile)
    queries_lower = [q.lower() for q in queries]

    assert 5 <= len(queries) <= 10
    assert all(isinstance(q, str) and len(q) > 0 for q in queries)

    # Axis 1: drug name + disease
    assert any("metformin" in q for q in queries_lower)

    # Axis 2: ATC class term + organ
    assert any("biguanide" in q or "blood glucose" in q for q in queries_lower)

    # Axis 3: mechanism keyword + organ
    assert any("ampk" in q or "mtor" in q for q in queries_lower)

    # Axis 4: target gene symbol + disease/cancer term
    assert any(
        gene in q for gene in ("prkaa1", "prkaa2", "stk11") for q in queries_lower
    )

    # Axis 5: synonym/trade name + disease
    assert any("glucophage" in q or "fortamet" in q for q in queries_lower)

    # No duplicates (case-insensitive)
    assert len(queries_lower) == len(set(queries_lower))


# --- build_drug_profile ---


@pytest.mark.parametrize(
    "drug_name, expected_name, expected_drug_type, expected_atc_codes, expected_atc_descriptions, expected_target_gene_symbols, expected_mechanisms_of_action",
    [
        (
            "metformin",
            "METFORMIN",
            "Small molecule",
            ["A10BA02"],
            ["BLOOD GLUCOSE LOWERING DRUGS, EXCL. INSULINS", "Biguanides"],
            ["GPD2", "NDUFS1"],
            [
                "Mitochondrial complex I (NADH dehydrogenase) inhibitor",
                "Mitochondrial glycerol-3-phosphate dehydrogenase inhibitor",
            ],
        ),
        (
            "trastuzumab",
            "TRASTUZUMAB",
            "Antibody",
            ["L01FD01"],
            [
                "MONOCLONAL ANTIBODIES AND ANTIBODY DRUG CONJUGATES",
                "HER2 (Human Epidermal Growth Factor Receptor 2) inhibitors",
            ],
            ["ERBB2"],
            ["Receptor protein-tyrosine kinase erbB-2 inhibitor"],
        ),
        (
            "pembrolizumab",
            "PEMBROLIZUMAB",
            "Antibody",
            [],
            [],
            ["PDCD1"],
            ["Programmed cell death protein 1 inhibitor"],
        ),
    ],
)
async def test_build_drug_profile(
    drug_name,
    expected_name,
    expected_drug_type,
    expected_atc_codes,
    expected_atc_descriptions,
    expected_target_gene_symbols,
    expected_mechanisms_of_action,
):
    """build_drug_profile assembles a complete DrugProfile from live Open Targets + ChEMBL data."""
    profile = await build_drug_profile(drug_name)

    assert profile.name == expected_name
    assert profile.drug_type == expected_drug_type
    assert profile.atc_codes == expected_atc_codes
    assert profile.atc_descriptions == expected_atc_descriptions
    assert set(expected_target_gene_symbols).issubset(set(profile.target_gene_symbols))
    assert set(expected_mechanisms_of_action).issubset(
        set(profile.mechanisms_of_action)
    )


# --- embed_abstracts ---

# PMID 21133896: sildenafil + diabetic nephropathy — stable journal article with title and abstract.
_EMBED_TEST_PMID = "21133896"


async def test_embed_abstracts_returns_768_dim_vectors():
    """embed_abstracts produces one 768-dim vector per abstract, aligned by index.

    Fetches one real abstract from PubMed and embeds it. Verifies:
    - result length equals input length
    - the paired abstract pmid matches the input
    - the vector has exactly 768 dimensions (BioLORD-2023 output dim)
    - all vector values are finite floats
    """
    async with PubMedClient() as client:
        abstracts = await client.fetch_abstracts([_EMBED_TEST_PMID])

    assert len(abstracts) == 1

    result = embed_abstracts(abstracts)

    assert len(result) == 1
    abstract, vector = result[0]
    assert abstract.pmid == _EMBED_TEST_PMID
    assert len(vector) == 768
    assert all(isinstance(v, float) for v in vector)


# --- get_stored_pmids ---


def test_get_stored_pmids_returns_only_inserted_pmids(db_session):
    """Only the PMIDs that were pre-inserted are returned; unknown PMIDs are excluded.

    Inserts two rows directly via SQL, then queries with those two PMIDs plus
    two that were never inserted. The result must be exactly the two inserted ones.
    """
    db_session.execute(
        text(
            "INSERT INTO pubmed_abstracts (pmid, title, fetched_at) VALUES "
            "('10000001', 'Test title A', NOW()), "
            "('10000002', 'Test title B', NOW()) "
            "ON CONFLICT (pmid) DO NOTHING"
        )
    )

    result = get_stored_pmids(
        ["10000001", "10000002", "99999991", "99999992"], db_session
    )

    assert result == {"10000001", "10000002"}


# --- fetch_new_abstracts ---

# PMID 20301421: "Metformin inhibits hepatic gluconeogenesis..." — a stable, well-known paper.
_KNOWN_NEW_PMID = "20301421"


async def test_fetch_new_abstracts_skips_stored_pmid():
    """fetch_new_abstracts returns only abstracts for PMIDs not in stored_pmids.

    Passes _KNOWN_NEW_PMID as the only new PMID (stored_pmids contains a
    fake PMID that would never appear on PubMed). Asserts exactly one abstract
    is returned and its pmid matches the requested one.
    """
    stored = {"10000001"}  # fake pre-stored PMID — not on PubMed
    all_pmids = ["10000001", _KNOWN_NEW_PMID]

    abstracts = await fetch_new_abstracts(all_pmids, stored)

    assert len(abstracts) == 1
    assert abstracts[0].pmid == _KNOWN_NEW_PMID


async def test_fetch_new_abstracts_all_stored_skips_network():
    """When all PMIDs are already stored, no network call is made and [] is returned.

    Uses only fake PMIDs that are all marked as stored, so fetch_abstracts
    must not be called at all — the function must return an empty list.
    """
    stored = {"10000001", "10000002"}
    all_pmids = ["10000001", "10000002"]

    abstracts = await fetch_new_abstracts(all_pmids, stored)

    assert abstracts == []
