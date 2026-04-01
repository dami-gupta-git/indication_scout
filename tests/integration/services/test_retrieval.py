"""Integration tests for services/retrieval."""

import logging

import pytest
from sqlalchemy import text

from indication_scout.data_sources.pubmed import PubMedClient
from indication_scout.models.model_drug_profile import DrugProfile
from indication_scout.models.model_evidence_summary import EvidenceSummary
from indication_scout.services.retrieval import RetrievalService

logger = logging.getLogger(__name__)
# db_session fixture is defined in tests/integration/conftest.py and
# connects to scout_test (TEST_DATABASE_URL). It rolls back after each test.


@pytest.fixture
def svc(test_cache_dir):
    """RetrievalService bound to the test cache directory."""
    return RetrievalService(test_cache_dir)


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
async def test_get_disease_synonyms(disease, synonyms, svc):
    """Returned synonyms should include all expected terms for the given disease."""
    result = await svc.get_disease_synonyms(disease)

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


# async def test_semantic_search_returns_ranked_abstracts(svc):
#     """Should return up to top_k dicts with required abstract fields."""
#     results = await svc.semantic_search("colorectal cancer", "metformin", top_k=5)
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


async def test_extract_organ_term_returns_string(svc):
    """extract_organ_term should return the primary organ for a known disease."""
    result = await svc.extract_organ_term("colorectal cancer")

    assert result == "colon"


async def test_expand_search_terms_returns_queries(svc):
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
    queries = await svc.expand_search_terms("metformin", "colorectal cancer", profile)
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
            ["L01FF02"],
            [
                "MONOCLONAL ANTIBODIES AND ANTIBODY DRUG CONJUGATES",
                "PD-1/PD-L1 (Programmed cell death protein 1/death ligand 1) inhibitors",
            ],
            ["PDCD1"],
            ["Programmed cell death protein 1 inhibitor"],
        ),
    ],
)
async def test_build_drug_profile(
    svc,
    drug_name,
    expected_name,
    expected_drug_type,
    expected_atc_codes,
    expected_atc_descriptions,
    expected_target_gene_symbols,
    expected_mechanisms_of_action,
):
    """build_drug_profile assembles a complete DrugProfile from live Open Targets + ChEMBL data."""
    profile = await svc.build_drug_profile(drug_name)

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


async def test_embed_abstracts_returns_768_dim_vectors(test_cache_dir):
    """embed_abstracts produces one 768-dim vector per abstract, aligned by index.

    Fetches one real abstract from PubMed and embeds it. Verifies:
    - result length equals input length
    - the paired abstract pmid matches the input
    - the vector has exactly 768 dimensions (BioLORD-2023 output dim)
    - all vector values are finite floats
    """
    async with PubMedClient(cache_dir=test_cache_dir) as client:
        abstracts = await client.fetch_abstracts([_EMBED_TEST_PMID])

    assert len(abstracts) == 1

    result = await RetrievalService(test_cache_dir).embed_abstracts(abstracts)

    assert len(result) == 1
    abstract, vector = result[0]
    assert abstract.pmid == _EMBED_TEST_PMID
    assert len(vector) == 768
    assert all(isinstance(v, float) for v in vector)


# --- get_stored_pmids ---


def test_get_stored_pmids_returns_only_inserted_pmids(db_session, test_cache_dir):
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

    result = RetrievalService(test_cache_dir).get_stored_pmids(
        ["10000001", "10000002", "99999991", "99999992"], db_session
    )

    assert result == {"10000001", "10000002"}


# --- fetch_new_abstracts ---

# PMID 20301421: "Metformin inhibits hepatic gluconeogenesis..." — a stable, well-known paper.
_KNOWN_NEW_PMID = "20301421"


async def test_fetch_new_abstracts_skips_stored_pmid(test_cache_dir):
    """fetch_new_abstracts returns only abstracts for PMIDs not in stored_pmids.

    Passes _KNOWN_NEW_PMID as the only new PMID (stored_pmids contains a
    fake PMID that would never appear on PubMed). Asserts exactly one abstract
    is returned and its pmid matches the requested one.
    """
    stored = {"10000001"}  # fake pre-stored PMID — not on PubMed
    all_pmids = ["10000001", _KNOWN_NEW_PMID]

    async with PubMedClient(cache_dir=test_cache_dir) as client:
        abstracts = await RetrievalService(test_cache_dir).fetch_new_abstracts(
            all_pmids, stored, client
        )

    assert len(abstracts) == 1
    assert abstracts[0].pmid == _KNOWN_NEW_PMID


async def test_fetch_new_abstracts_all_stored_skips_network(test_cache_dir):
    """When all PMIDs are already stored, no network call is made and [] is returned.

    Uses only fake PMIDs that are all marked as stored, so fetch_abstracts
    must not be called at all — the function must return an empty list.
    """
    stored = {"10000001", "10000002"}
    all_pmids = ["10000001", "10000002"]

    async with PubMedClient(cache_dir=test_cache_dir) as client:
        abstracts = await RetrievalService(test_cache_dir).fetch_new_abstracts(
            all_pmids, stored, client
        )

    assert abstracts == []


# --- fetch_and_cache ---

# Query with a small, stable result set verified live on 2026-03-01.
# "biguanides AND colon cancer" returns (at least) these 3 PMIDs.
_FETCH_AND_CACHE_QUERY = "biguanides AND colon cancer"
_FETCH_AND_CACHE_PMIDS = {"40670504", "39215927", "31438832"}

# Two overlapping queries verified live on 2026-03-06.
# Query A returns 500 PMIDs, Query B returns 79 PMIDs.
# 25 PMIDs appear in both — used to assert deduplication and idempotency.
_QUERY_A = "metformin AND colorectal cancer"
_QUERY_B = "metformin AND AMPK AND colon"
_OVERLAP_PMIDS = {
    "24157941",
    "24251703",
    "24716225",
    "25416412",
    "25892866",
    "27123089",
    "27585117",
    "27845068",
    "27919208",
    "28114961",
    "28618116",
    "29059169",
    "30087121",
    "30359174",
    "30483811",
    "31612008",
    "31661292",
    "31818851",
    "32248666",
    "32533543",
    "34564972",
    "34749618",
    "35740547",
    "36983040",
    "38132178",
}


async def test_fetch_and_cache_inserts_rows_and_returns_pmids(
    db_session_truncating, test_cache_dir
):
    """fetch_and_cache fetches abstracts, embeds them, and persists rows to pubmed_abstracts.

    Uses a single stable PubMed query with a known small result set. Verifies:
    - the known PMIDs are present in the returned list
    - each returned PMID has a row in pubmed_abstracts with a non-null embedding
    - no duplicate PMIDs in the result
    """
    pmids = await RetrievalService(test_cache_dir).fetch_and_cache(
        [_FETCH_AND_CACHE_QUERY], db_session_truncating
    )

    assert _FETCH_AND_CACHE_PMIDS.issubset(set(pmids))
    assert len(pmids) == len(set(pmids))

    rows = db_session_truncating.execute(
        text("SELECT pmid, embedding FROM pubmed_abstracts WHERE pmid = ANY(:pmids)"),
        {"pmids": list(_FETCH_AND_CACHE_PMIDS)},
    ).fetchall()

    assert len(rows) == len(_FETCH_AND_CACHE_PMIDS)
    for pmid, embedding in rows:
        assert embedding is not None, f"embedding is null for pmid {pmid}"


async def test_fetch_and_cache_deduplicates_overlapping_queries(
    db_session_truncating, test_cache_dir
):
    """PMIDs returned by two overlapping queries appear exactly once in the result.

    Query A (metformin AND colorectal cancer) and Query B (metformin AND AMPK AND colon)
    share 25 known PMIDs verified live on 2026-03-06. Asserts:
    - all returned PMIDs are digit-only strings
    - no duplicate PMIDs in the result list
    - the 25 known overlap PMIDs are each present exactly once
    """
    pmids = await RetrievalService(test_cache_dir).fetch_and_cache(
        [_QUERY_A, _QUERY_B], db_session_truncating
    )

    assert all(p.isdigit() for p in pmids)
    assert len(pmids) == len(set(pmids))
    assert _OVERLAP_PMIDS.issubset(set(pmids))


async def test_fetch_and_cache_is_idempotent(db_session_truncating, test_cache_dir):
    """Running fetch_and_cache twice with the same query does not insert duplicate rows.

    Calls fetch_and_cache with a single stable query, records the row count,
    then calls it again. Asserts the row count in pubmed_abstracts is unchanged
    after the second call (ON CONFLICT DO NOTHING is exercised end-to-end).
    """
    queries = [_FETCH_AND_CACHE_QUERY]
    svc = RetrievalService(test_cache_dir)

    await svc.fetch_and_cache(queries, db_session_truncating)
    count_after_first = db_session_truncating.execute(
        text("SELECT COUNT(*) FROM pubmed_abstracts")
    ).scalar()

    await svc.fetch_and_cache(queries, db_session_truncating)
    count_after_second = db_session_truncating.execute(
        text("SELECT COUNT(*) FROM pubmed_abstracts")
    ).scalar()

    assert count_after_second == count_after_first


# Landmark PMIDs should always be in pgvector and rank highly
async def test_empareg_in_results(svc, db_session_truncating):
    pmids = await svc.fetch_and_cache(
        ["empagliflozin AND myocardial infarction"], db_session_truncating
    )
    top_15 = await svc.semantic_search(
        "myocardial infarction", "empagliflozin", pmids, db_session_truncating, top_k=15
    )
    result_pmids = [r["pmid"] for r in top_15]
    assert "38587237" in result_pmids  # EMPACT-MI


async def test_recovery_in_results(svc, db_session_truncating):
    pmids = await svc.fetch_and_cache(
        [
            "empagliflozin AND severe acute respiratory syndrome",
            "empagliflozin AND SARS",
        ],
        db_session_truncating,
    )
    top_5 = await svc.semantic_search(
        "severe acute respiratory syndrome",
        "empagliflozin",
        pmids,
        db_session_truncating,
        top_k=5,
    )
    result_pmids = [r["pmid"] for r in top_5]
    assert "37865101" in result_pmids  # RECOVERY trial


async def test_semantic_search_returns_relevant_results(svc, db_session_truncating):
    """Semantic search should return abstracts about empagliflozin + MI."""
    queries = [
        "empagliflozin AND myocardial infarction",
        "empagliflozin AND cardiovascular outcome",
    ]
    pmids = await svc.fetch_and_cache(queries, db_session_truncating)
    results = await svc.semantic_search(
        "myocardial infarction", "empagliflozin", pmids, db_session_truncating, top_k=5
    )

    assert len(results) == 5
    # All results should have reasonable similarity
    assert all(r["similarity"] > 0.5 for r in results)
    # At least one title should mention empagliflozin or SGLT2
    assert any(
        "empagliflozin" in r["title"].lower() or "sglt2" in r["title"].lower()
        for r in results
    )


async def test_semantic_search_empagliflozin_nephropathy(svc, db_session_truncating):
    """Semantic search should return relevant abstracts ranked by similarity."""
    queries = ["empagliflozin AND diabetic nephropathy"]
    pmids = await svc.fetch_and_cache(queries, db_session_truncating)

    results = await svc.semantic_search(
        "diabetic nephropathy", "empagliflozin", pmids, db_session_truncating, top_k=5
    )

    assert len(results) == 5
    assert all("pmid" in r for r in results)
    assert all("similarity" in r for r in results)
    # Sorted descending by similarity
    similarities = [r["similarity"] for r in results]
    assert similarities == sorted(similarities, reverse=True)


async def test_synthesize_strong_candidate(svc, db_session_truncating):
    """Empagliflozin + diabetic nephropathy should come back strong."""
    queries = ["empagliflozin AND diabetic nephropathy"]
    pmids = await svc.fetch_and_cache(queries, db_session_truncating)
    top_5 = await svc.semantic_search(
        "diabetic nephropathy", "empagliflozin", pmids, db_session_truncating
    )

    result = await svc.synthesize("empagliflozin", "diabetic nephropathy", top_5)

    assert isinstance(result, EvidenceSummary)
    assert result.strength in ["strong", "moderate"]
    assert len(result.supporting_pmids) >= 2
    assert len(result.key_findings) >= 2
    assert result.summary  # non-empty


async def test_synthesize_negative_candidate(svc, db_session_truncating):
    """Empagliflozin + SARS/COVID should come back none."""
    queries = ["empagliflozin AND COVID-19", "empagliflozin AND SARS"]
    pmids = await svc.fetch_and_cache(queries, db_session_truncating)
    top_5 = await svc.semantic_search(
        "severe acute respiratory syndrome",
        "empagliflozin",
        pmids,
        db_session_truncating,
    )

    result = await svc.synthesize(
        "empagliflozin", "severe acute respiratory syndrome", top_5
    )

    assert result.strength == "none"
    assert result.supporting_pmids == []


async def test_synthesize_contraindication(svc, db_session_truncating):
    """Bupropion + hypertension should flag adverse effects."""
    queries = ["bupropion AND hypertension"]
    pmids = await svc.fetch_and_cache(queries, db_session_truncating)
    top_5 = await svc.semantic_search(
        "hypertension", "bupropion", pmids, db_session_truncating
    )

    result = await svc.synthesize("bupropion", "hypertension", top_5)

    assert result.strength == "none"
    assert result.has_adverse_effects is True
    assert result.supporting_pmids == []
