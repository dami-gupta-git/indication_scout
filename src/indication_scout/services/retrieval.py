"""Retrieval service: PubMed fetch/embed/cache and semantic search via pgvector."""

import logging
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from indication_scout.constants import CACHE_TTL, DEFAULT_CACHE_DIR
from indication_scout.data_sources.chembl import ChEMBLClient
from indication_scout.data_sources.open_targets import OpenTargetsClient
from indication_scout.data_sources.pubmed import PubMedClient
from indication_scout.models.model_drug_profile import DrugProfile
from indication_scout.models.model_pubmed_abstract import PubmedAbstract
from indication_scout.sqlalchemy.pubmed_abstracts import PubmedAbstracts
from indication_scout.services.embeddings import embed
from indication_scout.services.llm import parse_llm_response, query_small_llm
from indication_scout.utils.cache import cache_get, cache_set

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


async def build_drug_profile(drug_name: str) -> DrugProfile:
    """Fetch drug + target data from Open Targets, enrich with ATC descriptions from ChEMBL,
    and return a DrugProfile ready for use in search term expansion.

    Args:
        drug_name: Common drug name (e.g. "metformin").

    Returns:
        DrugProfile with all fields populated. atc_descriptions will be [] if the drug
        has no ATC classifications.
    """
    async with OpenTargetsClient() as open_targets_client:
        rich = await open_targets_client.get_rich_drug_data(drug_name)

    atc_descriptions = []
    if rich.drug.atc_classifications:
        async with ChEMBLClient() as chembl_client:
            for code in rich.drug.atc_classifications:
                atc_descriptions.append(await chembl_client.get_atc_description(code))

    return DrugProfile.from_rich_drug_data(rich, atc_descriptions)


def get_stored_pmids(pmids: list[str], db: Session) -> set[str]:
    """Return the subset of the given PMIDs that already exist in pubmed_abstracts.

    Used by fetch_and_cache to avoid re-fetching and re-embedding abstracts that
    are already in pgvector. A single bulk SELECT is used rather than one query
    per PMID to keep DB round-trips to a minimum.

    Args:
        pmids: Candidate PMIDs to check.
        db: Active SQLAlchemy session.

    Returns:
        Set of PMIDs that are present in the pubmed_abstracts table.
    """
    if not pmids:
        return set()

    rows = db.execute(
        text("SELECT pmid FROM pubmed_abstracts WHERE pmid = ANY(:pmids)"),
        {"pmids": pmids},
    ).fetchall()

    return {row[0] for row in rows}


async def fetch_new_abstracts(
    all_pmids: list[str], stored_pmids: set[str]
) -> list[PubmedAbstract]:
    """Fetch PubMed abstracts for PMIDs not already in the database.

    Computes the set difference between all_pmids and stored_pmids, then
    calls PubMedClient.fetch_abstracts on only the new ones. If there are
    no new PMIDs the network call is skipped entirely.

    Args:
        all_pmids: Full list of PMIDs from a PubMed search result.
        stored_pmids: PMIDs already present in pubmed_abstracts (from get_stored_pmids).

    Returns:
        List of PubmedAbstract objects for each newly fetched PMID.
        Empty list if all PMIDs were already stored.
    """
    new_pmids = [p for p in all_pmids if p not in stored_pmids]
    if not new_pmids:
        logger.debug("All %d PMIDs already stored; skipping fetch", len(all_pmids))
        return []

    logger.debug("Fetching %d new abstracts from PubMed", len(new_pmids))
    async with PubMedClient() as client:
        return await client.fetch_abstracts(new_pmids)


def embed_abstracts(
    abstracts: list[PubmedAbstract],
) -> list[tuple[PubmedAbstract, list[float]]]:
    """Embed a list of PubMed abstracts using BioLORD-2023.

    Builds embed text as "<title>. <abstract>" for each abstract and calls
    embed() in a single batch. Returns (abstract, vector) pairs aligned by index.

    Args:
        abstracts: Abstracts to embed.

    Returns:
        List of (PubmedAbstract, embedding vector) pairs in the same order as input.
        Empty list if abstracts is empty (embed() is not called).
    """
    if not abstracts:
        return []

    texts = [f"{a.title}. {a.abstract or ''}" for a in abstracts]
    vectors = embed(texts)
    return list(zip(abstracts, vectors))


def insert_abstracts(
    pairs: list[tuple[PubmedAbstract, list[float]]],
    db: Session,
) -> None:
    """Bulk-insert (abstract, embedding) pairs into pubmed_abstracts.

    Uses INSERT ... ON CONFLICT DO NOTHING so re-running with already-stored
    PMIDs is safe and idempotent. Does nothing when pairs is empty.

    Args:
        pairs: Output of embed_abstracts — (PubmedAbstract, vector) tuples.
        db: Active SQLAlchemy session.
    """
    if not pairs:
        return

    rows = [
        {
            "pmid": abstract.pmid,
            "title": abstract.title or "",
            "abstract": abstract.abstract,
            "authors": abstract.authors or [],
            "journal": abstract.journal,
            "pub_date": abstract.pub_date,
            "mesh_terms": abstract.mesh_terms or [],
            "embedding": vector,
        }
        for abstract, vector in pairs
    ]

    stmt = insert(PubmedAbstracts).values(rows).on_conflict_do_nothing(index_elements=["pmid"])
    db.execute(stmt)
    db.commit()
    logger.debug("Inserted %d abstracts into pubmed_abstracts", len(rows))


async def fetch_and_cache(queries: list[str], db: Session) -> list[str]:
    """Hit PubMed for each query, fetch new abstracts, embed, cache in pgvector.

    For each keyword query:
      1. Search PubMed → up to 500 PMIDs
      2. Filter to PMIDs not already in pgvector
      3. Fetch abstracts for new PMIDs
      4. Embed each abstract with BioLORD-2023
      5. Bulk INSERT into pgvector (ON CONFLICT DO NOTHING)

    Returns the deduplicated union of all PMIDs across all queries.

    Args:
        queries: PubMed keyword queries (e.g. from expand_search_terms).
        db: Active SQLAlchemy session.

    Returns:
        Deduplicated list of all PMIDs (cached + newly inserted).
    """
    all_pmids: list[str] = []

    async with PubMedClient() as client:
        for query in queries:
            pmids = await client.search(query, max_results=500)
            stored = get_stored_pmids(pmids, db)
            new_abstracts = await fetch_new_abstracts(pmids, stored)
            pairs = embed_abstracts(new_abstracts)
            insert_abstracts(pairs, db)
            all_pmids.extend(pmids)

    return list(dict.fromkeys(all_pmids))


async def semantic_search(disease: str, drug: str, top_k: int = 20) -> list[dict]:
    """Create a query from drug + disease, embed it, search pgvector, return ranked abstracts.
    [
        {"pmid": "29734553", "title": "Metformin suppresses colorectal...", "abstract": "...", "similarity": 0.89},
        {"pmid": "31245678", "title": "AMPK activation in colon...", "abstract": "...", "similarity": 0.85},
        {"pmid": "30198432", "title": "Biguanide compounds inhibit...", "abstract": "...", "similarity": 0.82},
    ...
    ]
    """
    raise NotImplementedError


def synthesize(drug: str, disease: str, top_5_abstracts: list[str]) -> dict:
    """get back a structured evidence summary with PMIDs
    Summarize the evidence for Metformin treating colorectal cancer based on these papers" and it returns a structured summary
     — study count, study types, strength assessment, key findings, and the PMIDs it drew from.
    That evidence summary is the final output of the RAG pipeline for one drug-disease pair.
    """
    raise NotImplementedError


async def extract_organ_term(disease_name: str) -> str:
    """Return the primary organ or tissue for a disease name via a small LLM call."""
    cached = cache_get("organ_term", {"disease_name": disease_name}, DEFAULT_CACHE_DIR)
    if cached is not None:
        logger.debug("Cache hit for organ_term: %s", disease_name)
        return cached

    template = (_PROMPTS_DIR / "extract_organ_term.txt").read_text()
    prompt = template.format(disease_name=disease_name)
    result = await query_small_llm(prompt)
    organ_term = result.strip()

    cache_set(
        "organ_term",
        {"disease_name": disease_name},
        organ_term,
        DEFAULT_CACHE_DIR,
        ttl=CACHE_TTL,
    )
    logger.debug("Extracted organ term '%s' for disease '%s'", organ_term, disease_name)
    return organ_term


async def expand_search_terms(
    drug_name: str, disease_name: str, drug_profile: DrugProfile
) -> list[str]:
    """Use LLM to generate diverse PubMed search queries from a drug-disease pair."""
    cached = cache_get(
        "expand_search_terms",
        {"drug_name": drug_name, "disease_name": disease_name},
        DEFAULT_CACHE_DIR,
    )
    if cached is not None:
        logger.debug(
            "Cache hit for expand_search_terms: %s / %s", drug_name, disease_name
        )
        return cached

    organ_term = await extract_organ_term(disease_name)

    template = (_PROMPTS_DIR / "expand_search_terms.txt").read_text()
    prompt = template.format(
        drug_name=drug_name,
        disease_name=disease_name,
        organ_term=organ_term,
        synonyms=", ".join(drug_profile.synonyms),
        target_gene_symbols=", ".join(drug_profile.target_gene_symbols),
        mechanisms_of_action=", ".join(drug_profile.mechanisms_of_action),
        atc_codes=", ".join(drug_profile.atc_codes),
        atc_descriptions=", ".join(drug_profile.atc_descriptions),
        drug_type=drug_profile.drug_type,
    )

    text = await query_small_llm(prompt)
    raw: list[str] = parse_llm_response(text)

    # Case-normalised dedup: lowercase+strip as key, preserve original casing
    seen: dict[str, str] = {}
    for term in raw:
        key = term.lower().strip()
        if key not in seen:
            seen[key] = term
    deduped = list(seen.values())

    cache_set(
        "expand_search_terms",
        {"drug_name": drug_name, "disease_name": disease_name},
        deduped,
        DEFAULT_CACHE_DIR,
        ttl=CACHE_TTL,
    )
    logger.debug(
        "expand_search_terms returned %d queries for %s / %s",
        len(deduped),
        drug_name,
        disease_name,
    )
    return deduped


async def get_disease_synonyms(disease: str) -> list[str]:
    template = (_PROMPTS_DIR / "disease_synonyms.txt").read_text()
    prompt = template.format(disease=disease)

    text = await query_small_llm(prompt)
    return parse_llm_response(text)
