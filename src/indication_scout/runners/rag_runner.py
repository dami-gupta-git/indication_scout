"""End-to-end RAG pipeline runner for drug repurposing evidence retrieval."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import wandb
from sqlalchemy.orm import Session

from indication_scout.models.model_drug_profile import DrugProfile

from indication_scout.constants import (
    DEFAULT_CACHE_DIR,
    RAG_DISEASE_CONCURRENCY,
    RAG_LLM_CONCURRENCY,
    RAG_PUBMED_CONCURRENCY,
)
from indication_scout.db.session import get_db
from indication_scout.models.model_evidence_summary import EvidenceSummary
from indication_scout.services.retrieval import RetrievalService
from indication_scout.utils.wandb_utils import wandb_run

logger = logging.getLogger(__name__)


async def _process_disease(
    disease: str,
    drug_name: str,
    drug_profile: DrugProfile,
    svc: RetrievalService,
    db: Session,
    sem_llm: asyncio.Semaphore,
    sem_pubmed: asyncio.Semaphore,
) -> tuple[str, EvidenceSummary]:
    """Run the RAG pipeline for a single disease indication.

    Acquires per-resource semaphores to limit concurrent access to
    rate-limited APIs and the shared embedding model.
    """
    logger.info("run_rag: processing disease '%s'", disease)

    t0 = time.perf_counter()
    async with sem_llm:
        queries = await svc.expand_search_terms(drug_name, disease, drug_profile)
    logger.info(
        "run_rag: expand_search_terms took %.2fs, %d queries for '%s'",
        time.perf_counter() - t0,
        len(queries),
        disease,
    )

    t0 = time.perf_counter()
    async with sem_pubmed:
        pmids = await svc.fetch_and_cache(queries, db)
    logger.info(
        "run_rag: fetch_and_cache took %.2fs, %d PMIDs for '%s'",
        time.perf_counter() - t0,
        len(pmids),
        disease,
    )

    t0 = time.perf_counter()
    top_abstracts = await svc.semantic_search(disease, drug_name, pmids, db)
    logger.info(
        "run_rag: semantic_search took %.2fs, %d abstracts for '%s'",
        time.perf_counter() - t0,
        len(top_abstracts),
        disease,
    )

    t0 = time.perf_counter()
    async with sem_llm:
        evidence = await svc.synthesize(drug_name, disease, top_abstracts)
    logger.info(
        "run_rag: synthesize took %.2fs, strength=%s for '%s'",
        time.perf_counter() - t0,
        evidence.strength,
        disease,
    )

    logger.info("  Strength: %s", evidence.strength)
    logger.info("  Adverse: %s", evidence.has_adverse_effects)
    logger.info("  Summary: %s...", evidence.summary[:100])

    return disease, evidence


# @wandb_run(project="indication-scout")
async def run_rag(
    drug_name: str, db: Session, cache_dir: Path = DEFAULT_CACHE_DIR
) -> dict[str, EvidenceSummary]:
    """Run the full RAG pipeline for a drug across its top competitor indications.

    Disease indications are processed concurrently (up to RAG_DISEASE_CONCURRENCY
    at a time), with per-resource semaphores gating access to the LLM API and
    PubMed API to stay within rate limits.

    Steps for each disease indication:
      1. get_drug_competitors  — fetch top 15 disease indications from Open Targets
      2. build_drug_profile    — fetch drug/target data from Open Targets + ChEMBL
      3. expand_search_terms   — LLM generates diverse PubMed queries
      4. fetch_and_cache       — search PubMed, embed new abstracts, store in pgvector
      5. semantic_search       — cosine similarity search for top-k abstracts
      6. synthesize            — LLM synthesizes evidence into EvidenceSummary

    Args:
        drug_name: Common drug name (e.g. "metformin").
        db: Active SQLAlchemy session connected to pgvector DB.
        cache_dir: Directory for file-based caching. Defaults to DEFAULT_CACHE_DIR.

    Returns:
        Dict mapping disease name to EvidenceSummary for each indication.
    """
    logger.info("run_rag: starting pipeline for drug '%s'", drug_name)

    svc = RetrievalService(cache_dir)

    top_15 = await svc.get_drug_competitors(drug_name)

    logger.info(
        "run_rag: %d disease indications found for \n%s",
        len(top_15),
        ", ".join(top_15.keys()),
    )

    drug_profile = await svc.build_drug_profile(drug_name)
    logger.info(
        "run_rag: drug profile built — %d targets",
        len(drug_profile.target_gene_symbols),
    )

    sem_disease = asyncio.Semaphore(RAG_DISEASE_CONCURRENCY)
    sem_llm = asyncio.Semaphore(RAG_LLM_CONCURRENCY)
    sem_pubmed = asyncio.Semaphore(RAG_PUBMED_CONCURRENCY)

    async def _bounded(disease: str) -> tuple[str, EvidenceSummary]:
        async with sem_disease:
            return await _process_disease(
                disease, drug_name, drug_profile, svc, db, sem_llm, sem_pubmed
            )

    pairs = await asyncio.gather(*[_bounded(disease) for disease in top_15])

    results: dict[str, EvidenceSummary] = dict(pairs)

    # searches_table = wandb.Table(columns=["drug", "disease", "avg_similarity"]) if wandb.run else None
    # if wandb.run and searches_table is not None:
    #     wandb.log({"searches": searches_table})

    ranking = {"strong": 3, "moderate": 2, "weak": 1, "none": 0}
    sorted_results = sorted(
        results.items(), key=lambda x: ranking[x[1].strength], reverse=True
    )

    logger.info("=== Final Ranking for %s ===", drug_name)
    for disease, summary in sorted_results:
        flag = "ADVERSE" if summary.has_adverse_effects else "OK"
        logger.info("  [%s] %s: %s", flag, disease, summary.strength)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def _main() -> None:
        db = next(get_db())
        try:
            results = await run_rag("duloxetine", db)
            for disease, evidence in results.items():
                logger.info("\n=== %s ===", disease)
                logger.info(evidence.model_dump_json(indent=2))
        finally:
            db.close()

    asyncio.run(_main())
