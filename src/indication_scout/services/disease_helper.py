"""
Disease term normalizer for PubMed search.

Converts raw disease terms from Open Targets (e.g., "narcolepsy-cataplexy syndrome")
into normalized PubMed-friendly search terms (e.g., "narcolepsy").

Strategy: LLM normalize → verify with PubMed count → cache everything.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import TypedDict

from indication_scout.constants import (
    BROADENING_BLOCKLIST,
    DEFAULT_CACHE_DIR,
)
from indication_scout.data_sources.base_client import DataSourceError
from indication_scout.data_sources.pubmed import PubMedClient
from indication_scout.services.llm import query_small_llm
from indication_scout.utils.cache import cache_get, cache_set

logger = logging.getLogger(__name__)

MIN_RESULTS = 3  # Minimum PubMed hits to consider a term useful

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


class MergeResult(TypedDict):
    merge: dict[str, list[str]]
    remove: list[str]


# ── LLM Normalize ───────────────────────────────────────────────────────────


async def llm_normalize_disease(raw_term: str) -> str:
    """
    Reduce disease to overarching

    Use an LLM to convert a disease term into a PubMed search term.

    Example:
        "narcolepsy-cataplexy syndrome"         → "narcolepsy"
        "renal tubular dysgenesis"              → "kidney disease"
        "CML"                                   → "chronic myeloid leukemia"
    """
    cached = cache_get("disease_norm", {"raw_term": raw_term}, DEFAULT_CACHE_DIR)
    if cached is not None:
        return cached

    prompt = (
        (_PROMPTS_DIR / "normalize_disease.txt").read_text().format(raw_term=raw_term)
    )
    response = await query_small_llm(prompt)
    normalized = response.strip().strip('"').strip("'")

    cache_set("disease_norm", {"raw_term": raw_term}, normalized, DEFAULT_CACHE_DIR)
    return normalized


async def llm_normalize_disease_batch(raw_terms: list[str]) -> dict[str, str]:
    """Normalize multiple disease terms in a single LLM call.

    Checks cache for each term first, batches only the cache misses into one
    LLM call, then caches the new results individually.

    Args:
        raw_terms: List of raw disease terms to normalize.

    Returns:
        Dict mapping each raw term to its normalized form.
    """
    results: dict[str, str] = {}
    uncached: list[str] = []

    for term in raw_terms:
        cached = cache_get("disease_norm", {"raw_term": term}, DEFAULT_CACHE_DIR)
        if cached is not None:
            results[term] = cached
        else:
            uncached.append(term)

    if not uncached:
        return results

    prompt = (
        (_PROMPTS_DIR / "normalize_disease_batch.txt")
        .read_text()
        .format(raw_terms=json.dumps(uncached))
    )
    response = await query_small_llm(prompt)
    cleaned = response.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.rsplit("```", 1)[0].strip()

    try:
        batch_results: dict[str, str] = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(
            "llm_normalize_disease_batch: failed to parse LLM response: %s\n"
            "Response was: %s",
            e,
            response,
        )
        # Fall back to individual calls for the uncached terms
        individual = await asyncio.gather(
            *[llm_normalize_disease(term) for term in uncached]
        )
        for term, normalized in zip(uncached, individual):
            results[term] = normalized
        return results

    for term in uncached:
        normalized = batch_results.get(term)
        if normalized is None:
            logger.warning(
                "Batch normalization missing term '%s', falling back to individual call",
                term,
            )
            normalized = await llm_normalize_disease(term)
        else:
            normalized = normalized.strip().strip('"').strip("'")
            cache_set("disease_norm", {"raw_term": term}, normalized, DEFAULT_CACHE_DIR)
        results[term] = normalized

    return results


async def merge_duplicate_diseases(
    diseases: list[str], drug_indications: list[str]
) -> MergeResult:
    prompt = (
        (_PROMPTS_DIR / "merge_diseases.txt")
        .read_text()
        .format(disease_names=diseases, drug_indications=drug_indications)
    )
    response = await query_small_llm(prompt)
    cleaned = response.strip()
    # Strip markdown code fences if present
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.rsplit("```", 1)[0].strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(
            "merge_duplicate_diseases: failed to parse LLM response: %s\nResponse was: %s",
            e,
            response,
        )
        return {"merge": {}, "remove": []}


# ── PubMed Count ─────────────────────────────────────────────────────────────


async def pubmed_count(query: str) -> int:
    """Return the number of PubMed results for a query string."""
    cached = cache_get("pubmed_count", {"query": query}, DEFAULT_CACHE_DIR)
    if cached is not None:
        return cached

    try:
        async with PubMedClient(cache_dir=DEFAULT_CACHE_DIR) as client:
            count = await client.get_count(query)
    except DataSourceError as e:
        logger.warning("PubMed count failed for '%s': %s", query, e)
        return 0

    cache_set("pubmed_count", {"query": query}, count, DEFAULT_CACHE_DIR)
    return count


# ── Main Orchestrator ────────────────────────────────────────────────────────


async def normalize_for_pubmed(raw_term: str, drug_name: str | None = None) -> str:
    """
    Short set of queries
    e.g. hepatocellular carcinoma -> 'liver cancer OR hepatocellular carcinoma'

    Normalize a raw disease term from Open Targets into a list of PubMed queries.

    Strategy:
        1. Check cache
        2. LLM normalize
        3. If drug_name provided, verify with PubMed count (drug + disease)
        4. If too few results, ask LLM to generalize further
        5. Cache and return

    Args:
        raw_term:  Disease term from Open Targets (e.g., "narcolepsy-cataplexy syndrome")
        drug_name: Optional drug being investigated (e.g., "bupropion") — used for verification.
                   If None, skips PubMed count verification and returns LLM result directly.

    Returns:
        Normalized PubMed search term (e.g., "narcolepsy")
    """
    # Step 1: LLM normalize
    normalized = await llm_normalize_disease(raw_term)

    # Reject if LLM collapsed to a blocklisted over-generic term
    normalized_terms = {t.strip().lower() for t in normalized.split("OR")}
    if normalized_terms <= BROADENING_BLOCKLIST:
        logger.info(
            f"Rejected over-broad normalization '{normalized}' for '{raw_term}', keeping raw term"
        )
        normalized = raw_term

    # Step 2: If drug provided, verify with PubMed count
    if drug_name:
        count = await pubmed_count(f"{drug_name} AND ({normalized})")

        if count < MIN_RESULTS:
            # Step 3: Ask LLM to generalize further
            broader = await llm_normalize_disease(
                f"{normalized} (generalize to a broader disease category)"
            )
            broader_terms = {t.strip().lower() for t in broader.split("OR")}
            if broader_terms & BROADENING_BLOCKLIST:
                logger.info(
                    f"Rejected over-broad fallback '{broader}' for '{normalized}'"
                )
            else:
                broader_count = await pubmed_count(f"{drug_name} AND ({broader})")
                if broader_count >= MIN_RESULTS:
                    normalized = broader

    logger.info(f"Normalized '{raw_term}' → '{normalized}'")

    return normalized


# ── Batch Convenience ────────────────────────────────────────────────────────


async def normalize_batch(
    terms: list[str], drug_name: str | None = None
) -> dict[str, str]:
    """
    Normalize a list of disease terms. Returns dict of raw → normalized.
    Processes sequentially to respect NCBI rate limits (3 req/sec without API key,
    10 req/sec with one).
    """
    results = {}
    for term in terms:
        results[term] = await normalize_for_pubmed(term, drug_name)
        await asyncio.sleep(0.35)  # Stay under NCBI rate limit

    return results
