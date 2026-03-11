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

import httpx

from indication_scout.constants import (
    BROADENING_BLOCKLIST,
    DEFAULT_CACHE_DIR,
    NCBI_BASE_URL,
)
from indication_scout.services.llm import (
    query_small_llm,
)  # Adjust import path as needed
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

    url = f"{NCBI_BASE_URL}/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": 0,
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            logger.warning(f"PubMed count failed for '{query}': {e}")
            return 0

    count = int(data.get("esearchresult", {}).get("count", 0))
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
