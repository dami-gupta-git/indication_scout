"""
Disease term normalizer for PubMed search.

Converts raw disease terms from Open Targets (e.g., "narcolepsy-cataplexy syndrome")
into normalized PubMed-friendly search terms (e.g., "narcolepsy").

Strategy: LLM normalize → verify with PubMed count → cache everything.
"""

import asyncio
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, TypedDict

import aiohttp

from indication_scout.config import get_settings
from indication_scout.constants import (
    BROADENING_BLOCKLIST,
    DEFAULT_CACHE_DIR,
    MESH_RESOLVER_MAX_CONCURRENT,
    MESH_RESOLVER_TTL_SECONDS,
    NCBI_ESEARCH_URL,
    NCBI_ESUMMARY_URL,
)
from indication_scout.data_sources.base_client import (
    DataSourceError,
    log_data_source_failure,
)
from indication_scout.data_sources.pubmed import PubMedClient
from indication_scout.services.llm import query_small_llm, strip_markdown_fences
from indication_scout.utils.cache import cache_get, cache_set

logger = logging.getLogger(__name__)

_settings = get_settings()
MIN_RESULTS = _settings.disease_pubmed_min_results

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
    cache_params = {"raw_term": raw_term, "small_llm_model": _settings.small_llm_model}
    cached = cache_get("disease_norm", cache_params, DEFAULT_CACHE_DIR)
    if cached is not None:
        return cached

    prompt = (
        (_PROMPTS_DIR / "normalize_disease.txt").read_text().format(raw_term=raw_term)
    )
    response = await query_small_llm(prompt)
    normalized = response.strip().strip('"').strip("'")

    cache_set("disease_norm", cache_params, normalized, DEFAULT_CACHE_DIR)
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
        cached = cache_get(
            "disease_norm",
            {"raw_term": term, "small_llm_model": _settings.small_llm_model},
            DEFAULT_CACHE_DIR,
        )
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
    cleaned = strip_markdown_fences(response)

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
            cache_set(
                "disease_norm",
                {"raw_term": term, "small_llm_model": _settings.small_llm_model},
                normalized,
                DEFAULT_CACHE_DIR,
            )
        results[term] = normalized

    return results


async def merge_duplicate_diseases(
    diseases: list[str],
    drug_indications: list[str],
    max_tokens: int | None = None,
) -> MergeResult:
    """
    Ask the small LLM to collapse synonymous/duplicate disease terms.

    Returns a MergeResult with a `merge` map (canonical → list of aliases) and a
    `remove` list (terms to drop, e.g. already-approved indications). Cached by the
    sorted input sets so identical disease/indication lists reuse the prior result.

    On an unparseable LLM response, logs the error and returns an empty result
    (no merges, no removals) rather than failing the pipeline.
    """
    # Cache key is order-independent: sort both lists so equivalent inputs collide.
    cache_params = {
        "diseases": sorted(diseases),
        "drug_indications": sorted(drug_indications),
        "small_llm_model": _settings.small_llm_model,
    }
    cached = cache_get("disease_merge", cache_params, DEFAULT_CACHE_DIR)
    if cached is not None:
        return cached

    prompt = (
        (_PROMPTS_DIR / "merge_diseases.txt")
        .read_text()
        .format(disease_names=diseases, drug_indications=drug_indications)
    )
    response = await query_small_llm(prompt, max_tokens=max_tokens)
    cleaned = strip_markdown_fences(response)
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(
            "merge_duplicate_diseases: failed to parse LLM response: %s\nResponse was: %s",
            e,
            response,
        )
        return {"merge": {}, "remove": []}

    # Only cache successfully-parsed results; parse failures are not persisted.
    cache_set("disease_merge", cache_params, result, DEFAULT_CACHE_DIR)
    return result


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


# ── MeSH Resolver ────────────────────────────────────────────────────────────

# Pre-emptive sleep before every NCBI call. Crude desynchronization: when several
# coroutines (e.g. parallel disease pairs from the supervisor) hit NCBI in the
# same scheduler tick they otherwise blow past the 10 req/s ceiling and 429.
# Jitter widens the firing window so concurrent callers don't land in lockstep.
_NCBI_PRECALL_SLEEP_BASE: float = 1.5
_NCBI_PRECALL_SLEEP_JITTER: float = 0.5

# Per-request timeout for NCBI MeSH calls. Without an explicit timeout a stalled
# connection can hang on aiohttp's 5min default before TimeoutError fires; a tight
# cap fails fast into the 3s/6s/9s retry. sock_connect bounds connection setup.
_NCBI_TIMEOUT_TOTAL: float = 30.0
_NCBI_SOCK_CONNECT: float = 10.0


async def _ncbi_get_json(
    session: aiohttp.ClientSession,
    url: str,
    params: dict[str, Any],
    indication: str,
) -> dict[str, Any]:
    """GET json from NCBI with up to 3 linearly-spaced retries on transient failure.

    Initial attempt + 3 retries (4 total tries). Each retry waits min(3*(attempt+1), 90)s
    (3s/6s/9s) before firing. If all 4 attempts fail, the program exits non-zero — NCBI is
    a hard dependency for MeSH resolution and continuing without it would
    silently produce degraded clinical analysis.

    A pre-emptive jittered sleep runs before every attempt to keep concurrent
    callers from saturating NCBI's per-second rate ceiling.
    """
    max_retries = 3
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        await asyncio.sleep(
            _NCBI_PRECALL_SLEEP_BASE + random.random() * _NCBI_PRECALL_SLEEP_JITTER
        )
        try:
            async with session.get(url, params=params) as resp:
                logger.debug(
                    "_ncbi_get_json attempt=%d status=%s indication=%r",
                    attempt + 1,
                    resp.status,
                    indication,
                )
                resp.raise_for_status()
                return await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last_exc = e
            if attempt < max_retries:
                # NCBI eutils is a per-second rate limit, so a transient
                # failure clears within ~1s — short linear backoff is enough.
                delay = min(3 * (attempt + 1), 90)
                logger.warning(
                    "MeSH resolver: NCBI request failed for '%s': %s; sleeping %ds "
                    "and retrying (attempt %d/%d)",
                    indication,
                    e,
                    delay,
                    attempt + 1,
                    max_retries,
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "MeSH resolver: NCBI request failed for '%s' after %d retries "
                    "(%d total attempts): %s. Exiting — NCBI is a hard dependency.",
                    indication,
                    max_retries,
                    max_retries + 1,
                    e,
                )
    # All attempts exhausted. NCBI is unreachable / sustainedly throttling us;
    # downstream MeSH-dependent analysis cannot proceed correctly without it.
    # Append a timestamped record to data_source_failures.log so a later
    # session can see *which* indication crashed the previous run, even
    # after stderr scrolls away.
    assert last_exc is not None
    log_data_source_failure(
        source="ncbi-mesh",
        url=url,
        context=indication,
        error=last_exc,
    )
    sys.exit(
        f"FATAL: NCBI eutils unreachable after {max_retries + 1} attempts for "
        f"indication {indication!r}: {last_exc}"
    )


_MESH_RESOLVER_SEMAPHORE: asyncio.Semaphore | None = None
_MESH_RESOLVER_LOOP: asyncio.AbstractEventLoop | None = None


def _mesh_semaphore() -> asyncio.Semaphore:
    # Rebind when the running event loop changes — an asyncio.Semaphore binds to
    # the loop it is created on, so a cached one would raise "bound to a
    # different event loop" on a second asyncio.run().
    global _MESH_RESOLVER_SEMAPHORE, _MESH_RESOLVER_LOOP
    loop = asyncio.get_running_loop()
    if _MESH_RESOLVER_SEMAPHORE is None or _MESH_RESOLVER_LOOP is not loop:
        _MESH_RESOLVER_SEMAPHORE = asyncio.Semaphore(MESH_RESOLVER_MAX_CONCURRENT)
        _MESH_RESOLVER_LOOP = loop
    return _MESH_RESOLVER_SEMAPHORE


async def resolve_mesh_id(indication: str) -> tuple[str, str] | None:
    """Resolve an indication to (descriptor_id, preferred_term) via MeSH ATM.

    esearch (sorted by relevance) returns candidate MeSH record UIDs; esummary
    on each yields its descriptor id (`ds_meshui`, the canonical D-number) and
    preferred heading (`ds_meshterms[0]`). We take the first esummary record
    that is a descriptor with a valid D-number — this skips non-descriptor
    records (qualifiers, supplementary concepts) that esearch can rank in.
    """
    cache_params = {"indication": indication.strip().lower()}
    cached = cache_get("mesh_resolver", cache_params, DEFAULT_CACHE_DIR)
    if cached is not None:
        return tuple(cached) if isinstance(cached, list) else cached

    api_key = get_settings().ncbi_api_key

    esearch_params: dict[str, Any] = {
        "db": "mesh",
        # Qualify to the MeSH-term field. A bare free-text term lets esearch match
        # every sub-concept that merely CONTAINS the word — e.g. "covid-19" ranks
        # "COVID-19 Testing"/"Vaccines"/"Serotherapy" above (or instead of) the bare
        # disease "COVID-19" (D000086382), silently resolving to the wrong descriptor
        # and returning a false-zero trial count. "[MeSH Terms]" makes esearch return
        # the disease descriptor first. Verified: fixes covid-19 and leaves the
        # previously-correct diseases unchanged.
        "term": f"{indication}[MeSH Terms]",
        "retmode": "json",
        # Fetch several candidates sorted by relevance: the top idlist entry can still
        # be a narrower variant (e.g. "hypertension" can return pulmonary-hypertension
        # first), so rank by relevance and pick the first true descriptor below.
        "retmax": 5,
        "sort": "relevance",
    }
    if api_key:
        esearch_params["api_key"] = api_key

    _timeout = aiohttp.ClientTimeout(
        total=_NCBI_TIMEOUT_TOTAL, sock_connect=_NCBI_SOCK_CONNECT
    )
    async with _mesh_semaphore(), aiohttp.ClientSession(timeout=_timeout) as session:
        # NCBI's MeSH backend intermittently returns empty idlist for valid
        # terms. Retry up to 3 times on empty before treating as a real miss.
        for attempt in range(3):
            await asyncio.sleep(0.1)
            async with PubMedClient._get_semaphore():
                esearch_data = await _ncbi_get_json(
                    session, NCBI_ESEARCH_URL, esearch_params, indication
                )
            uids = esearch_data.get("esearchresult", {}).get("idlist", [])
            if uids:
                break
            await asyncio.sleep(2)

        if not uids:
            logger.debug("MeSH resolver: no esearch hit for '%s'", indication)
            return None

        esummary_params: dict[str, Any] = {
            "db": "mesh",
            "id": ",".join(uids),
            "retmode": "json",
        }
        if api_key:
            esummary_params["api_key"] = api_key

        await asyncio.sleep(0.1)
        async with PubMedClient._get_semaphore():
            esummary_data = await _ncbi_get_json(
                session, NCBI_ESUMMARY_URL, esummary_params, indication
            )

    result = esummary_data.get("result", {})
    # Walk uids in esearch (relevance) order; take the first record that is a
    # MeSH descriptor with a D-number and a preferred heading.
    descriptor_id: str | None = None
    preferred_term: str | None = None
    for uid in uids:
        record = result.get(uid)
        if not isinstance(record, dict):
            continue
        meshui = record.get("ds_meshui")
        meshterms = record.get("ds_meshterms")
        if (
            record.get("ds_recordtype") == "descriptor"
            and isinstance(meshui, str)
            and meshui.startswith("D")
            and isinstance(meshterms, list)
            and meshterms
            and meshterms[0]
        ):
            descriptor_id = meshui
            preferred_term = meshterms[0]
            break

    if descriptor_id is None or preferred_term is None:
        logger.warning(
            "MeSH resolver: no descriptor record with a D-number for '%s' (uids=%s)",
            indication,
            uids,
        )
        return None

    logger.info(
        "MeSH resolver: '%s' → descriptor=%s pref=%r",
        indication,
        descriptor_id,
        preferred_term,
    )

    resolved = (descriptor_id, preferred_term)
    cache_set(
        "mesh_resolver",
        cache_params,
        resolved,
        DEFAULT_CACHE_DIR,
        ttl=MESH_RESOLVER_TTL_SECONDS,
    )
    return resolved
