"""Date-bounded literature signals for trial-risk features.

For each trial, we ask three questions of the PubMed corpus *as it stood
LOOKBACK_MONTHS before the trial's completion_date*:

  - failure_signal:  how strongly does the literature talk about failure /
                     termination / discontinuation of this drug for this disease?
  - safety_signal:   how strongly does it talk about safety / adverse events /
                     toxicity?
  - efficacy_signal: how strongly does it talk about efficacy / response /
                     positive outcomes?

Each signal is the mean cosine similarity of the top-k semantic-search hits
restricted to PMIDs whose `pub_date < cutoff_date`.

Reuses `services.retrieval.RetrievalService` for fetch+cache and
`services.embeddings.embed_async` for query embeddings. No new external calls
beyond what RetrievalService already does.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.orm import Session

from indication_scout.constants import DEFAULT_CACHE_DIR
from indication_scout.data_sources.chembl import get_all_drug_names, resolve_drug_name
from indication_scout.models.model_clinical_trials import Trial
from indication_scout.services.embeddings import embed_async
from indication_scout.services.retrieval import RetrievalService
from indication_scout.utils.cache import cache_get, cache_set

logger = logging.getLogger(__name__)

DEFAULT_LOOKBACK_MONTHS = 6
DEFAULT_TOP_K = 5
PUBMED_SLEEP_SECONDS = (
    1.0  # NCBI cushion (only matters on actual fetches; cache hits are free)
)
MAX_DRUG_ALIASES = 5  # cap drug-name expansion to avoid 30+ PubMed queries per pair

# Embedding-based queries (kept for inspect.py / backwards compat; not used by signals).
FAILURE_QUERY = (
    "Clinical trial failure, termination, or discontinuation of {drug} for "
    "{disease}, including reasons such as lack of efficacy, safety concerns, "
    "recruitment failure, or toxicity."
)
SAFETY_QUERY = (
    "Adverse events, serious adverse events, toxicity, or safety concerns of "
    "{drug} in {disease}."
)
EFFICACY_QUERY = (
    "Clinical efficacy, treatment response, or positive outcomes of {drug} "
    "for {disease}."
)

# Keyword sets for the v2 fraction-of-abstracts signal. Case-insensitive
# substring match against `title || ' ' || abstract` in pgvector.
FAILURE_KEYWORDS = [
    "terminat",
    "discontinu",
    "withdraw",
    "halted",
    "suspended",
    "lack of efficacy",
    "futility",
    "futile",
]
SAFETY_KEYWORDS = [
    "adverse event",
    "adverse drug",
    "side effect",
    "toxicit",
    "toxic",
    "safety concern",
    "serious adverse",
]
EFFICACY_KEYWORDS = [
    "efficac",
    "remission",
    "response rate",
    "improved",
    "improvement",
    "benefit",
    "effective treatment",
    "clinical response",
]


@dataclass
class LiteratureSignals:
    """The three lit-derived features for a single (drug, disease) pair."""

    failure_signal: float = 0.0
    safety_signal: float = 0.0
    efficacy_signal: float = 0.0
    available: bool = False  # False if no PMIDs were found before the cutoff
    # v3: dense BioLORD fingerprint (mean-pooled abstract embeddings).
    # 768-dim list when available, empty list when not.
    embedding_fingerprint: list[float] = field(default_factory=list)


def parse_trial_date(s: str | None) -> date | None:
    """Parse a ClinicalTrials.gov date (YYYY-MM-DD or YYYY-MM) to a date."""
    if not s:
        return None
    parts = s.split("-")
    try:
        if len(parts) == 3:
            return date(int(parts[0]), int(parts[1]), int(parts[2]))
        if len(parts) == 2:
            return date(int(parts[0]), int(parts[1]), 1)
    except (ValueError, IndexError):
        return None
    return None


def cutoff_for_trial(trial: Trial, lookback_months: int) -> date | None:
    """Return `completion_date - lookback_months`, or None if completion_date missing."""
    end = parse_trial_date(trial.completion_date)
    if end is None:
        return None
    # Subtract months by going back month-by-month to avoid date arithmetic edge cases.
    year, month = end.year, end.month
    for _ in range(lookback_months):
        month -= 1
        if month == 0:
            month = 12
            year -= 1
    day = min(end.day, 28)  # safe day for any month
    return date(year, month, day)


def _mean_pooled_embedding(
    pmids: list[str],
    cutoff_date: date,
    db: Session,
    top_k: int,
    query_vector: list[float] | None = None,
) -> list[float]:
    """Pull top-k abstracts' embeddings (pub_date < cutoff), mean-pool, return as list[float].

    If `query_vector` is provided, ranks by cosine similarity to it. Otherwise
    falls back to most recent by pub_date.
    """
    if not pmids:
        return []
    if query_vector is not None:
        rows = db.execute(
            text("""
                SELECT embedding
                FROM pubmed_abstracts
                WHERE pmid = ANY(:pmids)
                  AND embedding IS NOT NULL
                  AND pub_date IS NOT NULL
                  AND pub_date < :cutoff
                ORDER BY embedding <=> CAST(:qv AS vector) ASC
                LIMIT :top_k
            """),
            {
                "pmids": pmids,
                "cutoff": cutoff_date.isoformat(),
                "qv": "[" + ",".join(str(x) for x in query_vector) + "]",
                "top_k": top_k,
            },
        ).fetchall()
    else:
        rows = db.execute(
            text("""
                SELECT embedding
                FROM pubmed_abstracts
                WHERE pmid = ANY(:pmids)
                  AND embedding IS NOT NULL
                  AND pub_date IS NOT NULL
                  AND pub_date < :cutoff
                ORDER BY pub_date DESC
                LIMIT :top_k
            """),
            {"pmids": pmids, "cutoff": cutoff_date.isoformat(), "top_k": top_k},
        ).fetchall()
    if not rows:
        return []
    vectors: list[list[float]] = []
    for r in rows:
        emb = r[0]
        if isinstance(emb, str):
            # pgvector text format: "[0.1,0.2,...]"
            vectors.append([float(x) for x in emb.strip("[]").split(",")])
        else:
            vectors.append([float(x) for x in emb])
    dim = len(vectors[0])
    pooled = [sum(v[i] for v in vectors) / len(vectors) for i in range(dim)]
    return pooled


def _keyword_or_clause(keywords: list[str], param_prefix: str) -> tuple[str, dict]:
    """Build an OR-of-ILIKEs SQL fragment + the bound params dict."""
    fragments = []
    params: dict[str, str] = {}
    for i, kw in enumerate(keywords):
        key = f"{param_prefix}_{i}"
        fragments.append(f"(title || ' ' || COALESCE(abstract, '')) ILIKE :{key}")
        params[key] = f"%{kw}%"
    return "(" + " OR ".join(fragments) + ")", params


def _keyword_fractions(
    pmids: list[str],
    cutoff_date: date,
    db: Session,
) -> tuple[float, float, float, int]:
    """Return (failure_frac, safety_frac, efficacy_frac, n_abstracts) for the PMID pool.

    Each fraction is the count of abstracts whose title/abstract text matches
    any of the category keywords, divided by the total number of abstracts in
    the pool with `pub_date < cutoff_date`.
    """
    if not pmids:
        return 0.0, 0.0, 0.0, 0

    fail_sql, fail_params = _keyword_or_clause(FAILURE_KEYWORDS, "fail")
    safe_sql, safe_params = _keyword_or_clause(SAFETY_KEYWORDS, "safe")
    eff_sql, eff_params = _keyword_or_clause(EFFICACY_KEYWORDS, "eff")
    sql = f"""
        SELECT
            COUNT(*) AS n,
            SUM(CASE WHEN {fail_sql} THEN 1 ELSE 0 END) AS n_fail,
            SUM(CASE WHEN {safe_sql} THEN 1 ELSE 0 END) AS n_safe,
            SUM(CASE WHEN {eff_sql} THEN 1 ELSE 0 END) AS n_eff
        FROM pubmed_abstracts
        WHERE pmid = ANY(:pmids)
          AND pub_date IS NOT NULL
          AND pub_date < :cutoff
    """
    params = {"pmids": pmids, "cutoff": cutoff_date.isoformat()}
    params.update(fail_params)
    params.update(safe_params)
    params.update(eff_params)
    row = db.execute(text(sql), params).fetchone()
    if row is None or row[0] == 0:
        return 0.0, 0.0, 0.0, 0
    n = int(row[0])
    return (
        float(row[1] or 0) / n,
        float(row[2] or 0) / n,
        float(row[3] or 0) / n,
        n,
    )


async def compute_signals(
    drug: str,
    disease: str,
    cutoff_date: date,
    db: Session,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    top_k: int = DEFAULT_TOP_K,
) -> LiteratureSignals:
    """Compute the three lit signals for one (drug, disease) pair at a cutoff.

    Fetches the broad PubMed PMID universe for `drug AND disease` (cached by
    pair, reused across trials and cutoffs), then runs semantic search against
    pgvector restricted to those PMIDs AND pub_date < cutoff_date. The date
    bound is enforced at the pgvector layer, not at PubMed — so the same
    PubMed query is reused across all trials sharing this (drug, disease)
    pair regardless of their individual cutoff dates.
    """
    if not drug or not disease:
        return LiteratureSignals()

    service = RetrievalService(cache_dir=cache_dir)

    try:
        chembl_id = await resolve_drug_name(drug, cache_dir)
        drug_aliases = (await get_all_drug_names(chembl_id, cache_dir))[
            :MAX_DRUG_ALIASES
        ]
    except Exception as exc:
        logger.warning(
            "Drug name expansion failed for %s: %s — using bare name", drug, exc
        )
        drug_aliases = [drug]

    queries = [f"{alias} AND {disease}" for alias in drug_aliases if alias]
    pmids: list[str] = []
    for q in queries:
        try:
            pair_pmids = await service.fetch_and_cache([q], db, date_before=None)
            pmids.extend(pair_pmids)
        except Exception as exc:
            logger.warning("PubMed fetch failed for %r: %s", q, exc)
        await asyncio.sleep(PUBMED_SLEEP_SECONDS)

    pmids = list(dict.fromkeys(pmids))  # order-preserving dedup
    if not pmids:
        return LiteratureSignals(available=False)

    failure, safety, efficacy, n = _keyword_fractions(pmids, cutoff_date, db)
    if n == 0:
        return LiteratureSignals(available=False)

    # Top-k abstracts most similar to a generic "evidence for {drug} in {disease}" query.
    query_text = (
        f"Evidence for {drug} as a treatment for {disease}, including clinical "
        "trials, efficacy data, mechanism of action, and preclinical studies"
    )
    query_vector = (await embed_async([query_text]))[0]
    fingerprint = _mean_pooled_embedding(
        pmids,
        cutoff_date,
        db,
        top_k=20,
        query_vector=query_vector,
    )
    return LiteratureSignals(
        failure_signal=failure,
        safety_signal=safety,
        efficacy_signal=efficacy,
        available=True,
        embedding_fingerprint=fingerprint,
    )


_TRIAL_LIT_NAMESPACE = "trial_risk_lit_signals"


async def signals_for_trial(
    trial: Trial,
    drug: str,
    db: Session,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    lookback_months: int = DEFAULT_LOOKBACK_MONTHS,
    top_k: int = DEFAULT_TOP_K,
) -> LiteratureSignals:
    """Compute lit signals for a trial, averaged across its mesh_conditions.

    Result is cached by (nct_id, lookback_months, top_k) so subsequent runs
    skip the PubMed/embed/pgvector work entirely.

    Returns LiteratureSignals(available=False) when the trial has no
    completion_date or no mesh_conditions.
    """
    cache_params = {
        "nct_id": trial.nct_id,
        "lookback_months": lookback_months,
        "top_k": top_k,
    }
    cached = cache_get(_TRIAL_LIT_NAMESPACE, cache_params, cache_dir)
    if cached is not None:
        return LiteratureSignals(**cached)

    cutoff = cutoff_for_trial(trial, lookback_months)
    if cutoff is None:
        result = LiteratureSignals()
        cache_set(_TRIAL_LIT_NAMESPACE, cache_params, result.__dict__, cache_dir)
        return result

    diseases = [m.term for m in trial.mesh_conditions if m.term]
    if not diseases:
        result = LiteratureSignals()
        cache_set(_TRIAL_LIT_NAMESPACE, cache_params, result.__dict__, cache_dir)
        return result

    per_pair = []
    for disease in diseases:
        sig = await compute_signals(drug, disease, cutoff, db, cache_dir, top_k)
        if sig.available:
            per_pair.append(sig)

    if not per_pair:
        result = LiteratureSignals(available=False)
    else:
        n = len(per_pair)
        # Average fingerprints across pairs that have one (skip empty).
        fps = [s.embedding_fingerprint for s in per_pair if s.embedding_fingerprint]
        if fps:
            dim = len(fps[0])
            avg_fp = [sum(fp[i] for fp in fps) / len(fps) for i in range(dim)]
        else:
            avg_fp = []
        result = LiteratureSignals(
            failure_signal=sum(s.failure_signal for s in per_pair) / n,
            safety_signal=sum(s.safety_signal for s in per_pair) / n,
            efficacy_signal=sum(s.efficacy_signal for s in per_pair) / n,
            available=True,
            embedding_fingerprint=avg_fp,
        )
    cache_set(_TRIAL_LIT_NAMESPACE, cache_params, result.__dict__, cache_dir)
    return result
