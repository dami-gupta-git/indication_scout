"""Inspect what abstracts the lit-signal queries actually retrieve.

Given an NCT ID, prints the top-K abstracts retrieved for each of the three
signal queries (failure / safety / efficacy), restricted to the same date
cutoff used at training time.

Run:
    python -m indication_scout.ml_models.trial_risk.inspect NCT00064337
    python -m indication_scout.ml_models.trial_risk.inspect NCT00064337 --top-k 10
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from sqlalchemy import text

from indication_scout.constants import DEFAULT_CACHE_DIR
from indication_scout.db.session import get_db
from indication_scout.services.embeddings import embed_async
from indication_scout.services.retrieval import RetrievalService
from indication_scout.ml_models.trial_risk.data import load_labeled_trials
from indication_scout.ml_models.trial_risk.literature import (
    DEFAULT_LOOKBACK_MONTHS,
    EFFICACY_QUERY,
    FAILURE_QUERY,
    MAX_DRUG_ALIASES,
    SAFETY_QUERY,
    cutoff_for_trial,
)
from indication_scout.data_sources.chembl import get_all_drug_names, resolve_drug_name

logger = logging.getLogger(__name__)


async def _top_abstracts(
    query: str,
    pmids: list[str],
    cutoff_date,
    db,
    top_k: int,
) -> list[tuple[str, str, str, float]]:
    """Return [(pmid, title, abstract, similarity), ...] sorted desc."""
    if not pmids:
        return []
    vec = (await embed_async([query]))[0]
    rows = db.execute(
        text("""
            SELECT pmid, title, abstract,
                   1 - (embedding <=> CAST(:q AS vector)) AS sim
            FROM pubmed_abstracts
            WHERE pmid = ANY(:pmids)
              AND pub_date IS NOT NULL
              AND pub_date < :cutoff
            ORDER BY sim DESC
            LIMIT :k
        """),
        {
            "q": "[" + ",".join(str(x) for x in vec) + "]",
            "pmids": pmids,
            "cutoff": cutoff_date.isoformat(),
            "k": top_k,
        },
    ).fetchall()
    return [(r[0], r[1] or "", r[2] or "", float(r[3])) for r in rows]


def _print_block(label: str, hits: list[tuple[str, str, str, float]]) -> None:
    print(f"\n{'=' * 80}")
    print(f"  {label}")
    print(f"{'=' * 80}")
    if not hits:
        print("  (no hits)")
        return
    for i, (pmid, title, abstract, sim) in enumerate(hits, 1):
        print(f"\n[{i}] PMID {pmid}  cosine={sim:.3f}")
        print(f"    {title}")
        snippet = abstract.strip().replace("\n", " ")
        if len(snippet) > 400:
            snippet = snippet[:400] + "…"
        print(f"    {snippet}")


async def main_async(nct_id: str, top_k: int, cache_dir: Path) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )

    labeled = load_labeled_trials(cache_dir)
    by_nct = {lt.trial.nct_id: lt for lt in labeled}
    if nct_id not in by_nct:
        logger.error("NCT ID %s not in cache.", nct_id)
        return

    lt = by_nct[nct_id]
    trial = lt.trial
    cutoff = cutoff_for_trial(trial, DEFAULT_LOOKBACK_MONTHS)
    diseases = [m.term for m in trial.mesh_conditions if m.term]

    print(f"\nTrial:        {trial.nct_id}")
    print(f"Title:        {trial.title}")
    print(f"Drug:         {lt.drug}")
    print(f"Phase:        {trial.phase}")
    print(f"Status:       {trial.overall_status}  (label={lt.label})")
    print(f"why_stopped:  {trial.why_stopped or '-'}")
    print(f"completion:   {trial.completion_date}")
    print(f"cutoff:       {cutoff}")
    print(f"mesh:         {diseases}")

    if cutoff is None or not diseases:
        print("\n(no cutoff or no mesh conditions — nothing to inspect)")
        return

    db_gen = get_db()
    db = next(db_gen)
    service = RetrievalService(cache_dir=cache_dir)

    try:
        chembl_id = await resolve_drug_name(lt.drug, cache_dir)
        aliases = (await get_all_drug_names(chembl_id, cache_dir))[:MAX_DRUG_ALIASES]
    except Exception:
        aliases = [lt.drug]

    for disease in diseases:
        print(f"\n\n{'#' * 80}")
        print(f"#  Disease: {disease}")
        print(f"{'#' * 80}")

        all_pmids: list[str] = []
        for alias in aliases:
            q = f"{alias} AND {disease}"
            pmids = await service.fetch_and_cache([q], db, date_before=None)
            all_pmids.extend(pmids)
        all_pmids = list(dict.fromkeys(all_pmids))
        print(f"PMIDs (deduped, broad search): {len(all_pmids)}")

        for label, query in [
            ("FAILURE", FAILURE_QUERY.format(drug=lt.drug, disease=disease)),
            ("SAFETY", SAFETY_QUERY.format(drug=lt.drug, disease=disease)),
            ("EFFICACY", EFFICACY_QUERY.format(drug=lt.drug, disease=disease)),
        ]:
            hits = await _top_abstracts(query, all_pmids, cutoff, db, top_k)
            mean_sim = sum(h[3] for h in hits[:5]) / max(len(hits[:5]), 1)
            _print_block(f"{label}  (mean cosine top-5 = {mean_sim:.3f})", hits)

    db.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect lit-signal abstracts for a trial."
    )
    parser.add_argument("nct_id", help="NCT ID to inspect.")
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of abstracts to show per category."
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    args = parser.parse_args()
    asyncio.run(main_async(args.nct_id, args.top_k, args.cache_dir))


if __name__ == "__main__":
    main()
