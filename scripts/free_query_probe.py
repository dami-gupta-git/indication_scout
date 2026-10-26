"""Probe: rank a drug's already-stored abstracts against free-text queries by embedding similarity.

Read-only: no PubMed calls, no new embeddings, no LLM calls. The pool is every abstract in ``pubmed_abstracts`` whose title or
abstract mentions the drug or one of its ChEMBL aliases (word-boundary regex). Each query is embedded with the pipeline's
model and the pool is ranked by cosine similarity. No pubtype boost, so the ranking is the raw embedding behaviour.

A query may contain ``{drug}``, replaced by the drug's preferred ChEMBL name.

With --keywords, a case-insensitive regex is checked against title + abstract. The report then shows how many pool abstracts
match, how many of them reach the top-k, and the worst-ranked match, as a rough lexical baseline for the embedding.

Usage:
    .venv/bin/python scripts/free_query_probe.py -d ibuprofen \
        -q "find papers that have info about gastrointestinal side effects" \
        -q "Gastrointestinal side effects caused by {drug}" \
        --keywords "gastrointestinal|gastric|ulcer|bleed|hemorrhag|dyspepsia"
"""

import argparse
import asyncio
import logging
import re

from sqlalchemy import text

from indication_scout.constants import DEFAULT_CACHE_DIR
from indication_scout.data_sources.chembl import get_all_drug_names, resolve_drug_name
from indication_scout.db.session import make_session_factory
from indication_scout.services.embeddings import embed_async

logger = logging.getLogger("free_query_probe")

TITLE_WIDTH = 95

RankedRow = tuple[str, str, str, str | None, float]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("-d", "--drug", required=True, help="Drug name or brand name.")
    parser.add_argument(
        "-q",
        "--query",
        action="append",
        required=True,
        help="Free-text query to embed; repeatable. May contain {drug}.",
    )
    parser.add_argument(
        "-k", "--top-k", type=int, default=15, help="Rows to show per query."
    )
    parser.add_argument(
        "--keywords", help="Regex for the lexical baseline check (case-insensitive)."
    )
    return parser.parse_args()


def _alias_patterns(aliases: list[str]) -> list[str]:
    """Return word-boundary PostgreSQL regex patterns for the unique, non-empty aliases."""
    unique = list(dict.fromkeys(alias.strip() for alias in aliases if alias.strip()))
    return [rf"\m{re.escape(alias)}\M" for alias in unique]


def _rank_pool(
    db_session, query_vector: list[float], patterns: list[str]
) -> list[RankedRow]:
    """Return (pmid, title, abstract, pub_date, similarity) for every stored abstract naming the drug, best first."""
    rows = db_session.execute(
        text("""
            SELECT pmid, title, COALESCE(abstract, ''), pub_date,
                   1 - (embedding <=> CAST(:query_vec AS vector)) AS similarity
            FROM pubmed_abstracts
            WHERE embedding IS NOT NULL
              AND (title || ' ' || COALESCE(abstract, '')) ~* ANY(CAST(:patterns AS text[]))
            ORDER BY embedding <=> CAST(:query_vec AS vector)
        """),
        {
            "query_vec": "[" + ",".join(str(x) for x in query_vector) + "]",
            "patterns": patterns,
        },
    ).fetchall()
    return [(str(r[0]), r[1], r[2], r[3], float(r[4])) for r in rows]


def _report(
    query: str, ranked: list[RankedRow], top_k: int, keywords: re.Pattern[str] | None
) -> None:
    logger.info("")
    logger.info("=" * 120)
    logger.info("QUERY: %s", query)
    logger.info("pool: %d stored abstracts naming the drug", len(ranked))
    logger.info("-" * 120)

    hit_ranks = {
        i
        for i, (_, title, abstract, _, _) in enumerate(ranked, 1)
        if keywords and keywords.search(f"{title} {abstract}")
    }

    for rank, (pmid, title, _, pub_date, sim) in enumerate(ranked[:top_k], 1):
        flag = ("KW" if rank in hit_ranks else "--") if keywords else "  "
        logger.info(
            "%2d  %.3f  %s  %-9s %-8s %s",
            rank,
            sim,
            flag,
            pmid,
            (pub_date or "")[:8],
            title[:TITLE_WIDTH],
        )

    if keywords:
        in_top = sum(1 for r in hit_ranks if r <= top_k)
        logger.info("-" * 120)
        logger.info(
            "keyword baseline: %d of %d pool abstracts match; %d of the top %d match%s",
            len(hit_ranks),
            len(ranked),
            in_top,
            min(top_k, len(ranked)),
            f"; worst-ranked match is #{max(hit_ranks)}" if hit_ranks else "",
        )
        missed = [(r, ranked[r - 1]) for r in sorted(hit_ranks) if r > top_k][:5]
        if missed:
            logger.info("keyword matches ranked below the top %d (first 5):", top_k)
            for r, (pmid, title, _, _, sim) in missed:
                logger.info("  #%-4d %.3f  %-9s %s", r, sim, pmid, title[:TITLE_WIDTH])


async def _main() -> None:
    args = _parse_args()
    keywords = re.compile(args.keywords, re.IGNORECASE) if args.keywords else None

    chembl_id = await resolve_drug_name(args.drug, DEFAULT_CACHE_DIR)
    aliases = await get_all_drug_names(chembl_id, DEFAULT_CACHE_DIR)
    patterns = _alias_patterns([args.drug, *aliases])
    logger.info("drug: %s (%s); %d aliases", aliases[0], chembl_id, len(patterns))

    queries = [q.replace("{drug}", aliases[0]) for q in args.query]
    vectors = await embed_async(queries)

    with make_session_factory()() as db_session:
        for query, vector in zip(queries, vectors):
            _report(
                query, _rank_pool(db_session, vector, patterns), args.top_k, keywords
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(_main())
