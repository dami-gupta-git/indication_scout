"""Search stored PubMed embeddings for cardiac side effects of a drug.

This script is read-only: it searches abstracts already present in
``pubmed_abstracts`` and does not fetch papers or call an LLM.

Usage:
    .venv/bin/python scripts/search_cardiac_side_effects.py \
        --drug semaglutide --top-k 20
"""

import argparse
import asyncio
import logging
import re

from sqlalchemy import text

from indication_scout.constants import DEFAULT_CACHE_DIR
from indication_scout.data_sources.chembl import (
    get_all_drug_names,
    resolve_drug_name,
)
from indication_scout.db.session import make_session_factory
from indication_scout.services.embeddings import embed_async

logger = logging.getLogger("search_cardiac_side_effects")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--drug", required=True, help="Drug name or brand name.")
    parser.add_argument(
        "--top-k", required=True, type=int, help="Number of matching abstracts to show."
    )
    return parser.parse_args()


def _query_text(drug_name: str) -> str:
    return f"Cardiac side effects caused by {drug_name}"


def _alias_patterns(aliases: list[str]) -> list[str]:
    """Return case-insensitive PostgreSQL word-boundary patterns for unique aliases."""
    unique = list(dict.fromkeys(alias.strip() for alias in aliases if alias.strip()))
    return [rf"\m{re.escape(alias)}\M" for alias in unique]


async def _main() -> None:
    args = _parse_args()
    if args.top_k <= 0:
        raise SystemExit("--top-k must be greater than zero")

    chembl_id = await resolve_drug_name(args.drug, DEFAULT_CACHE_DIR)
    aliases = await get_all_drug_names(chembl_id, DEFAULT_CACHE_DIR)
    patterns = _alias_patterns([args.drug, *aliases])
    query = _query_text(aliases[0])
    query_vector = (await embed_async([query]))[0]

    session_factory = make_session_factory()
    with session_factory() as db:
        rows = db.execute(
            text("""
                SELECT pmid,
                       title,
                       pub_date,
                       1 - (embedding <=> CAST(:query_vec AS vector)) AS similarity,
                       (
                           SELECT alias_pattern
                           FROM unnest(CAST(:patterns AS text[])) AS alias_pattern
                           WHERE (title || ' ' || COALESCE(abstract, '')) ~* alias_pattern
                           LIMIT 1
                       ) AS matched_pattern
                FROM pubmed_abstracts
                WHERE embedding IS NOT NULL
                  AND (title || ' ' || COALESCE(abstract, '')) ~* ANY(
                      CAST(:patterns AS text[])
                  )
                ORDER BY embedding <=> CAST(:query_vec AS vector)
                LIMIT :top_k
            """),
            {
                "query_vec": "[" + ",".join(str(value) for value in query_vector) + "]",
                "patterns": patterns,
                "top_k": args.top_k,
            },
        ).fetchall()

    logger.info("drug: %s (%s)", aliases[0], chembl_id)
    logger.info("query: %s", query)
    logger.info("aliases checked: %d", len(patterns))
    logger.info("results: %d", len(rows))
    for rank, row in enumerate(rows, 1):
        matched_alias = row.matched_pattern.removeprefix(r"\m").removesuffix(r"\M")
        logger.info(
            "%2d  %.3f  PMID %-9s  %-12s  [%s]  %s",
            rank,
            float(row.similarity),
            row.pmid,
            row.pub_date or "",
            matched_alias,
            row.title,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(_main())
