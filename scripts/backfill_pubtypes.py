"""One-off maintenance script: backfill the pubtypes file cache for PMIDs already in pgvector.

Context: `fetch_and_cache` now warms `cache/pubmed_pubtypes/` from efetch XML, but only for
NEWLY fetched abstracts. PMIDs stored in pubmed_abstracts before that change have no cached
pubtype, so semantic_search still pays a cold esummary round-trip for them. This script closes
that gap once: it reads every PMID from pubmed_abstracts, diffs against the pubtypes cache, and
fetches the missing ones via the same `fetch_pubtypes` path (which writes the cache as it goes).

Idempotent: re-running only fetches whatever is still missing. Safe to interrupt and resume.
Will sppedup 

Run:
    python scripts/backfill_pubtypes.py
    python scripts/backfill_pubtypes.py --dry-run    # just report the gap, fetch nothing
"""

import argparse
import asyncio
import logging

from sqlalchemy import text

from indication_scout.constants import DEFAULT_CACHE_DIR
from indication_scout.data_sources.pubmed import PubMedClient
from indication_scout.db.session import _make_session_factory
from indication_scout.utils.cache import cache_get

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("indication_scout.backfill_pubtypes")

# Progress log cadence and esummary fan-out chunk. fetch_pubtypes batches internally
# (pubmed_esummary_batch_size, default 200); we chunk here so progress is visible and a
# crash loses at most one chunk of work.
_CHUNK_SIZE = 1000


def _stored_pmids(cache_dir) -> tuple[list[str], int]:
    """Return PMIDs in pubmed_abstracts that have no cached pubtype, plus the total count."""
    factory = _make_session_factory()
    with factory() as db:
        rows = db.execute(text("SELECT pmid FROM pubmed_abstracts")).fetchall()
    all_pmids = [r[0] for r in rows]
    missing = [
        pmid
        for pmid in all_pmids
        if cache_get("pubmed_pubtypes", {"pmid": pmid}, cache_dir) is None
    ]
    return missing, len(all_pmids)


async def main(dry_run: bool) -> None:
    cache_dir = DEFAULT_CACHE_DIR
    missing, total = _stored_pmids(cache_dir)
    logger.info(
        "pubmed_abstracts=%d, pubtypes cached=%d, missing=%d",
        total,
        total - len(missing),
        len(missing),
    )
    if not missing:
        logger.info("Nothing to backfill; pubtypes cache is complete.")
        return
    if dry_run:
        logger.info("Dry run: would fetch pubtypes for %d PMIDs.", len(missing))
        return

    fetched = 0
    async with PubMedClient(cache_dir=cache_dir) as client:
        for i in range(0, len(missing), _CHUNK_SIZE):
            chunk = missing[i : i + _CHUNK_SIZE]
            # fetch_pubtypes writes each PMID to the cache as a side effect; we
            # ignore the return value and rely on that cache-warming behaviour.
            await client.fetch_pubtypes(chunk)
            fetched += len(chunk)
            logger.info("Backfilled %d/%d PMIDs", fetched, len(missing))

    logger.info("Done. Backfilled pubtypes for %d PMIDs.", fetched)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the cache gap without fetching anything.",
    )
    args = parser.parse_args()
    asyncio.run(main(args.dry_run))
