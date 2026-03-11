"""One-off script to fetch PMIDs needed for fetch_and_cache integration tests.

Fetches PMIDs for two overlapping queries and prints:
- PMIDs for each query
- The intersection (overlap)

Run:
    python scripts/fetch_pmids_for_tests.py
"""

import asyncio
import logging

from indication_scout.data_sources.pubmed import PubMedClient

logging.basicConfig(level=logging.WARNING)

QUERY_A = "metformin AND colorectal cancer"
QUERY_B = "metformin AND AMPK AND colon"


async def main() -> None:
    async with PubMedClient() as client:
        pmids_a = await client.search(QUERY_A, max_results=500)
        pmids_b = await client.search(QUERY_B, max_results=500)

    overlap = set(pmids_a) & set(pmids_b)

    print(f"Query A: {QUERY_A!r}")
    print(f"  Count: {len(pmids_a)}")
    print(f"  PMIDs: {sorted(pmids_a)}\n")

    print(f"Query B: {QUERY_B!r}")
    print(f"  Count: {len(pmids_b)}")
    print(f"  PMIDs: {sorted(pmids_b)}\n")

    print(f"Overlap ({len(overlap)} PMIDs): {sorted(overlap)}")


if __name__ == "__main__":
    asyncio.run(main())