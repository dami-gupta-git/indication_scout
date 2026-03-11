"""One-off script to fetch get_drug_competitors output for known drugs.

Used to pin exact assertions in integration tests.

Run:
    python scripts/fetch_competitors_for_tests.py
"""

import asyncio
import logging

from indication_scout.data_sources.open_targets import OpenTargetsClient

logging.basicConfig(level=logging.WARNING)

DRUGS = ["bupropion", "sildenafil"]


async def main() -> None:
    async with OpenTargetsClient() as client:
        for drug in DRUGS:
            result = await client.get_drug_competitors(drug)
            print(f"\n=== {drug} ===")
            print(f"Candidate count: {len(result)}")
            for disease, competitors in result.items():
                print(f"  {disease!r}: {sorted(competitors)}")


if __name__ == "__main__":
    asyncio.run(main())
