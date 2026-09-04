"""Probe the competitor-candidate stage standalone, to debug a missed disease.

When a holdout run scores -1 for a known indication, the first question is "did
find_candidates even produce this disease?". This runs only the competitor path
that feeds find_candidates' allowlist — Open Targets sibling ranking, top-N
prefetch, and the dedup/merge — without invoking any LLM agent. It reports
whether a target disease survived to the merged candidate list, and (from the
raw OT ranking) where it fell if it did not.

Usage:
    probe_candidates.py <drug> <YYYY-MM-DD> [target disease substring]

Example:
    probe_candidates.py imatinib 2006-05-05 eosinophil
"""

import asyncio
import logging
import os
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

VALIDATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VALIDATION_DIR.parent.parent


def _load_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    # Validation must mirror production, not the cheap test profile.
    constants_file = os.environ.get("CONSTANTS_FILE", ".env.constants")
    constants_path = PROJECT_ROOT / constants_file
    load_dotenv(constants_path)
    os.environ["CONSTANTS_FILE"] = str(constants_path)


_load_env()

from indication_scout.constants import DEFAULT_CACHE_DIR  # noqa: E402
from indication_scout.data_sources.chembl import resolve_drug_name  # noqa: E402
from indication_scout.data_sources.open_targets import OpenTargetsClient  # noqa: E402
from indication_scout.services.retrieval import RetrievalService  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("probe")


async def raw_ot_ranking(
    client: OpenTargetsClient, chembl_id: str, cutoff: date
) -> list[tuple[str, int]]:
    """The OT sibling ranking pre-truncation: [(disease, competitor_count)].

    Calls the client's own ranking method so the probe cannot drift from what the pipeline
    computes; get_drug_competitors truncates the same ranking to the prefetch max.
    """
    ranking = await client.rank_competitor_siblings(chembl_id, date_before=cutoff)
    return [(d, len(s)) for d, s in ranking["siblings"].items()]


async def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    drug = sys.argv[1]
    cutoff = date.fromisoformat(sys.argv[2])
    target = sys.argv[3].lower() if len(sys.argv) > 3 else None

    svc = RetrievalService(DEFAULT_CACHE_DIR)
    chembl_id = await resolve_drug_name(drug, svc.cache_dir)
    print(f"drug={drug}  chembl={chembl_id}  cutoff={cutoff.isoformat()}")

    # Final merged competitor candidate list — what find_candidates seeds from.
    merged = await svc.get_drug_competitors(chembl_id, date_before=cutoff)
    candidates = sorted(merged.keys())
    print(f"\nMerged competitor candidates ({len(candidates)}):")
    for c in candidates:
        print(f"  - {c}")

    if target is None:
        return

    hits = [c for c in candidates if target in c.lower()]
    print(f"\nTarget '{target}' in merged candidates: {hits or 'NO'}")

    # If absent, show where it sat in the full pre-truncation OT ranking.
    if not hits:
        async with OpenTargetsClient(cache_dir=svc.cache_dir) as client:
            ranking = await raw_ot_ranking(client, chembl_id, cutoff)
        positions = [
            (i, d, n) for i, (d, n) in enumerate(ranking) if target in d.lower()
        ]
        print(f"\nFull OT ranking has {len(ranking)} diseases (truncated to top-N).")
        if positions:
            for i, d, n in positions:
                print(f"  rank #{i} of {len(ranking)}: {d}  (siblings={n})")
        else:
            print(f"  '{target}' not present in OT sibling ranking at all.")


if __name__ == "__main__":
    asyncio.run(main())
