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

from indication_scout.constants import (  # noqa: E402
    BROADENING_BLOCKLIST,
    CLINICAL_STAGE_RANK,
    DEFAULT_CACHE_DIR,
)
from indication_scout.data_sources.chembl import resolve_drug_name  # noqa: E402
from indication_scout.data_sources.open_targets import OpenTargetsClient  # noqa: E402
from indication_scout.services.retrieval import RetrievalService  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("probe")


async def raw_ot_ranking(
    client: OpenTargetsClient, chembl_id: str, min_stage: str = "PHASE_3"
) -> list[tuple[str, int]]:
    """Reproduce the OT sibling ranking pre-truncation: [(disease, sibling_count)].

    Mirrors get_drug_competitors' ranking logic so we can see the full ordered
    list and find where a disease ranks before the top-N prefetch cut.
    """
    min_rank = CLINICAL_STAGE_RANK.get(min_stage, 0)
    drug = await client.get_drug(chembl_id)
    all_summaries = await asyncio.gather(
        *[client.get_target_data_drug_summaries(t.target_id) for t in drug.targets]
    )
    siblings: dict[str, set[str]] = {}
    for summaries in all_summaries:
        for s in summaries:
            if CLINICAL_STAGE_RANK.get(s.max_clinical_stage or "", 0) < min_rank:
                continue
            for cd in s.diseases:
                if cd.disease_name is None:
                    continue
                siblings.setdefault(cd.disease_name.lower(), set()).add(s.drug_name)
    for key in list(siblings):
        if {w.lower() for w in key.split()} <= BROADENING_BLOCKLIST:
            del siblings[key]
    return sorted(
        ((d, len(s)) for d, s in siblings.items()), key=lambda x: x[1], reverse=True
    )


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
            ranking = await raw_ot_ranking(client, chembl_id)
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
