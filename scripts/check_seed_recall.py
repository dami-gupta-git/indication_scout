"""Assert seed-phase candidate recall for the regression drugs.

CI guard, not part of the live pipeline. For every runbook row belonging to a drug named in
`tests/regression/specs/seed_recall.yaml`, run the seed phase under that row's holdout cutoff and
check that the target indication reaches the merged candidate list. Matching is exact name against
the row's `indication` plus its `accepted` column — the same rule
`scripts/validation/gen_seed_candidate_recall.py` scores with.

Rows listed in the spec's `known_missing` are expected to be absent. A known-missing row that
starts passing is also a failure, so a fix gets recorded in the spec instead of going unnoticed.

Writes a JSON result file (`--out`, default `results/ci/seed_recall.json`): a `summary` block with
the run's counts and recall, and one `rows` entry per runbook row carrying its position, matched
name, source and verdict. The shape is shared with the precision check so both metrics land in the
same directory in the same form.

Exit status is 0 when every row matches its expectation, 1 otherwise.

Run:
    CONSTANTS_FILE=.env.constants python scripts/check_seed_recall.py [--out path.json]
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import yaml
from langchain_anthropic import ChatAnthropic
from sqlalchemy.orm import sessionmaker

from indication_scout.config import get_settings
from indication_scout.constants import DEFAULT_CACHE_DIR
from indication_scout.db.session import _make_engine
from indication_scout.services.retrieval import RetrievalService

sys.path.insert(0, str(Path(__file__).resolve().parent / "validation"))
from gen_seed_candidate_recall import (  # noqa: E402 - needs the sys.path entry above
    _match_accepted,
    _merged_for_drug,
)

# Own handler, not basicConfig: importing the package pulls in a module that configures the root
# logger at WARNING on import, which would silently drop every line this script logs.
logger = logging.getLogger("check_seed_recall")
logger.setLevel(logging.INFO)
logger.propagate = False
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_handler)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNBOOK = PROJECT_ROOT / "scripts" / "validation" / "runbook.txt"
SPEC = PROJECT_ROOT / "tests" / "regression" / "specs" / "seed_recall.yaml"
DEFAULT_OUT = PROJECT_ROOT / "results" / "ci" / "seed_recall.json"


def _load_spec() -> tuple[list[str], set[tuple[str, str]]]:
    """Return the drugs to check and the (drug, indication) pairs expected to be absent."""
    spec = yaml.safe_load(SPEC.read_text())
    drugs = [d.strip().lower() for d in spec["drugs"]]
    known_missing = {
        (entry["drug"].strip().lower(), entry["indication"].strip().lower())
        for entry in spec.get("known_missing") or []
    }
    return drugs, known_missing


def _rows_for(drugs: list[str]) -> list[dict[str, str]]:
    with open(RUNBOOK, newline="") as f:
        return [r for r in csv.DictReader(f) if r["drug"].strip().lower() in drugs]


def _write_results(
    out_path: Path, rows: list[dict[str, Any]], failures: list[str]
) -> None:
    """Write the per-row detail plus a summary block. Shared shape with the precision check."""
    settings = get_settings()
    scored = [r for r in rows if not r["expected_missing"]]
    found = [r for r in scored if r["present"]]
    payload = {
        "metric": "seed_candidate_recall",
        "generated_at": datetime.now(UTC).isoformat(),
        "settings": {
            "supervisor_candidate_cap": settings.supervisor_candidate_cap,
            "supervisor_investigation_cap": settings.supervisor_investigation_cap,
            "open_targets_competitor_prefetch_max": settings.open_targets_competitor_prefetch_max,
            "mechanism_associations_per_target": settings.mechanism_associations_per_target,
            "mechanism_top_candidates": settings.mechanism_top_candidates,
        },
        "summary": {
            "rows": len(rows),
            "scored": len(scored),
            "found": len(found),
            "known_missing": len(rows) - len(scored),
            "recall": round(len(found) / len(scored), 4) if scored else None,
            "failures": len(failures),
            "passed": not failures,
        },
        "failures": failures,
        "rows": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info("wrote %s", out_path)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"JSON result file (default: {DEFAULT_OUT.relative_to(PROJECT_ROOT)})",
    )
    args = parser.parse_args()

    drugs, known_missing = _load_spec()
    rows = _rows_for(drugs)
    if not rows:
        logger.error("No runbook rows for %s", drugs)
        return 1

    settings = get_settings()
    llm = ChatAnthropic(
        model=settings.llm_model,
        temperature=0,
        max_tokens=4096,
        anthropic_api_key=settings.anthropic_api_key,
    )
    svc = RetrievalService(DEFAULT_CACHE_DIR)
    session_factory = sessionmaker(
        autocommit=False, autoflush=False, bind=_make_engine()
    )

    # One seed-phase run per distinct (drug, cutoff); runbook rows sharing one reuse it. The runs
    # are independent — each builds its own tool closure and DB session — so they go concurrently,
    # bounded by the same semaphore width the supervisor's own fan-out uses.
    unique_runs = list(
        dict.fromkeys((r["drug"].strip(), r["date"].strip()) for r in rows)
    )
    semaphore = asyncio.Semaphore(settings.supervisor_investigation_concurrency)

    async def seed_once(drug: str, cutoff: str) -> list[tuple[str, str]]:
        async with semaphore:
            return await _merged_for_drug(
                llm, svc, session_factory, drug, date.fromisoformat(cutoff)
            )

    logger.info(
        "seeding %d drug/cutoff pairs for %d rows (concurrency %d)",
        len(unique_runs),
        len(rows),
        settings.supervisor_investigation_concurrency,
    )
    merged_lists = await asyncio.gather(
        *(seed_once(drug, cutoff) for drug, cutoff in unique_runs)
    )
    merged_by_run: dict[tuple[str, str], list[tuple[str, str]]] = dict(
        zip(unique_runs, merged_lists, strict=True)
    )

    failures: list[str] = []
    results: list[dict[str, Any]] = []

    for row in rows:
        drug, indication, cutoff = (
            row["drug"].strip(),
            row["indication"].strip(),
            row["date"].strip(),
        )
        merged = merged_by_run[(drug, cutoff)]
        names = [name for name, _source in merged]

        accepted = [indication] + [
            a for a in (row.get("accepted") or "").split(";") if a.strip()
        ]
        idx = _match_accepted(accepted, names)
        expected_missing = (drug.lower(), indication.lower()) in known_missing

        result: dict[str, Any] = {
            "drug": drug,
            "indication": indication,
            "cutoff": cutoff,
            "accepted": accepted,
            "candidates": len(merged),
            "present": idx is not None,
            "position": idx + 1 if idx is not None else None,
            "matched": names[idx] if idx is not None else None,
            "source": merged[idx][1] if idx is not None else None,
            "expected_missing": expected_missing,
        }

        label = f"{drug} / {indication}"
        if idx is None and not expected_missing:
            failures.append(f"{label}: absent from the candidate list")
            result["verdict"] = "fail_absent"
            logger.info(
                "FAIL       %-58s absent from %d candidates", label, len(merged)
            )
        elif idx is not None and expected_missing:
            failures.append(
                f"{label}: listed as known_missing but now present at "
                f"{idx + 1} ({names[idx]}) — remove it from {SPEC.name}"
            )
            result["verdict"] = "fail_unexpected_pass"
            logger.info(
                "FAIL       %-58s now present at %d (%s) — expected missing",
                label,
                idx + 1,
                names[idx],
            )
        elif idx is None:
            result["verdict"] = "known_missing"
            logger.info("known-miss %-58s absent, as recorded in %s", label, SPEC.name)
        else:
            result["verdict"] = "ok"
            logger.info(
                "ok         %-58s %d/%d via %s [%s]",
                label,
                idx + 1,
                len(merged),
                names[idx],
                merged[idx][1],
            )

        results.append(result)

    _write_results(args.out, results, failures)

    scored = [r for r in results if not r["expected_missing"]]
    found = [r for r in scored if r["present"]]
    logger.info(
        "\n%d rows over %d drugs: %d of %d scored rows found (%.0f%% recall), "
        "%d known-missing, %d failed",
        len(results),
        len(drugs),
        len(found),
        len(scored),
        100 * len(found) / len(scored) if scored else 0.0,
        len(results) - len(scored),
        len(failures),
    )
    logger.info(
        "prefetch=%d candidate cap=%d investigation cap=%d",
        settings.open_targets_competitor_prefetch_max,
        settings.supervisor_candidate_cap,
        settings.supervisor_investigation_cap,
    )
    logger.info("detail: %s", args.out)

    if failures:
        logger.info("\nFAILED — %d of %d rows:", len(failures), len(results))
        for f in failures:
            logger.info("  %s", f)
        return 1

    logger.info("PASSED — every row matches %s", SPEC.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
