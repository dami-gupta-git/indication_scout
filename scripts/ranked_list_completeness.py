"""Measure how often the supervisor omits investigated candidates from its ranked list.

Each disease in `disease_findings` should appear once: ranked, in "Not ranked:", or in "Evidence gate exclusions:".
Reports, per run and in aggregate, diseases investigated but absent from the ranking, and those that flip between
ranked and absent across runs of the same drug.

    # Classify saved payloads (no LLM calls)
    python scripts/ranked_list_completeness.py --reports [--since 2026-09-10] [--drug bupropion ...]

    # Run the pipeline N times per drug and classify each run
    python scripts/ranked_list_completeness.py --run bupropion semaglutide --repeats 3 [--out DIR]

Live runs save payloads to `<out>/<drug>_<ts>.json` (default `test_reports/completeness/`) for later `--reports --dir`.
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# load_dotenv MUST run before any indication_scout imports (base_client.py reads settings at import).
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")
constants_path = PROJECT_ROOT / ".env.constants"
load_dotenv(constants_path)
os.environ["CONSTANTS_FILE"] = str(constants_path)
# .env points SCOUT_CACHE_DIR at the Docker mount; use the repo-local cache here.
os.environ.pop("SCOUT_CACHE_DIR", None)

from indication_scout.agents.supervisor.supervisor_tools import (  # noqa: E402
    _NOT_RANKED_REASON_ABSENT,
)

logger = logging.getLogger("indication_scout.scripts.ranked_list_completeness")

DEFAULT_REPORTS_DIR = PROJECT_ROOT / "test_reports"
DEFAULT_OUT_DIR = PROJECT_ROOT / "test_reports" / "completeness"

# Same shape finalize_supervisor uses for a ranked line: "N. <disease> — <tail>" or "N. <disease>".
_RANK_LINE = re.compile(r"^\s*(?P<rank>\d+)\.\s+(?P<head>.+?)(?:\s+—\s+(?P<tail>.+))?$")
_NOT_RANKED_LINE = re.compile(r"^\s*Not\s+ranked\s*:\s*(?P<body>.*)$", re.IGNORECASE)
_GATE_LINE = re.compile(
    r"^\s*Evidence\s+gate\s+exclusions\s*:\s*(?P<body>.*)$", re.IGNORECASE
)
# "<drug>_<YYYY-MM-DD_HH-MM-SS>.json", with optional infix such as "_holdout_2005-06-01".
_PAYLOAD_NAME = re.compile(
    r"^(?P<drug>[^_]+)(?:_.*)?_(?P<ts>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.json$"
)


def _norm(name: str) -> str:
    return name.strip().lower()


def _split_footer(body: str) -> dict[str, str]:
    """'a — reason; b — reason' -> {a: reason, b: reason}, keys normalised."""
    out: dict[str, str] = {}
    for entry in body.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        disease, _, reason = entry.partition(" — ")
        out[_norm(disease)] = reason.strip()
    return out


def classify(payload: dict) -> dict:
    """Bucket every investigated disease of one SupervisorOutput payload by where the summary put it."""
    investigated = [_norm(f["disease"]) for f in payload.get("disease_findings", [])]
    ranked: list[str] = []
    not_ranked: dict[str, str] = {}
    gate: dict[str, str] = {}
    for line in (payload.get("summary") or "").splitlines():
        m = _NOT_RANKED_LINE.match(line)
        if m:
            not_ranked.update(_split_footer(m.group("body")))
            continue
        m = _GATE_LINE.match(line)
        if m:
            gate.update(_split_footer(m.group("body")))
            continue
        m = _RANK_LINE.match(line)
        if m:
            ranked.append(_norm(m.group("head")))
    absent = sorted(d for d, r in not_ranked.items() if r == _NOT_RANKED_REASON_ABSENT)
    other_not_ranked = sorted(
        d for d, r in not_ranked.items() if r != _NOT_RANKED_REASON_ABSENT
    )
    accounted = set(ranked) | set(not_ranked) | set(gate)
    unaccounted = sorted(d for d in investigated if d not in accounted)
    return {
        "drug": payload.get("drug_name", ""),
        "investigated": investigated,
        "ranked": ranked,
        "absent": absent,
        "other_not_ranked": other_not_ranked,
        "gate_excluded": sorted(gate),
        "unaccounted": unaccounted,
    }


def load_reports(
    reports_dir: Path, since: str | None, drugs: set[str]
) -> list[tuple[str, dict]]:
    """Return (run label, payload) for every matching SupervisorOutput JSON in `reports_dir`."""
    runs: list[tuple[str, dict]] = []
    for path in sorted(reports_dir.glob("*.json")):
        m = _PAYLOAD_NAME.match(path.name)
        if not m:
            continue
        if since and m.group("ts")[:10] < since:
            continue
        if drugs and m.group("drug").lower() not in drugs:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("skipping %s: %s", path.name, e)
            continue
        if (
            not isinstance(payload, dict)
            or "disease_findings" not in payload
            or "summary" not in payload
        ):
            continue
        runs.append((path.stem, payload))
    return runs


async def live_runs(
    drugs: list[str], repeats: int, out_dir: Path
) -> list[tuple[str, dict]]:
    """Run the full pipeline `repeats` times per drug, saving each payload; return (label, payload) pairs."""
    from indication_scout.services.analysis_runner import run_analysis

    out_dir.mkdir(parents=True, exist_ok=True)
    runs: list[tuple[str, dict]] = []
    for drug in drugs:
        for i in range(repeats):
            logger.info("run %d/%d for %s", i + 1, repeats, drug)
            output, _report = await run_analysis(drug)
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            path = out_dir / f"{drug}_{ts}.json"
            path.write_text(output.model_dump_json(indent=2), encoding="utf-8")
            runs.append((path.stem, json.loads(path.read_text(encoding="utf-8"))))
    return runs


def report(runs: list[tuple[str, dict]]) -> int:
    """Print per-run and aggregate tables. Returns 1 if any disease is unaccounted for, else 0."""
    rows = [(label, classify(p)) for label, p in runs]
    header = f"{'run':<50} {'inv':>4} {'rank':>4} {'abs':>4} {'gate':>4} {'oth':>4} {'unacc':>5}"
    print(header)
    print("-" * len(header))
    for label, c in rows:
        print(
            f"{label:<50} {len(c['investigated']):>4} {len(c['ranked']):>4} {len(c['absent']):>4} "
            f"{len(c['gate_excluded']):>4} {len(c['other_not_ranked']):>4} {len(c['unaccounted']):>5}"
        )
        if c["absent"]:
            print(f"{'':<50}   absent: {', '.join(c['absent'])}")
        if c["unaccounted"]:
            print(f"{'':<50}   UNACCOUNTED: {', '.join(c['unaccounted'])}")

    n = len(rows)
    n_absent = sum(1 for _, c in rows if c["absent"])
    total_absent = sum(len(c["absent"]) for _, c in rows)
    n_unacc = sum(1 for _, c in rows if c["unaccounted"])
    print()
    print(f"runs: {n}")
    print(
        f"runs with >=1 absent candidate: {n_absent} ({100 * n_absent / n:.0f}%)"
        if n
        else "runs: 0"
    )
    print(f"absent candidates total: {total_absent}")
    print(f"runs with unaccounted diseases (neither ranked nor in a footer): {n_unacc}")

    # Cross-run stability: per drug, diseases ranked in some runs and absent in others.
    ranked_in: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    absent_in: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    runs_per_drug: dict[str, int] = defaultdict(int)
    for _, c in rows:
        runs_per_drug[c["drug"]] += 1
        for d in c["ranked"]:
            ranked_in[c["drug"]][d] += 1
        for d in c["absent"]:
            absent_in[c["drug"]][d] += 1
    unstable: list[tuple[str, str, int, int, int]] = []
    for drug, per_disease in absent_in.items():
        for disease, n_abs in per_disease.items():
            n_rank = ranked_in[drug].get(disease, 0)
            if n_rank:
                unstable.append((drug, disease, n_rank, n_abs, runs_per_drug[drug]))
    print()
    if unstable:
        print(
            "diseases ranked in some runs but absent in others (drug, disease, ranked, absent, runs):"
        )
        for drug, disease, n_rank, n_abs, total in sorted(unstable):
            print(f"  {drug:<16} {disease:<40} {n_rank:>3} {n_abs:>3} {total:>3}")
    else:
        print("no disease flips between ranked and absent across runs of the same drug")
    return 1 if n_unacc else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--reports", action="store_true", help="classify saved payload JSONs"
    )
    mode.add_argument(
        "--run", nargs="+", metavar="DRUG", help="run the pipeline for these drugs"
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
        help="payload dir for --reports",
    )
    parser.add_argument(
        "--since", help="only payloads dated on/after YYYY-MM-DD (--reports)"
    )
    parser.add_argument(
        "--drug", nargs="*", default=[], help="restrict --reports to these drugs"
    )
    parser.add_argument("--repeats", type=int, default=3, help="runs per drug (--run)")
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT_DIR, help="where --run saves payloads"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    logger.setLevel(logging.INFO)

    if args.reports:
        runs = load_reports(args.dir, args.since, {d.lower() for d in args.drug})
    else:
        runs = asyncio.run(live_runs(args.run, args.repeats, args.out))
    if not runs:
        print("no runs found")
        return 1
    return report(runs)


if __name__ == "__main__":
    sys.exit(main())
