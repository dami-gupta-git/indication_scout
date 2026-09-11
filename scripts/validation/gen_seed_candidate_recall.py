"""Generate a holdout candidate-recall markdown table (seed phase only).

Validation probe, not part of the live pipeline. Per runbook row: under a holdout cutoff,
does the drug's known target indication surface in the seed-phase candidate list, and at
what rank? Runs only the cheap seed phase (mechanism + competitor surfacing + merge).

Per row: run `analyze_mechanism` and `find_candidates` concurrently as the ReAct loop does,
snapshot `get_merged_allowlist()` (merged competitor + mechanism list, insertion order — the
same order `investigate_top_candidates[:N]` slices), then LLM-match the target indication
into it. Every run passes `date_before=cutoff`, so the mechanism score excludes
clinical_precedence and no post-cutoff approval signal can inflate a rank.

One seed-phase run per distinct (drug, cutoff); rows sharing it reuse the cached result.
Rows are written as they complete.

The leading `#` column is the 1-based runbook data-row index — the same number `--lines` takes.

Score: 1 = one of the row's accepted names is in the merged list, 0 = none is, ERROR = run
failed. The accepted names are the runbook's `indication` plus its `accepted` column (semicolon-
separated), compared to the candidate names by exact equality after lowercasing and trimming.
Nothing else is inferred: no LLM matching, no fuzzy or substring matching. When a target surfaces
under a name that is not listed, the row reads 0 until the name is added to the runbook by hand.

Args: <runbook.txt> [output.md] [--lines 3-7,12]. Runbook columns: drug,indication,date,accepted
(accepted = semicolon-separated candidate names that count as the target, may be empty).
`--lines` selects 1-based data rows; default runs every row. With no output.md, writes to
the next free results/holdout_validation/validation_results_N.md (never overwrites).

Run (per-target read widened to 30 to test deeper recall):
    MECHANISM_ASSOCIATIONS_PER_TARGET=30 CONSTANTS_FILE=.env.constants \\
        .venv/bin/python scripts/validation/gen_seed_candidate_recall.py \\
        scripts/validation/runbook.txt --lines 1-5,48
"""

import asyncio
import csv
import logging
import os
import sys
from datetime import date
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from sqlalchemy.orm import sessionmaker

from indication_scout.agents.supervisor.supervisor_tools import build_supervisor_tools
from indication_scout.config import get_settings
from indication_scout.constants import DEFAULT_CACHE_DIR
from indication_scout.db.session import _make_engine
from indication_scout.helpers.drug_helpers import normalize_drug_name
from indication_scout.services.precision_metrics import (
    CandidatePrediction,
    stable_review_id,
)
from indication_scout.services.retrieval import RetrievalService

logging.basicConfig(level=logging.ERROR, format="%(message)s")
logger = logging.getLogger("gen_seed_candidate_recall")

VALIDATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VALIDATION_DIR.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "holdout_validation"


def _next_out_md() -> Path:
    """First unused validation_results_N.md (N starts at 11)."""
    n = 11
    while (RESULTS_DIR / f"validation_results_{n}.md").exists():
        n += 1
    return RESULTS_DIR / f"validation_results_{n}.md"


OUT_MD = _next_out_md()


def _tool_call(tool, drug: str) -> dict:
    """ToolCall-shaped input so .ainvoke returns a ToolMessage (matches the ReAct loop)."""
    return {
        "name": tool.name,
        "args": {"drug_name": drug},
        "id": f"probe_{tool.name}",
        "type": "tool_call",
    }


async def _merged_for_drug(
    llm, svc: RetrievalService, session_factory, drug: str, cutoff: date
) -> list[tuple[str, str]]:
    """Return the post-merge allowlist as [(canonical_name, source)] in insertion order."""
    drug = normalize_drug_name(drug)
    db = session_factory()
    try:
        tools, get_merged_allowlist, _, _ = build_supervisor_tools(
            llm=llm, svc=svc, db=db, session_factory=session_factory, date_before=cutoff
        )
        by_name = {t.name: t for t in tools}
        mech = by_name["analyze_mechanism"]
        find = by_name["find_candidates"]

        # Concurrent, like the ReAct loop; the mechanism gate enforces ordering.
        await asyncio.gather(
            mech.ainvoke(_tool_call(mech, drug)),
            find.ainvoke(_tool_call(find, drug)),
        )

        allowlist = get_merged_allowlist()  # lc -> (canonical, source)
        return [(canonical, source) for (canonical, source) in allowlist.values()]
    finally:
        db.close()


def _match_accepted(accepted: list[str], names: list[str]) -> int | None:
    """Index of the first candidate whose name equals one of the accepted names (lowercased, trimmed)."""
    wanted = {a.strip().lower() for a in accepted if a.strip()}
    for i, name in enumerate(names):
        if name.strip().lower() in wanted:
            return i
    return None


def _rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _parse_lines(spec: str) -> set[int]:
    """Parse a 1-based selector like '3-7,12,15' into a set of data-row numbers."""
    wanted: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            wanted.update(range(lo, hi + 1))
        elif part:
            wanted.add(int(part))
    return wanted


def _ensure_header(cap: int) -> None:
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    s = get_settings()
    # A CLI env override beats the constants file.
    per_target = int(
        os.environ.get(
            "MECHANISM_ASSOCIATIONS_PER_TARGET", s.mechanism_associations_per_target
        )
    )
    env_line = (
        f"### env constants : SUPERVISOR_CANDIDATE_CAP={s.supervisor_candidate_cap} "
        f"SUPERVISOR_INVESTIGATION_CAP={s.supervisor_investigation_cap}, "
        f"MECHANISM_ASSOCIATIONS_PER_TARGET={per_target}, "
        f"MECHANISM_TOP_CANDIDATES={s.mechanism_top_candidates}"
    )
    lines = [
        "# Holdout Validation — leak-free candidate recall (seed phase only)",
        "",
        env_line,
        "",
        "Mechanism ranking uses the leak-free recomputed OT score (clinical_precedence excluded) in "
        f"holdout mode; `MECHANISM_ASSOCIATIONS_PER_TARGET={per_target}`; investigation cap={cap}.",
        "",
        "`Score`: 1 = one of the row's accepted names (runbook `indication` + `accepted` column) is "
        "in the merged candidate list by exact name, 0 = none is. "
        "Holdout measures presence only (no ranking), so position does not affect the score. "
        "`List position`/`Source` = the target's 1-based spot in the merged list and its origin "
        "(competitor/mechanism/both), for context only.",
        "",
        "| # | Drug | Indication | Cutoff | Score | List position | Notes | Source | Matched |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")


def _append_row(r: dict) -> None:
    if r["present"] == "ERROR":
        score = "ERROR"
    else:
        score = "1" if r["present"] == "in" else "0"
    with OUT_MD.open("a") as f:
        f.write(
            f"| {r['n']} | {r['drug']} | {r['indication']} | {r['cutoff']} | {score} "
            f"| {r['rank']} | {r.get('note', '')} | {r['source']} "
            f"| {r.get('matched', '')} |\n"
        )


def _write_candidate_predictions(
    writer: csv.DictWriter,
    drug: str,
    cutoff: str,
    merged: list[tuple[str, str]],
) -> None:
    """Write the candidates eligible to enter the configured investigation fan-out."""
    settings = get_settings()
    for position, (disease, source) in enumerate(
        merged[: settings.supervisor_investigation_cap], start=1
    ):
        prediction = CandidatePrediction(
            review_id=stable_review_id(drug, cutoff, disease),
            drug=drug,
            cutoff=date.fromisoformat(cutoff),
            position=position,
            source=source,
            disease=disease,
            investigation_limit=settings.supervisor_investigation_cap,
            supervisor_candidate_cap=settings.supervisor_candidate_cap,
            mechanism_associations_per_target=settings.mechanism_associations_per_target,
            mechanism_top_candidates=settings.mechanism_top_candidates,
        )
        writer.writerow(
            {
                **prediction.model_dump(mode="json"),
            }
        )


async def main() -> None:
    global OUT_MD
    argv = sys.argv[1:]

    # --lines 3-7,12 or --lines=3-7,12; default = all.
    line_spec: str | None = None
    candidate_out: Path | None = None
    candidates_only = False
    skip = set()
    for i, a in enumerate(argv):
        if a.startswith("--lines="):
            line_spec = a.split("=", 1)[1]
            skip.add(i)
        elif a == "--lines" and i + 1 < len(argv):
            line_spec = argv[i + 1]
            skip.update({i, i + 1})
        elif a.startswith("--candidate-out="):
            candidate_out = Path(a.split("=", 1)[1])
            skip.add(i)
        elif a == "--candidate-out" and i + 1 < len(argv):
            candidate_out = Path(argv[i + 1])
            skip.update({i, i + 1})
        elif a == "--candidates-only":
            candidates_only = True
            skip.add(i)

    paths = [a for i, a in enumerate(argv) if i not in skip and not a.startswith("-")]
    if not paths:
        print(__doc__)
        sys.exit(1)
    rows = _rows(paths[0])
    if len(paths) > 1:
        OUT_MD = Path(paths[1])

    # The number is the 1-based runbook data-row index, so it survives --lines and stays the
    # selector you would pass back to re-run that row.
    numbered = list(enumerate(rows, start=1))
    if line_spec:
        wanted = _parse_lines(line_spec)
        numbered = [(i, r) for i, r in numbered if i in wanted]

    settings = get_settings()
    cap = settings.supervisor_investigation_cap
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

    candidate_handle = None
    candidate_writer = None
    if candidate_out is not None:
        candidate_out.parent.mkdir(parents=True, exist_ok=True)
        candidate_handle = candidate_out.open("w", newline="", encoding="utf-8")
        candidate_writer = csv.DictWriter(
            candidate_handle,
            fieldnames=list(CandidatePrediction.model_fields),
        )
        candidate_writer.writeheader()

    if candidates_only:
        if candidate_writer is None or candidate_handle is None:
            raise ValueError("--candidates-only requires --candidate-out")
        unique_runs = list(
            dict.fromkeys((r["drug"].strip(), r["date"].strip()) for _, r in numbered)
        )
        semaphore = asyncio.Semaphore(settings.supervisor_investigation_concurrency)

        async def export_one(
            drug: str, cutoff: str
        ) -> tuple[str, str, list[tuple[str, str]] | None, Exception | None]:
            async with semaphore:
                try:
                    merged = await _merged_for_drug(
                        llm,
                        svc,
                        session_factory,
                        drug,
                        date.fromisoformat(cutoff),
                    )
                except Exception as error:  # noqa: BLE001 - record failure and continue
                    return drug, cutoff, None, error
                return drug, cutoff, merged, None

        try:
            tasks = [export_one(drug, cutoff) for drug, cutoff in unique_runs]
            for completed in asyncio.as_completed(tasks):
                drug, cutoff, merged, error = await completed
                if error is not None or merged is None:
                    logger.error("%s / %s -> ERROR: %s", drug, cutoff, error)
                    continue
                _write_candidate_predictions(candidate_writer, drug, cutoff, merged)
                candidate_handle.flush()
                logger.error(
                    "%s / %s -> exported %d candidates",
                    drug,
                    cutoff,
                    min(len(merged), settings.supervisor_investigation_cap),
                )
        finally:
            candidate_handle.close()
        return

    _ensure_header(cap)

    # One seed-phase run per distinct (drug, cutoff).
    cache: dict[tuple[str, str], list[tuple[str, str]]] = {}
    try:
        for n, r in numbered:
            drug, indication, cutoff = (
                r["drug"].strip(),
                r["indication"].strip(),
                r["date"].strip(),
            )
            key = (drug, cutoff)
            try:
                if key not in cache:
                    cache[key] = await _merged_for_drug(
                        llm, svc, session_factory, drug, date.fromisoformat(cutoff)
                    )
                    if candidate_writer is not None:
                        _write_candidate_predictions(
                            candidate_writer, drug, cutoff, cache[key]
                        )
                        candidate_handle.flush()
            except Exception as e:  # noqa: BLE001 - record ERROR, keep going
                logger.error("%s / %s -> ERROR: %s", drug, indication, e)
                _append_row(
                    {
                        "n": n,
                        "drug": drug,
                        "indication": indication,
                        "cutoff": cutoff,
                        "present": "ERROR",
                        "rank": "",
                        "source": "",
                    }
                )
                continue
            merged = cache[key]
            names = [n for n, _ in merged]
            accepted = [indication] + [
                a for a in (r.get("accepted") or "").split(";") if a.strip()
            ]
            idx = _match_accepted(accepted, names)
            if idx is None:
                row = {
                    "n": n,
                    "drug": drug,
                    "indication": indication,
                    "cutoff": cutoff,
                    "present": "out",
                    "rank": "",
                    "source": "",
                }
            else:
                name, source = merged[idx]
                row = {
                    "n": n,
                    "drug": drug,
                    "indication": indication,
                    "cutoff": cutoff,
                    "present": "in",
                    "rank": str(idx + 1),
                    "source": source,
                    "matched": name,
                }
            _append_row(row)  # write as we go
            logger.error(
                "%s / %s -> %s", drug, indication, row.get("rank") or row["present"]
            )
    finally:
        if candidate_handle is not None:
            candidate_handle.close()
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    asyncio.run(main())
