"""Generate a holdout candidate-recall markdown table (seed phase only).

This is a validation probe, not part of the live pipeline. It answers one question per
runbook row: under a holdout cutoff, does the drug's known target indication actually
surface in the seed-phase candidate list, and at what rank? It exercises only the cheap
seed phase (mechanism + competitor surfacing + merge), so it runs in seconds per row
rather than the minutes a full lit/trials/synthesis report takes.

How a row is produced:

  1. Run the real supervisor seed-phase tools — `analyze_mechanism` and `find_candidates`
     — exactly as the ReAct loop fires them: both concurrently, with find_candidates
     awaiting the mechanism gate and then running the centralized merge_and_dedup. The
     result is snapshotted from `get_merged_allowlist()`: the ground-truth merged
     competitor + mechanism list in insertion order — the same order
     `investigate_top_candidates[:N]` slices. There is no OT-score re-ranking here; the
     mechanism set is the real signal-filtered/capped one, NOT the full association pool
     that probe_ot_score_rank.py inspects.
  2. Because every run passes `date_before=cutoff`, ranking is always holdout/leak-free:
     the mechanism score is the recomputed OT score with clinical_precedence excluded, so
     a post-cutoff approval signal can never inflate a candidate's rank.
  3. LLM-match the row's target indication into that merged list (synonyms/abbreviations
     count; broader parents and siblings do not) and record its rank + source.

One seed-phase run is performed per distinct (drug, cutoff); rows that share it reuse the
cached result. Rows are written as they complete so a mid-run crash never loses progress.

Output mirrors results/holdout_validation/validation_results_bak_9.md exactly:

  | Drug | Indication | Cutoff | Score | Rank | Notes | Source | Matched |

Score is candidate-presence (NOT the 1/0/-1 of validation_results.md):
  1  = present and investigated (rank <= cap)
  0  = present but rank > cap
  -1 = not in the merged list (or ERROR)

The Notes column is emitted empty; any prose in a committed table was added by hand.

Args: <runbook.txt> [output.md] [--lines 3-7,12]. Runbook columns: drug,indication,date.
`--lines` selects 1-based data rows (header excluded; ranges and lists allowed); default
runs every row. With no output.md given, writes to the next free
results/holdout_validation/validation_results_N.md (never overwrites).

Run (per-target read widened to 30 to test deeper recall):
    MECHANISM_ASSOCIATIONS_PER_TARGET=30 CONSTANTS_FILE=.env.constants \\
        .venv/bin/python scripts/validation/gen_candidate_recall.py \\
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
from indication_scout.services.llm import query_small_llm
from indication_scout.services.retrieval import RetrievalService

logging.basicConfig(level=logging.ERROR, format="%(message)s")
logger = logging.getLogger("gen_candidate_recall")

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


async def _match(target: str, names: list[str]) -> int | None:
    """LLM-resolve the target indication to an index in `names`, or None."""
    if not names:
        return None
    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(names))
    prompt = (
        "You map a target disease to a list of candidate disease names.\n"
        f'Target: "{target}"\n\n'
        "Candidates:\n"
        f"{numbered}\n\n"
        "Return ONLY the integer index of the candidate that best refers to the target "
        "indication — the SAME disease, an abbreviation, or a synonym. Do NOT match to a broader "
        "parent that drops the target's specificity, nor to a sibling under a shared parent. If no "
        "candidate fits, return -1. Return only the number."
    )
    resp = (await query_small_llm(prompt)).strip()
    try:
        idx = int(resp.split()[0])
    except (ValueError, IndexError):
        return None
    return idx if 0 <= idx < len(names) else None


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
    # Reflect the effective per-target value (a CLI env override beats the constants file).
    per_target = int(
        os.environ.get(
            "MECHANISM_ASSOCIATIONS_PER_TARGET", s.mechanism_associations_per_target
        )
    )
    env_line = (
        f"### env constants : SUPERVISOR_CANDIDATE_CAP={s.supervisor_candidate_cap} "
        f"SUPERVISOR_INVESTIGATION_CAP={s.supervisor_investigation_cap}, "
        f"MECHANISM_ASSOCIATIONS_PER_TARGET={per_target}"
    )
    lines = [
        "# Holdout Validation — leak-free candidate recall (seed phase only)",
        "",
        env_line,
        "",
        "Mechanism ranking uses the leak-free recomputed OT score (clinical_precedence excluded) in "
        f"holdout mode; `MECHANISM_ASSOCIATIONS_PER_TARGET={per_target}`; investigation cap={cap}.",
        "",
        "`present`: in = target indication is in the merged candidate list. `rank`/`source` = its "
        "position and origin (competitor/mechanism/both). `investigated` = rank <= cap (reaches the "
        "deep dive). This is candidate-presence, NOT the 1/0/-1 of validation_results.md.",
        "",
        "| Drug | Indication | Cutoff | Score | Rank | Notes | Source | Matched |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")


def _append_row(r: dict) -> None:
    # 1 = investigated, 0 = present but rank > cap, -1 = out/ERROR.
    if r["present"] == "in":
        score = "1" if r["investigated"] == "yes" else "0"
    else:
        score = "-1"
    with OUT_MD.open("a") as f:
        f.write(
            f"| {r['drug']} | {r['indication']} | {r['cutoff']} | {score} "
            f"| {r['rank']} | {r.get('note', '')} | {r['source']} "
            f"| {r.get('matched', '')} |\n"
        )


async def main() -> None:
    global OUT_MD
    argv = sys.argv[1:]

    # --lines 3-7,12 (or --lines=3-7,12) selects 1-based data rows; default = all.
    line_spec: str | None = None
    skip = set()
    for i, a in enumerate(argv):
        if a.startswith("--lines="):
            line_spec = a.split("=", 1)[1]
            skip.add(i)
        elif a == "--lines" and i + 1 < len(argv):
            line_spec = argv[i + 1]
            skip.update({i, i + 1})

    paths = [a for i, a in enumerate(argv) if i not in skip and not a.startswith("-")]
    if not paths:
        print(__doc__)
        sys.exit(1)
    rows = _rows(paths[0])
    if len(paths) > 1:
        OUT_MD = Path(paths[1])

    if line_spec:
        wanted = _parse_lines(line_spec)
        rows = [r for i, r in enumerate(rows, start=1) if i in wanted]

    settings = get_settings()
    cap = settings.supervisor_investigation_cap
    llm = ChatAnthropic(
        model=settings.llm_model,
        temperature=0,
        max_tokens=4096,
        anthropic_api_key=settings.anthropic_api_key,
    )
    svc = RetrievalService(DEFAULT_CACHE_DIR)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=_make_engine())

    _ensure_header(cap)

    # One seed-phase run per distinct (drug, cutoff).
    cache: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for r in rows:
        drug, indication, cutoff = r["drug"].strip(), r["indication"].strip(), r["date"].strip()
        key = (drug, cutoff)
        try:
            if key not in cache:
                cache[key] = await _merged_for_drug(
                    llm, svc, session_factory, drug, date.fromisoformat(cutoff)
                )
        except Exception as e:  # noqa: BLE001 - record ERROR, keep going
            logger.error("%s / %s -> ERROR: %s", drug, indication, e)
            _append_row({"drug": drug, "indication": indication, "cutoff": cutoff,
                         "present": "ERROR", "rank": "", "source": "", "investigated": ""})
            continue
        merged = cache[key]
        names = [n for n, _ in merged]
        idx = await _match(indication, names)
        if idx is None:
            row = {"drug": drug, "indication": indication, "cutoff": cutoff,
                   "present": "out", "rank": "", "source": "", "investigated": ""}
        else:
            name, source = merged[idx]
            row = {"drug": drug, "indication": indication, "cutoff": cutoff,
                   "present": "in", "rank": str(idx + 1), "source": source,
                   "investigated": "yes" if idx < cap else "no", "matched": name}
        _append_row(row)  # write as we go
        logger.error("%s / %s -> %s", drug, indication, row.get("rank") or row["present"])
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    asyncio.run(main())
