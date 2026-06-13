"""Holdout validation harness.

Reads drug,indication,date rows from runbook.xt, runs `scout find -d <drug>
--date-before <date>` for each, then scores whether the known runbook indication
shows up in the holdout report:

    1   indication matches a disease in the report's Summary (ranked signals)
    0   indication matches only a disease in 'Diseases Considered'
   -1   indication not found in either section

Matching is an LLM equivalence judgement (not substring) so synonyms,
abbreviations, and differing surface forms ("CML" vs "Chronic Myelogenous
Leukemia", "MASH" vs "metabolic dysfunction-associated steatohepatitis") match
correctly. The candidate sets are restricted to the report's own Summary and
'Diseases Considered' lists, so the judge can only match diseases the run
actually grounded in upstream data.
"""

import asyncio
import csv
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

VALIDATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VALIDATION_DIR.parent.parent


def _load_env() -> None:
    """Mirror the CLI's env loading so the in-process LLM judge sees keys/constants.

    Must run before importing the LLM service, which reads settings at import time.
    """
    load_dotenv(PROJECT_ROOT / ".env")
    # Validation must mirror production, not the cheap test profile.
    constants_file = os.environ.get("CONSTANTS_FILE", ".env.constants")
    constants_path = PROJECT_ROOT / constants_file
    load_dotenv(constants_path)
    os.environ["CONSTANTS_FILE"] = str(constants_path)


_load_env()

from indication_scout.services.llm import query_small_llm, strip_markdown_fences  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("validation")
RUNBOOK = VALIDATION_DIR / "runbook.xt"
REPORTS_DIR = PROJECT_ROOT / "results" / "holdout_validation"
RESULTS = REPORTS_DIR / "validation_results.md"
HOLDOUTS_DIR = PROJECT_ROOT / "snapshots" / "holdouts"
LOGS_DIR = REPORTS_DIR / "logs"  # per-row scout stdout+stderr (TIMING/429 internals)

JUDGE_PROMPT = """You are validating a drug-repurposing pipeline.

A known approved indication for the drug (the TARGET) is:
  "{indication}"

The pipeline produced these candidate diseases (the CANDIDATES):
{candidates}

Decide whether the pipeline actually found the TARGET. Do this by classifying,
for the BEST candidate, its clinical relationship to the TARGET — then apply the
match rule. Do NOT skip to a yes/no; classify first.

relationship (pick exactly one for the best candidate):
- "exact"          — same disease, same name or trivially so.
- "synonym"        — same disease, different name/abbreviation
                     (e.g. "CML" ↔ "Chronic Myelogenous Leukemia";
                      "MASH" ↔ "metabolic dysfunction-associated steatohepatitis";
                      "smoking cessation" ↔ "nicotine dependence").
- "candidate_narrower" — the candidate is a MORE SPECIFIC subtype of the TARGET
                     (the candidate sits UNDER the target in the disease hierarchy).
- "candidate_broader"  — the candidate is a BROADER umbrella/parent that merely
                     CONTAINS the target (target sits under the candidate),
                     e.g. target "seasonal affective disorder" vs candidate
                     "depressive disorder"; target "nicotine dependence" vs
                     candidate "substance use disorder".
- "sibling"        — both sit under a shared parent but are distinct
                     (e.g. "hypereosinophilic syndrome" vs "leukemia";
                      "nicotine dependence" vs "cocaine dependence").
- "unrelated"      — different conditions.

MATCH RULE (strict): match = true ONLY when relationship is "exact", "synonym",
or "candidate_narrower". A "candidate_broader" relationship is NOT a match — a
broad umbrella surfacing does not mean the specific TARGET was found. "sibling"
and "unrelated" are never matches.

Reply with ONLY a JSON object, no prose:
{{"relationship": "<one of the values above>", "match": true|false, "matched_disease": "<exact candidate name, or empty string>"}}"""


def read_runbook(start: int = 0, count: int | None = None) -> list[dict[str, str]]:
    """Parse runbook.xt into a list of {drug, indication, date} rows.

    start: 0-based index of the first data row to include.
    count: number of rows from start (None = to end).
    """
    with RUNBOOK.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    end = None if count is None else start + count
    return rows[start:end]


def newest_holdout(drug: str, date: str) -> Path | None:
    """Return the most recent holdout report for a drug/cutoff, or None."""
    matches = sorted(HOLDOUTS_DIR.glob(f"{drug}_holdout_{date}_*.md"))
    return matches[-1] if matches else None


SCOUT_TIMEOUT_SECONDS = 900  # a run is normally ~3 min; kill a hung run past 15 min


def run_scout(drug: str, date: str, force: bool = False) -> Path | None:
    """Get the holdout report for a drug/cutoff, running `scout find` only if needed.

    The saved report (snapshots/holdouts/<drug>_holdout_<date>_*.md) is the run's
    durable artifact. If one already exists it is reused, so tweaking the scoring
    logic costs seconds instead of re-running the ~3-min pipeline. Pass force=True
    to regenerate regardless.

    Returns None on non-zero exit or on timeout (the hung child is killed so one
    stuck run doesn't block the rest of the batch).
    """
    if not force:
        existing = newest_holdout(drug, date)
        if existing is not None:
            logger.info("Reusing existing report: %s", existing.name)
            return existing

    cmd = ["scout", "find", "-d", drug, "--date-before", date]
    logger.info("Running: %s", " ".join(cmd))
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"{drug}_{date}.log"
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=SCOUT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as e:
        # Persist whatever the child emitted before the kill so a hang is inspectable.
        partial = (e.stdout or "") + (e.stderr or "")
        log_path.write_text(partial, encoding="utf-8")
        logger.error(
            "scout TIMED OUT after %ds for %s/%s — killed (log: %s)",
            SCOUT_TIMEOUT_SECONDS, drug, date, log_path,
        )
        return None
    # Tee the full scout stdout+stderr (TIMING / 429 retries / model load) per row.
    log_path.write_text((result.stdout or "") + (result.stderr or ""), encoding="utf-8")
    if result.returncode != 0:
        logger.error(
            "scout failed for %s/%s (log: %s):\n%s",
            drug, date, log_path, result.stderr[-2000:],
        )
        return None
    return newest_holdout(drug, date)


def _section_lines(lines: list[str], header: str) -> list[str]:
    """Return the lines between a '## <header>' line and the next '## ' or '---'."""
    out: list[str] = []
    in_section = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## ") and stripped[3:].strip() == header:
            in_section = True
            continue
        if in_section:
            if stripped.startswith("## ") or stripped.startswith("---"):
                break
            out.append(line)
    return out


def extract_summary(report_path: Path) -> list[str]:
    """Entries from the Summary's numbered ranked list.

    Lines look like: '1. Chronic Myelogenous Leukemia — literature: ...'. We strip
    only the 'N.' prefix and hand the rest to the LLM judge, which extracts the
    disease itself — avoiding fragile dash-splitting on the name/stats boundary.
    """
    lines = report_path.read_text(encoding="utf-8").splitlines()
    diseases: list[str] = []
    for line in _section_lines(lines, "Summary"):
        m = re.match(r"\s*\d+\.\s+(.*)", line)
        if m:
            diseases.append(m.group(1).strip())
    return diseases


def extract_considered(report_path: Path) -> list[str]:
    """Disease names from the 'Diseases Considered' bullet list."""
    lines = report_path.read_text(encoding="utf-8").splitlines()
    diseases: list[str] = []
    for line in _section_lines(lines, "Diseases Considered"):
        stripped = line.strip()
        if stripped.startswith("- "):
            diseases.append(stripped[2:].strip())
    return diseases


async def judge_match(indication: str, candidates: list[str]) -> tuple[bool, str]:
    """Ask the LLM whether the indication matches any candidate; return (match, name)."""
    if not candidates:
        return False, ""
    prompt = JUDGE_PROMPT.format(
        indication=indication,
        candidates="\n".join(f"- {c}" for c in candidates),
    )
    response = await query_small_llm(prompt)
    try:
        parsed = json.loads(strip_markdown_fences(response))
    except json.JSONDecodeError:
        logger.error("Judge returned non-JSON for '%s': %s", indication, response)
        return False, ""
    # Enforce the strict rule in code, not just the prompt: only exact / synonym /
    # candidate_narrower count, regardless of the LLM's own `match` flag. A broad
    # umbrella surfacing ("candidate_broader") is never a find.
    relationship = str(parsed.get("relationship", "")).strip().lower()
    matched = relationship in {"exact", "synonym", "candidate_narrower"}
    if matched != bool(parsed.get("match")):
        logger.warning(
            "Judge match flag (%s) overridden by relationship=%r for '%s'",
            parsed.get("match"), relationship, indication,
        )
    return matched, str(parsed.get("matched_disease", "")) if matched else ""


async def score_report(report_path: Path, indication: str) -> tuple[int, str]:
    """Score the indication against Summary (1) then Diseases Considered (0); else -1."""
    in_summary, name = await judge_match(indication, extract_summary(report_path))
    if in_summary:
        return 1, name
    in_considered, name = await judge_match(indication, extract_considered(report_path))
    if in_considered:
        return 0, name
    return -1, ""


async def process_row(row: dict[str, str], force: bool = False) -> dict[str, str]:
    """Run scout for one runbook row, score it, and write its result immediately.

    The row is appended to the report the moment it finishes, so a later hang,
    timeout, or kill never loses an already-completed entry. With force=False an
    existing holdout report is reused instead of re-running the pipeline.
    """
    drug, indication, date = row["drug"], row["indication"], row["date"]
    report = run_scout(drug, date, force=force)
    if report is None:
        result = {
            "drug": drug,
            "indication": indication,
            "date": date,
            "score": "ERROR",
            "matched": "",
            "report": "",
        }
        append_result(result)
        return result
    score, matched_name = await score_report(report, indication)
    result = {
        "drug": drug,
        "indication": indication,
        "date": date,
        "score": str(score),
        "matched": matched_name,
        "report": report.relative_to(PROJECT_ROOT).as_posix(),
    }
    logger.info("%s / %s -> score %s (%s)", drug, indication, score, matched_name)
    append_result(result)
    return result


def ensure_header() -> None:
    """Write the report title, legend, and table header once, if not already present."""
    if RESULTS.exists() and RESULTS.stat().st_size > 0:
        return
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("a", encoding="utf-8") as f:
        f.write("# Holdout Validation Results\n\n")
        f.write("_Score: 1 = in Summary, 0 = in Diseases Considered only, -1 = not found._\n\n")
        f.write(
            "| Drug | Runbook indication | Cutoff | Score | Matched disease "
            "| Notes | Report |\n"
        )
        f.write("|---|---|---|---|---|---|---|\n")


def append_result(result: dict[str, str]) -> None:
    """Append a single result row to validation_results.md (header ensured first)."""
    ensure_header()
    matched = result["matched"].replace("|", "\\|")
    note = result.get("note", "").replace("|", "\\|")
    with RESULTS.open("a", encoding="utf-8") as f:
        f.write(
            f"| {result['drug']} | {result['indication']} | {result['date']} "
            f"| {result['score']} | {matched} | {note} | {result['report']} |\n"
        )


async def main() -> None:
    # Usage: run_validation.py [count] [start] [--force]
    #   count   : number of rows to process (default: all)
    #   start   : 0-based index of first row (default: 0)
    #   --force : regenerate reports even if a saved one exists (default: reuse)
    argv = sys.argv[1:]
    force = "--force" in argv
    positional = [a for a in argv if a != "--force"]
    count = int(positional[0]) if len(positional) > 0 else None
    start = int(positional[1]) if len(positional) > 1 else 0
    rows = read_runbook(start=start, count=count)
    logger.info(
        "Validating %d runbook entries (start=%d, force=%s)", len(rows), start, force
    )

    # Serial: each run is heavy. process_row writes its own result row as it
    # finishes, so a hang/kill mid-batch never loses completed entries. Each row
    # is isolated — an unexpected failure on one is recorded as ERROR and the
    # remaining rows still run.
    for row in rows:
        try:
            await process_row(row, force=force)
        except Exception:
            logger.exception(
                "Unexpected failure on %s / %s — recording ERROR, continuing",
                row.get("drug"),
                row.get("indication"),
            )
            append_result(
                {
                    "drug": row.get("drug", ""),
                    "indication": row.get("indication", ""),
                    "date": row.get("date", ""),
                    "score": "ERROR",
                    "matched": "",
                    "report": "",
                }
            )

    logger.info("Done; results in %s", RESULTS)


if __name__ == "__main__":
    asyncio.run(main())
