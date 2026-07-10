"""Test that the supervisor's `watch` blurb field only cites THIS candidate's own NCTs.

Bug observed (semaglutide run): Parkinson (#3, "None active", only trial NCT03659682) got
"Watch: NCT06082063" — a Type 1 Diabetes trial. An NCT from one candidate's trial set leaked into
another candidate's watch line. `watch` is LLM-authored in supervisor.txt; the fix is a one-line
prompt rule ("before citing an NCT, check it belongs to THIS candidate's trials").

This harness feeds the REAL supervisor.txt as the system prompt and several candidates, each with
its OWN distinct NCT ids, then asks for blurbs with watch lines. It asserts that each candidate's
watch cites ONLY its own NCTs — never another candidate's. The crux candidate (Parkinson-shape) has
NO active trial, so its watch should be empty.

Run: .venv/bin/python tests/harness_tests/watch_nct_harness.py [model]
"""

import asyncio
import json
import re
import sys
from pathlib import Path

from anthropic import AsyncAnthropic

from indication_scout.config import get_settings

client = AsyncAnthropic(api_key=get_settings().anthropic_api_key)
MODEL = sys.argv[1] if len(sys.argv) > 1 else "claude-sonnet-4-6"
RUNS = 8

_PROMPTS_DIR = Path(__file__).parent.parent / "src" / "indication_scout" / "prompts"
SUPERVISOR_PROMPT = (_PROMPTS_DIR / "supervisor.txt").read_text()

# Each candidate: own NCTs only. "Alpha" is the active-Phase-3 one (lots of live NCTs the model
# might be tempted to reuse); "Gamma" is the Parkinson-shape: no active trial, one inactive NCT.
CANDIDATES = [
    {
        "disease": "Alphadiabetes",
        "own_ncts": ["NCT11110001", "NCT11110002", "NCT11110003", "NCT11110004"],
        "ct_summary": (
            "Active Phase 3 development. Recruiting Phase 3 trials: NCT11110001 (CV outcomes), "
            "NCT11110002 (adjunct therapy), NCT11110003 (combination), NCT11110004 (not yet "
            "recruiting). Multiple ongoing readouts expected."
        ),
        "lit": "Literature: 6 PMIDs, strength=moderate, direction=supports, study_design=rct_or_controlled, evidence_basis=drug_specific.",
        "active_programs": "4 Phase 3 recruiting (NCT11110001, NCT11110002, NCT11110003, NCT11110004)",
    },
    {
        "disease": "Betasteatosis",
        "own_ncts": ["NCT22220001", "NCT22220002"],
        "ct_summary": (
            "Phase 2 completed for this indication, no Phase 3. Completed: NCT22220001 (Phase 2), "
            "NCT22220002 (Phase 2). No active pivotal program."
        ),
        "lit": "Literature: 0 PMIDs, strength=none, direction=none, study_design=undetermined, evidence_basis=approved.",
        "active_programs": "None active",
    },
    {
        "disease": "Gammakinson",  # the Parkinson-shape crux: no active trial
        "own_ncts": ["NCT33330001"],
        "ct_summary": (
            "Early-phase only, no completed pivotal readout. A single Phase 2 study NCT33330001 "
            "has unknown status. No active or recruiting trials."
        ),
        "lit": "Literature: 0 PMIDs, strength=none, direction=none, study_design=undetermined, evidence_basis=class_level.",
        "active_programs": "None active",
    },
]

ALL_NCTS = {n for c in CANDIDATES for n in c["own_ncts"]}


def _task() -> str:
    blocks = []
    for c in CANDIDATES:
        blocks.append(
            f"### Candidate: {c['disease']}\n"
            f"{c['lit']}\n"
            f"clinical_trials.summary: {c['ct_summary']}\n"
            f"active_programs: {c['active_programs']}"
        )
    body = "\n\n".join(blocks)
    names = [c["disease"] for c in CANDIDATES]
    return (
        "You investigated these candidates for drug DRUG this run (both sub-agent calls ran for "
        "each). Write the per-candidate blurb fields, especially `watch`, following your "
        "instructions.\n\n"
        f"{body}\n\n"
        f"Candidates: {names}\n\n"
        'Respond with ONLY a JSON object: {"blurbs": [{"disease": "...", "watch": "..."}, ...]} '
        "with one entry per candidate. watch is a string (may be empty)."
    )


def _ncts_in(text: str) -> set[str]:
    return set(re.findall(r"NCT\d+", text or ""))


def _parse(text: str) -> list[dict]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return []
    try:
        return list(json.loads(m.group(0)).get("blurbs", []))
    except json.JSONDecodeError:
        return []


async def _run_one() -> tuple[bool, list[str]]:
    resp = await client.messages.create(
        model=MODEL,
        max_tokens=600,
        system=SUPERVISOR_PROMPT,
        messages=[{"role": "user", "content": _task()}],
    )
    blurbs = _parse(resp.content[0].text)
    violations: list[str] = []
    own_by_disease = {c["disease"].lower(): set(c["own_ncts"]) for c in CANDIDATES}
    for b in blurbs:
        d = (b.get("disease") or "").strip().lower()
        cited = _ncts_in(b.get("watch", ""))
        own = own_by_disease.get(d, set())
        foreign = cited - own
        # An NCT not belonging to this candidate (whether another candidate's or invented).
        if foreign:
            violations.append(f"{b.get('disease')}: watch cites foreign {sorted(foreign)}")
        # Gammakinson (no active trial) should have an empty watch.
        if d == "gammakinson" and cited:
            violations.append(f"Gammakinson: watch should be empty, cites {sorted(cited)}")
    return (not violations), violations


async def main() -> None:
    print(f"Model: {MODEL}  |  {RUNS} runs\n")
    results = await asyncio.gather(*(_run_one() for _ in range(RUNS)))
    passes = sum(1 for ok, _ in results if ok)
    for i, (ok, v) in enumerate(results):
        if not ok:
            print(f"  run {i}: {v}")
    print(f"\nTOTAL: {passes}/{RUNS} clean (no foreign NCT in any watch line)")


if __name__ == "__main__":
    asyncio.run(main())
