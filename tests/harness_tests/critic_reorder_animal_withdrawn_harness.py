"""Test whether the ranking critic can demote a murine-only / withdrawn-trial candidate.

Bug (humira run, 2026-07-10): asthma ranked #1 over chronic recurrent multifocal osteomyelitis
(CRMO). Asthma's real evidence is a SINGLE murine OVA model + one Phase 2 trial that was WITHDRAWN
before enrolling. CRMO has real human observational use but 0 registry trials. The critic ranked
asthma first — and this harness shows WHY: the FACT string the critic sees today cannot express
either discriminator.

  - No "withdrawn" dev_stage tier: a withdrawn-only trial collapses to the same `untested` state as
    CRMO's zero-trial state (agents/_trial_signals.py). So asthma looks trial-equivalent to CRMO.
  - No animal/preclinical literature descriptor: a murine study grades as `weak, supports,
    drug_specific` — indistinguishable from a weak HUMAN observational study
    (models/model_evidence_summary.py has evidence_basis + is_observational, but no in_vivo/animal
    field).

Two modes, N runs each, over the REAL critic prompt (_RANKING_CRITIC_SYSTEM) in the exact
FACT-block + blurbs format _run_fact_critic uses:

  CURRENT  — FACT strings as the code emits them today. EXPECTED TO FAIL to reliably demote asthma:
             the data to justify the demotion isn't in the prompt.
  ENRICHED — FACT strings with the withdrawn count + animal-only tag added (the proposed plumbing).
             EXPECTED TO PASS: with the data present, the critic can reason CRMO above asthma.

If CURRENT already passes, the misorder is a pure prompt/variance issue, not missing data. If
CURRENT fails and ENRICHED passes, the fix is data plumbing (surface withdrawn + animal-only into
the FACT/ranking line), not prompt scolding.

Run: .venv/bin/python tests/harness_tests/critic_reorder_animal_withdrawn_harness.py [model]
"""

import asyncio
import json
import re
import sys

from anthropic import AsyncAnthropic

from indication_scout.config import get_settings
from indication_scout.agents.supervisor.supervisor_tools import _RANKING_CRITIC_SYSTEM

client = AsyncAnthropic(api_key=get_settings().anthropic_api_key)
MODEL = sys.argv[1] if len(sys.argv) > 1 else "claude-sonnet-4-6"
RUNS = 8

# Deliberately WRONG order: asthma (murine-only, withdrawn trial) is #1, CRMO (real human
# observational use, no registry trials) is below it. A correct critic ranks CRMO above asthma.

# CURRENT: FACT strings as the code emits them TODAY. Asthma's withdrawn trial collapses to
# `untested` (same tier as CRMO's no-trials state), and its murine study reads as plain
# `weak, supports` with no animal tag — so the two candidates look near-identical here.
FACT_LINES_CURRENT = [
    "1. asthma | FACT: authoritative dev_stage = untested (no completed or active pivotal "
    "trials for this indication) — do not contradict this stage | verdict: — | literature: "
    "weak, supports | blocker: —",
    "2. chronic recurrent multifocal osteomyelitis | FACT: authoritative dev_stage = untested "
    "(no registry trials for this indication) — do not contradict this stage | verdict: — | "
    "literature: weak, supports | blocker: —",
]

# ENRICHED: same two candidates, but the FACT now carries the two discriminators the proposed
# plumbing would surface — asthma's sole trial was WITHDRAWN before enrolling, and its sole study
# is a murine model; CRMO's evidence is human observational.
FACT_LINES_ENRICHED = [
    "1. asthma | FACT: authoritative dev_stage = untested (1 trial on record, WITHDRAWN before "
    "enrolling — never dosed a patient) — do not contradict this stage | verdict: — | "
    "literature: weak, supports — single murine (OVA mouse-model) study, no human data | "
    "blocker: —",
    "2. chronic recurrent multifocal osteomyelitis | FACT: authoritative dev_stage = untested "
    "(no registry trials) — do not contradict this stage | verdict: — | literature: weak, "
    "supports — human observational case series | blocker: —",
]

BLURBS_CURRENT = [
    {
        "disease": "asthma",
        "stage": "Early-phase only, no completed pivotal readout",
        "literature": "weak, supports",
        "blocker": "Evidence base remains limited to early-phase work",
        "verdict": "",
        "prose": "One registered trial is on record but no program is currently active.",
    },
    {
        "disease": "chronic recurrent multifocal osteomyelitis",
        "stage": "No registry trials on record for this indication",
        "literature": "weak, supports",
        "blocker": "No clinical development initiated",
        "verdict": "",
        "prose": "The hypothesis rests on observational literature with no registry trials.",
    },
]

BLURBS_ENRICHED = [
    {
        "disease": "asthma",
        "stage": "One trial withdrawn before enrolling; no active program",
        "literature": "weak, supports — murine model only, no human data",
        "blocker": "Sole trial never enrolled; evidence is animal-only",
        "verdict": "",
        "prose": "The single registered trial was withdrawn before dosing and the only "
        "supporting study is a mouse model.",
    },
    {
        "disease": "chronic recurrent multifocal osteomyelitis",
        "stage": "No registry trials on record for this indication",
        "literature": "weak, supports — human observational case series",
        "blocker": "No clinical development initiated",
        "verdict": "",
        "prose": "The hypothesis rests on human observational use with no registry trials.",
    },
]


def _task(fact_lines: list[str], blurbs: list[dict]) -> str:
    return (
        "Current ranking (top to bottom), each with its authoritative FACT:\n"
        + "\n".join(fact_lines)
        + "\n\nFull blurbs to audit and repair (return all of them):\n"
        + json.dumps(blurbs)
    )


def _order(text: str) -> list[str]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [
        (b.get("disease") or "").strip().lower()
        for b in data.get("blurbs", [])
        if isinstance(b, dict)
    ]


async def _run_one(fact_lines: list[str], blurbs: list[dict]) -> list[str]:
    resp = await client.messages.create(
        model=MODEL,
        max_tokens=1500,
        system=_RANKING_CRITIC_SYSTEM,
        messages=[{"role": "user", "content": _task(fact_lines, blurbs)}],
    )
    return _order(resp.content[0].text)


async def _run_mode(name: str, fact_lines: list[str], blurbs: list[dict]) -> None:
    results = await asyncio.gather(
        *(_run_one(fact_lines, blurbs) for _ in range(RUNS))
    )
    crmo = "chronic recurrent multifocal osteomyelitis"
    passes = 0
    for o in results:
        # PASS: CRMO (real human evidence) ranked above asthma (murine-only, withdrawn); set kept.
        ok = (
            len(o) == 2
            and o[0] == crmo
            and set(o) == {"asthma", crmo}
        )
        passes += ok
        if not ok:
            print(f"  [{name}] bad order:", o)
    print(f"[{name}] {passes}/{RUNS} ranked CRMO above asthma\n")


async def main() -> None:
    print(f"Model: {MODEL}  |  {RUNS} runs per mode\n")
    print("CURRENT  = FACT as emitted today (expected: fails to reliably demote asthma)")
    print("ENRICHED = FACT + withdrawn/animal-only (expected: demotes asthma)\n")
    await _run_mode("CURRENT", FACT_LINES_CURRENT, BLURBS_CURRENT)
    await _run_mode("ENRICHED", FACT_LINES_ENRICHED, BLURBS_ENRICHED)


if __name__ == "__main__":
    asyncio.run(main())
