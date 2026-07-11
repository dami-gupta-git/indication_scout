"""Measure how reliably the ranking critic demotes a withdrawn-only / animal-only candidate below a
human-observational one — the humira × asthma vs CRMO case.

Uses THREE candidates (asthma, CRMO, T1DM) to match what the live run feeds the critic. Asthma's
FACT carries the real tension: dev_stage untested BUT "1 trial on record" (withdrawn), plus
animal/in-vitro-only literature — so we test whether "a trial exists" wrongly outweighs the
withdrawn + animal caveats.

Drives the REAL critic prompt (_RANKING_CRITIC_SYSTEM) in the exact FACT-block + blurbs format
_run_fact_critic builds. Real disease names (the critic must rank on the FACTs regardless).
PASS = CRMO ranked above asthma.

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

# WRONG input order (asthma #1) mirroring the bad live run. FACT strings as _run_fact_critic emits
# them now (withdrawn clause + the animal-only clause both present).
FACT_LINES = [
    "1. asthma | FACT: authoritative dev_stage = untested (no completed or active pivotal trials "
    "for this indication) — do not contradict this stage; all 1 on-record trial(s) WITHDRAWN "
    "before enrolling (never dosed a patient — not a live registered trial); supporting "
    "literature is ANIMAL/in-vitro only (no human data — not clinical evidence) | verdict: — | "
    "literature: weak, supports, animal/in-vitro only | blocker: —",
    "2. chronic recurrent multifocal osteomyelitis | FACT: authoritative dev_stage = untested (no "
    "registry trials for this indication) — do not contradict this stage | verdict: — | "
    "literature: weak, supports, observational (human case report) | blocker: —",
    "3. type 1 diabetes mellitus | FACT: authoritative dev_stage = untested — do not contradict "
    "this stage; supporting literature is class-level only (no direct evidence for this drug) | "
    "verdict: — | literature: class-level signal (no direct evidence for this drug) | blocker: —",
]

BLURBS = [
    {
        "disease": "asthma",
        "stage": "Early-phase only, no completed pivotal readout",
        "literature": "weak, supports, animal/in-vitro only",
        "blocker": "Evidence confined to preclinical models; sole trial withdrawn",
        "verdict": "",
        "prose": "One registered trial is on record but was withdrawn; supporting literature is "
        "limited to animal and in-vitro work.",
    },
    {
        "disease": "chronic recurrent multifocal osteomyelitis",
        "stage": "No registry trials on record for this indication",
        "literature": "weak, supports, observational (human case report)",
        "blocker": "No clinical development initiated",
        "verdict": "",
        "prose": "The hypothesis rests on human observational use with no registry trials.",
    },
    {
        "disease": "type 1 diabetes mellitus",
        "stage": "Early-phase only, no completed pivotal readout",
        "literature": "class-level signal (no direct evidence for this drug)",
        "blocker": "No drug-specific evidence",
        "verdict": "",
        "prose": "Only class-level signal; no direct adalimumab evidence.",
    },
]


def _task() -> str:
    return (
        "Current ranking (top to bottom), each with its authoritative FACT:\n"
        + "\n".join(FACT_LINES)
        + "\n\nFull blurbs to audit and repair (return all of them):\n"
        + json.dumps(BLURBS)
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


async def _run_one() -> list[str]:
    resp = await client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=_RANKING_CRITIC_SYSTEM,
        messages=[{"role": "user", "content": _task()}],
    )
    return _order(resp.content[0].text)


async def main() -> None:
    print(f"Model: {MODEL}  |  {RUNS} runs\n")
    crmo = "chronic recurrent multifocal osteomyelitis"
    results = await asyncio.gather(*(_run_one() for _ in range(RUNS)))
    passes = 0
    for o in results:
        ok = crmo in o and "asthma" in o and o.index(crmo) < o.index("asthma")
        passes += ok
        print(("  PASS" if ok else "  FAIL"), o)
    print(f"\nTOTAL: {passes}/{RUNS} ranked CRMO above asthma")


if __name__ == "__main__":
    asyncio.run(main())
