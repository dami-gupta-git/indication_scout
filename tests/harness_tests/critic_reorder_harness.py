"""Test that the ranking critic REORDERS a bad order (not just confirms a good one).

Bug: imatinib run put GBM (strong, CONTRADICTS, closed — disproven) at #1 over live candidates.
The supervisor's ranking is variance-prone; the restored critic (_RANKING_CRITIC_SYSTEM) is the
guard. This harness feeds the critic a DELIBERATELY WRONG order (the disproven candidate first) in
the exact FACT-block + blurbs format the code uses, and asserts the critic moves the
tested-and-failed / closed candidate to last, keeping the live ones on top — by reasoning, not a
spelled-out rule.

Run: .venv/bin/python tests/harness_tests/critic_reorder_harness.py [model]
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

# Deliberately WRONG order: the disproven/closed candidate is #1, live candidates below it.
# FACT lines mirror _run_fact_critic's format (disease | FACT | verdict | literature | blocker).
FACT_LINES = [
    "1. glioblastoma multiforme | FACT: authoritative dev_stage = completed_phase3 (Phase 3 "
    "completed for this indication) — do not contradict this stage | verdict: — | literature: "
    "Strong, contradicts — multiple trials show no benefit | blocker: —",
    "2. breast cancer | FACT: authoritative dev_stage = completed_phase2 (Phase 2 completed, no "
    "Phase 3) — do not contradict this stage | verdict: — | literature: Moderate, mixed — some "
    "supportive signals | blocker: —",
    "3. leukemia | FACT: authoritative dev_stage = completed_phase3 (Phase 3 completed) — do not "
    "contradict this stage | verdict: — | literature: None (approved sub-indication basis) | "
    "blocker: —",
]

BLURBS = [
    {
        "disease": "glioblastoma multiforme",
        "stage": "Phase 3 completed for this indication",
        "literature": "strong, contradicts",
        "blocker": "Controlled trials contradict benefit; development abandoned",
        "verdict": "",
        "prose": "RCT-backed evidence has effectively disproven the hypothesis.",
    },
    {
        "disease": "breast cancer",
        "stage": "Phase 2 completed for this indication, no Phase 3",
        "literature": "moderate, mixed",
        "blocker": "No pivotal program; mixed evidence",
        "verdict": "",
        "prose": "Mixed controlled evidence; one non-pivotal study ongoing.",
    },
    {
        "disease": "leukemia",
        "stage": "Phase 3 completed for this indication",
        "literature": "evidence is for an already-approved sub-indication (not repurposing)",
        "blocker": "Entangled with an approved indication",
        "verdict": "",
        "prose": "Evidence inseparable from an adjacent approved use.",
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
        max_tokens=1500,
        system=_RANKING_CRITIC_SYSTEM,
        messages=[{"role": "user", "content": _task()}],
    )
    return _order(resp.content[0].text)


async def main() -> None:
    print(f"Model: {MODEL}  |  {RUNS} runs\n")
    results = await asyncio.gather(*(_run_one() for _ in range(RUNS)))
    passes = 0
    for o in results:
        # PASS: GBM (disproven/closed) is no longer #1, and is last; set preserved.
        ok = (
            len(o) == 3
            and o[-1] == "glioblastoma multiforme"
            and o[0] != "glioblastoma multiforme"
            and set(o) == {"glioblastoma multiforme", "breast cancer", "leukemia"}
        )
        passes += ok
        if not ok:
            print("  bad order:", o)
    print(f"\nTOTAL: {passes}/{RUNS} reordered GBM (disproven) out of #1 to last")


if __name__ == "__main__":
    asyncio.run(main())
