"""Isolated harness for the A2 critic-rewrite step ONLY — no supervisor, no finalize.

Builds the exact prompt critique_ranking sends (per-disease FACT block + full blurbs as
JSON), calls the real critic LLM with _RANKING_CRITIC_SYSTEM, and checks that it:
  - REWRITES a false "no Phase 3 trials" claim when FACT says a completed Phase 3 is on record,
  - LEAVES a true "no regulatory/NDA program" claim untouched (the trial-vs-program distinction),
  - LEAVES fields alone when FACT says no completed Phase 3.

Each case is hand-labeled: which field should change, which must NOT. Scored pass/fail.

Run: .venv/bin/python tests/harness_tests/critic_rewrite_harness.py
"""

import asyncio
import json
import logging

from indication_scout.agents.supervisor.supervisor_tools import _RANKING_CRITIC_SYSTEM
from indication_scout.services.llm import query_llm, strip_markdown_fences

logger = logging.getLogger("critic_rewrite_harness")


def _fact_line(i: int, blurb: dict, has_p3: bool, highest: str | None) -> str:
    disease = (blurb.get("disease") or "").strip() or "(unnamed)"
    if has_p3:
        fact = f"relevant COMPLETED Phase 3 IS on record (highest={highest or 'Phase 3'})"
    else:
        fact = "no relevant completed Phase 3 on record"
    return (
        f"{i}. {disease} | FACT: {fact} | "
        f"verdict: {blurb.get('verdict') or '—'} | "
        f"literature: {blurb.get('literature') or '—'} | "
        f"blocker: {blurb.get('blocker') or '—'}"
    )


async def _run_critic(blurb: dict, has_p3: bool, highest: str | None) -> dict:
    """Send the single-blurb prompt exactly as critique_ranking builds it; return repaired blurb."""
    prompt = (
        "Current ranking (top to bottom), each with its authoritative FACT:\n"
        + _fact_line(1, blurb, has_p3, highest)
        + "\n\nFull blurbs to audit and repair (return all of them):\n"
        + json.dumps([blurb])
    )
    raw = await query_llm(prompt, system=_RANKING_CRITIC_SYSTEM)
    data = json.loads(strip_markdown_fences(raw.strip()))
    return data["blurbs"][0]


# Each case: (label, has_completed_phase3, highest, blurb, must_change_fields, must_keep_fields)
# must_change: the false token expected GONE from that field.
# must_keep:   substring that MUST remain (true claim or unrelated text).
CASES = [
    (
        "PCOS: false 'no Phase 3' in key_risk, true 'no regulatory program' must stay",
        True,
        "Phase 3",
        {
            "disease": "polycystic ovary syndrome",
            "stage": "Phase 4 exploratory only (no dedicated development program)",
            "key_risk": "No Phase 3 pivotal program on record; all activity is Phase 4 off-label",
            "constraint": "No dedicated regulatory development program; off-label use dominates",
            "verdict": "Live, off-label",
            "literature": "Strong, RCTs and meta-analyses",
        },
        {  # field -> token that must be GONE (case-insensitive)
            "key_risk": "all activity is phase 4",
            "stage": "exploratory only",
        },
        {  # field -> substring that must REMAIN
            "constraint": "regulatory",
        },
    ),
    (
        "True Phase-4-only pair: nothing should change",
        False,
        "Phase 4",
        {
            "disease": "disease y",
            "stage": "Phase 4 exploratory only (no dedicated development program)",
            "key_risk": "No Phase 3 on record; all activity is Phase 4",
            "verdict": "Untested",
            "literature": "Weak",
        },
        {},  # must_change: none
        {  # must_keep: the exploratory claim is TRUE here
            "stage": "exploratory only",
            "key_risk": "phase 4",
        },
    ),
]


def _score(repaired: dict, must_change: dict, must_keep: dict) -> list[str]:
    fails = []
    for field, gone in must_change.items():
        if gone.lower() in (repaired.get(field) or "").lower():
            fails.append(f"  FAIL must_change[{field}]: still contains {gone!r} -> {repaired.get(field)!r}")
    for field, keep in must_keep.items():
        if keep.lower() not in (repaired.get(field) or "").lower():
            fails.append(f"  FAIL must_keep[{field}]: lost {keep!r} -> {repaired.get(field)!r}")
    return fails


async def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    total_fails = 0
    for label, has_p3, highest, blurb, must_change, must_keep in CASES:
        print("=" * 88)
        print(label)
        repaired = await _run_critic(blurb, has_p3, highest)
        for k in ("stage", "key_risk", "constraint"):
            if k in blurb:
                print(f"  {k}: {repaired.get(k)!r}")
        fails = _score(repaired, must_change, must_keep)
        if fails:
            total_fails += len(fails)
            print("\n".join(fails))
        else:
            print("  PASS")
    print("=" * 88)
    print(f"TOTAL FAILURES: {total_fails}")


if __name__ == "__main__":
    asyncio.run(main())
