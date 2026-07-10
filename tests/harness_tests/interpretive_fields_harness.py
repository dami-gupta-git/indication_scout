"""Harness: does an ISOLATED call that is HANDED the authoritative facts (stage,
active_programs, literature) and asked ONLY for the interpretive fields (constraint, key_risk,
assessment) stay consistent with those facts — i.e. NOT contradict the stage?

The bug (reports 17:15 .. 17:30): in the 8-field blurb pass the LLM writes "no dedicated
Phase 2/3 program" / "exploratory only" in constraint/key_risk/assessment even though Stage
says "Phase 3 completed" and active_programs lists 6 recruiting Phase 3s. Prompt rules in the
big pass don't hold. This tests whether a focused call, given the facts as INPUT, behaves.

Pass = the three interpretive fields contain NO phase-tier understatement that contradicts the
given stage (no "no dedicated Phase 2/3 program", "exploratory only", "Phase 4 only", "post-
Phase 2", "no Phase 3", etc. when stage says a Phase 3 is completed/active).

Run: .venv/bin/python tests/harness_tests/interpretive_fields_harness.py
"""

import asyncio
import json
import sys
from collections import Counter

from anthropic import AsyncAnthropic

from indication_scout.config import get_settings

client = AsyncAnthropic(api_key=get_settings().anthropic_api_key)
MODEL = sys.argv[1] if len(sys.argv) > 1 else "claude-sonnet-4-6"
RUNS_PER_CASE = 5

PROMPT = """You are a drug-repurposing analyst writing three short interpretive fields for a \
candidate card. You are GIVEN the authoritative facts below — they are already decided. Your \
job is ONLY to interpret them. Do NOT restate, re-derive, or contradict them.

AUTHORITATIVE FACTS (already decided — treat as ground truth):
- Development stage: {stage}
- Active programs (what is still moving): {active_programs}
- Literature: {literature}
- Approval relationship: {approval}

Write four fields:
- constraint: what is holding this repurposing opportunity back (regulatory, commercial, \
evidence gap). One short line. Do NOT contradict the stage — if a Phase 3 is completed or \
active, do NOT write "no Phase 3" or "no dedicated development program".
- key_risk: the single biggest risk to the hypothesis. One short line. Phase-free; interpretive.
- assessment: a short interpretive verdict tag (e.g. "Live but bottlenecked", "Maturing, \
awaiting readout", "Stalled, regulatory gap", "Untested at scale", "Closed signal"). Do NOT \
name a phase tier.
- prose: EXACTLY 2 sentences interpreting the state of the hypothesis. Must be consistent with \
the stage and active programs above. If the literature direction is "contradicts", surface \
that the drug failed/was disproven. Do NOT name a phase tier that disagrees with the stage.

Respond with ONLY JSON: \
{{"constraint": "...", "key_risk": "...", "assessment": "...", "prose": "..."}}"""

# Phrases that contradict a "Phase 3 completed/active" stage.
_BAD = (
    "no dedicated phase 2/3",
    "no dedicated phase 2 or phase 3",
    "no phase 2/3 program",
    "no dedicated development program",
    "no formal development program",
    "no development program",
    "exploratory only",
    "phase 4 exploratory",
    "phase 4 only",
    "no phase 3",
    "no completed phase 3",  # tricky: true if completed but stage may be "active" only
    "post-phase 2",
    "stalled post-phase 2",
    "no pivotal program",
    "no dedicated pivotal",
)

# stage is the AUTHORITATIVE input. _exploratory flag = the "no program" language is CORRECT
# for this case (genuine Phase-4-only), so the contradiction check is skipped.
CASES = [
    (
        "T1DM: completed P3 + many active P3 (the recurring contradiction)",
        {
            "stage": "Phase 3 completed for this indication",
            "active_programs": (
                "Phase 2/Phase 3 active (NCT03899402); Phase 3 recruiting "
                "(NCT06082063, NCT05819138, NCT06894784); Phase 3 not yet recruiting "
                "(NCT06909006)"
            ),
            "literature": "Moderate, RCT-backed controlled studies",
            "approval": "related_family (approved for type 2 diabetes)",
        },
    ),
    (
        "active P3 only, none completed",
        {
            "stage": "Active Phase 3 development on record for this indication",
            "active_programs": "Phase 3 recruiting (NCT_A, NCT_B)",
            "literature": "Moderate, supports",
            "approval": "none",
        },
    ),
    (
        "completed P3, nothing active (genuinely stalled)",
        {
            "stage": "Phase 3 completed for this indication",
            "active_programs": "None active",
            "literature": "Strong, RCTs",
            "approval": "none",
        },
    ),
    (
        "genuinely exploratory Phase 4 only (the 'no program' verdict IS correct here)",
        {
            "stage": (
                "Phase 4 exploratory only (post-approval off-label study; no dedicated "
                "development program for this indication)"
            ),
            "active_programs": "None active",
            "literature": "Weak, observational",
            "approval": "related_family",
        },
    ),
    (
        "completed P3 but literature CONTRADICTS (drug failed in trials)",
        {
            "stage": "Phase 3 completed for this indication",
            "active_programs": "None active",
            "literature": "Strong, contradicts — multiple RCTs show no benefit",
            "approval": "none",
        },
    ),
    (
        "Phase 3 terminated for SAFETY (closed signal)",
        {
            "stage": "Phase 3 terminated for cause (safety/efficacy stop)",
            "active_programs": "None active",
            "literature": "Moderate, mixed",
            "approval": "none",
        },
    ),
    (
        "completed P2 only, no Phase 3 (here 'no Phase 3' IS accurate)",
        {
            "stage": "Phase 2 completed for this indication, no Phase 3",
            "active_programs": "Phase 2 recruiting (NCT_X)",
            "literature": "Moderate, supports",
            "approval": "none",
        },
    ),
    (
        "untested, rationale only",
        {
            "stage": "No trials on record for this indication",
            "active_programs": "None active",
            "literature": "Weak, preclinical/observational only",
            "approval": "none",
        },
    ),
    (
        "completed P3 + active P3 + adverse safety signal in literature",
        {
            "stage": "Phase 3 completed for this indication",
            "active_programs": "Phase 3 recruiting (NCT_Z)",
            "literature": "Strong, mixed — efficacy signal but adverse safety reports",
            "approval": "none",
        },
    ),
    (
        "active P3 unknown status (on record, status unknown)",
        {
            "stage": "Phase 3 on record, status unknown",
            "active_programs": "None active",
            "literature": "Moderate, supports",
            "approval": "none",
        },
    ),
    (
        "early phase only, recruiting",
        {
            "stage": "Early-phase only, no completed pivotal readout",
            "active_programs": "Phase 1 recruiting (NCT_E)",
            "literature": "Weak, limited",
            "approval": "none",
        },
    ),
]


async def judge(facts):
    resp = await client.messages.create(
        model=MODEL,
        max_tokens=400,
        messages=[{"role": "user", "content": PROMPT.format(**facts)}],
    )
    text = resp.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```")[1].lstrip("json").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"constraint": "PARSE_FAIL", "key_risk": "", "assessment": ""}


def contradicts(fields, stage):
    """Return the list of phase understatements in the interpretive fields (incl. prose) that
    contradict the stage. Only enforced when the stage asserts a COMPLETED or ACTIVE Phase 3 —
    for genuinely sub-Phase-3 stages (Phase 4 only, completed P2, terminated, unknown-status,
    early, untested) a 'no Phase 3 program' statement is ACCURATE, not a contradiction."""
    s = stage.lower()
    asserts_phase3 = (
        ("phase 3 completed" in s)
        or ("active phase 3" in s)
        or ("phase 3 development" in s)
    )
    if not asserts_phase3:
        return []  # 'no program' / 'no Phase 3' language is accurate for these stages
    blob = " ".join(
        str(fields.get(k, ""))
        for k in ("constraint", "key_risk", "assessment", "prose")
    ).lower()
    return [p for p in _BAD if p in blob]


async def main():
    print(f"=== model: {MODEL} ===")
    for name, facts in CASES:
        results = await asyncio.gather(*(judge(facts) for _ in range(RUNS_PER_CASE)))
        bad_runs = [contradicts(r, facts["stage"]) for r in results]
        n_clean = sum(1 for b in bad_runs if not b)
        verdict = "PASS" if n_clean == RUNS_PER_CASE else (
            "FLAKY" if n_clean else "FAIL"
        )
        print(f"[{verdict}] {n_clean}/{RUNS_PER_CASE}  {name}")
        # Show one contradicting example if any.
        for r, bad in zip(results, bad_runs):
            if bad:
                print(f"        contradiction {bad}: {json.dumps(r)[:200]}")
                break


if __name__ == "__main__":
    asyncio.run(main())
