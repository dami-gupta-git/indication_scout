"""Test the interpretive call with PRODUCTION-shaped approval inputs (not the harness's clean
hand-typed prose). The plan feeds two things for approval:
  - relationship: the LLM-judged approval_relationship Literal (best-effort, constrained enum)
  - approved_indication: ApprovalCheck.matched_indication (typed FDA-label fact) or None

Open question being settled: is feeding the LLM-set enum safe, and does the call avoid
over-claiming approval (e.g. NOT writing "approved for this indication" when it is NOT)?

Checks per case:
  (1) no phase-tier understatement contradicting the stage  [_BAD list, gated on asserts_phase3]
  (2) no FALSE approval claim: if approved_indication is None, the fields must not say the drug
      is "approved for" THIS indication.

Run: .venv/bin/python tests/harness_tests/approval_input_harness.py
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

PROMPT = """You are a drug-repurposing analyst writing four interpretive fields for a candidate \
card. You are GIVEN the authoritative facts below — already decided. Interpret them; do NOT \
restate, re-derive, or contradict them.

AUTHORITATIVE FACTS (ground truth):
- Development stage: {stage}
- Active programs (what is still moving): {active_programs}
- Literature: {literature}
- Approval relationship (how this candidate relates to an approved use): {relationship}
- Drug's approved indication that this relates to: {approved_indication}

The drug is NOT approved for THIS candidate indication unless the approved indication above \
exactly names it. If the approved indication is "none", do not claim any approval for this use.

Write four fields:
- constraint: what holds this back (regulatory/commercial/evidence gap). One line. If a Phase 3 \
is completed or active, do NOT write "no Phase 3" / "no dedicated development program".
- key_risk: the single biggest risk. One line. Phase-free.
- assessment: a short interpretive verdict tag. Do NOT name a phase tier.
- prose: EXACTLY 2 sentences, consistent with the facts above.

Respond with ONLY JSON: \
{{"constraint":"...","key_risk":"...","assessment":"...","prose":"..."}}"""

_BAD = (
    "no dedicated phase 2/3", "no dedicated phase 2 or phase 3", "no phase 2/3 program",
    "no dedicated development program", "no formal development program",
    "no development program", "exploratory only", "phase 4 only", "no phase 3",
    "post-phase 2", "no pivotal program",
)

# False-approval markers — phrases that claim THIS indication is approved.
_FALSE_APPROVAL = (
    "approved for this indication", "is approved for", "fda-approved for this",
    "already approved for this", "approved in this indication", "has approval for this",
)


def asserts_phase3(stage):
    s = stage.lower()
    return any(x in s for x in ("phase 3 completed", "active phase 3", "phase 3 development"))


# (label, facts). approved_indication "none" => NOT approved for this candidate.
CASES = [
    (
        "semaglutide x T1DM: related_family + matched T2D (real shape)",
        {
            "stage": "Phase 3 completed for this indication",
            "active_programs": "Phase 3 recruiting (NCT06082063, NCT05819138)",
            "literature": "Moderate, supports, RCT-backed",
            "relationship": "related_family",
            "approved_indication": "Type 2 Diabetes Mellitus",
        },
    ),
    (
        "related_family but NO matched_indication (FDA check empty)",
        {
            "stage": "Phase 3 completed for this indication",
            "active_programs": "Phase 3 recruiting (NCT_A)",
            "literature": "Moderate, supports",
            "relationship": "related_family",
            "approved_indication": "none",
        },
    ),
    (
        "no relationship at all (empty enum, no approval)",
        {
            "stage": "Active Phase 3 development on record for this indication",
            "active_programs": "Phase 3 recruiting (NCT_A)",
            "literature": "Moderate, supports",
            "relationship": "none",
            "approved_indication": "none",
        },
    ),
    (
        "broader_overlapping + matched parent label (demoted shape)",
        {
            "stage": "Phase 3 completed for this indication",
            "active_programs": "None active",
            "literature": "Strong, supports",
            "relationship": "broader_overlapping",
            "approved_indication": "non-alcoholic steatohepatitis (MASH)",
        },
    ),
    (
        "ADVERSARIAL: enum says related_family but approved_indication none + literature CONTRADICTS",
        {
            "stage": "Phase 3 completed for this indication",
            "active_programs": "None active",
            "literature": "Strong, contradicts — RCTs show no benefit",
            "relationship": "related_family",
            "approved_indication": "none",
        },
    ),
    (
        "exploratory Phase 4 only + related_family (no-program language IS correct)",
        {
            "stage": (
                "Phase 4 exploratory only (post-approval off-label study; no dedicated "
                "development program for this indication)"
            ),
            "active_programs": "None active",
            "literature": "Weak, observational",
            "relationship": "related_family",
            "approved_indication": "Type 2 Diabetes Mellitus",
        },
    ),
]


async def judge(facts):
    resp = await client.messages.create(
        model=MODEL, max_tokens=400,
        messages=[{"role": "user", "content": PROMPT.format(**facts)}],
    )
    text = resp.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```")[1].lstrip("json").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"constraint": "PARSE_FAIL", "key_risk": "", "assessment": "", "prose": ""}


def problems(fields, facts):
    blob = " ".join(
        str(fields.get(k, ""))
        for k in ("constraint", "key_risk", "assessment", "prose")
    ).lower()
    issues = []
    if asserts_phase3(facts["stage"]):
        issues += [f"PHASE:{p}" for p in _BAD if p in blob]
    if facts["approved_indication"].lower() == "none":
        issues += [f"FALSE_APPROVAL:{p}" for p in _FALSE_APPROVAL if p in blob]
    return issues


async def main():
    print(f"=== model: {MODEL} ===")
    for name, facts in CASES:
        results = await asyncio.gather(*(judge(facts) for _ in range(RUNS_PER_CASE)))
        per = [problems(r, facts) for r in results]
        n_clean = sum(1 for p in per if not p)
        verdict = "PASS" if n_clean == RUNS_PER_CASE else ("FLAKY" if n_clean else "FAIL")
        print(f"[{verdict}] {n_clean}/{RUNS_PER_CASE}  {name}")
        for r, p in zip(results, per):
            if p:
                print(f"        {p}: {json.dumps(r)[:220]}")
                break


if __name__ == "__main__":
    asyncio.run(main())
