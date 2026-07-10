"""Standalone harness: when the CT trial-section summary is FED the already-resolved
development STAGE (the authoritative dev_stage phrase), does the prose stop contradicting it?

The bug this settles (snapshot semaglutide_2026-06-14_18-58-57.md, T1DM): the in-loop summary
judged the tier on its own and wrote "no completed Phase 3 specifically for T1DM" while the
authoritative dev_stage said "Phase 3 completed" (a completed Phase 2/Phase 3 counts). The fix
is to author the prose AFTER the stage is resolved and FEED it that stage so it cannot
contradict it.

This harness proves the fed-the-stage prose:
  1. NEVER writes a tier-contradiction (e.g. "no completed Phase 3", "no dedicated Phase 3
     program", "Phase 4 / exploratory only") when the fed stage says a completed Phase 3 exists.
  2. Judges CLOSURE correctly (live vs closed): closed ONLY on a relevant Phase 3 terminated for
     safety/benefit:risk; an old/off-patent drug with no approval is NOT closure.

It mirrors the prompt that services/clinical_trials_summary.py will carry. Run N times per case
to catch drift, not just ignorance.

Run: .venv/bin/python tests/harness_tests/ct_summary_harness.py [model] [subset]
"""

import asyncio
import json
import sys
from collections import Counter

from anthropic import AsyncAnthropic

from indication_scout.config import get_settings

client = AsyncAnthropic(api_key=get_settings().anthropic_api_key)
MODEL = sys.argv[1] if len(sys.argv) > 1 else "claude-sonnet-4-6"
_SUBSET = len(sys.argv) > 2 and sys.argv[2] == "subset"
RUNS_PER_CASE = 5

# Mirrors the planned _CT_SUMMARY_PROMPT. The stage is GIVEN as ground truth; the model
# describes the trials and judges CLOSURE only — it must NOT re-judge the tier. No approval
# input (the supervisor owns approval framing); first_approval is fed only for the
# old-drug-no-approval-is-not-closure closure rule.
PROMPT = """You are a clinical development analyst writing the Clinical Trials section for one \
drug x indication pair.

The development STAGE has ALREADY been determined authoritatively and is GIVEN to you below as
ground truth. Your job is to DESCRIBE the trials and judge CLOSURE (is the pair still live or
closed). You must NOT re-judge or contradict the stage.

GIVEN (ground truth — do not contradict):
- stage: {stage}
- active_programs: {active_programs}
- first_approval (year the drug was first approved anywhere; "unknown" if not known): \
{first_approval}

Hard rules on the prose:
- The stage is authoritative. NEVER write a phrase that re-judges or contradicts it — e.g. "no
  completed Phase 3", "no dedicated Phase 3 program", "no pivotal trial on record", "the program
  is exploratory / Phase 4 only" — when the given stage says otherwise.
- A completed "Phase 2/Phase 3" or "Phase 3/Phase 4" trial HAS a Phase 3 arm. If such a trial is
  on record, describe it as such; never write that "no completed Phase 3 exists".
- Describe the trials by name with phase + status + title; note completed/terminated/active
  facts. Leave the "what stage has this reached" conclusion to the given stage.

CLOSURE (live vs closed) — judge from the RELEVANT trials only:
- Call the pair CLOSED only on a real negative: a relevant Phase 3 terminated for SAFETY or
  benefit:risk, or literature reporting the drug failed for this indication. Operational stops
  (low enrollment, funding, sponsor decision) are NOT closure.
- An old/off-patent drug (first_approval many years before now) with NO approval for this
  indication is NOT closure — it reflects no commercial NDA filing, not efficacy failure.
- If neither holds, the pair is live (or unknown if you genuinely cannot tell).
- Say NOTHING about approval status in the prose.

Trials (relevant set):
{trials}

Respond with ONLY a JSON object:
{{"prose": "<the trial-section prose>", "closure": "live"|"closed"|"unknown", \
"closure_reason": "<one short sentence>"}}"""


def fmt(trials):
    return "\n".join(
        f"- {t['nct']}: phase={t['phase']}, status={t['status']}, title={t.get('title', '')}"
        + (f", why_stopped={t['why_stopped']}" if t.get("why_stopped") else "")
        for t in trials
    )


# Phrases that re-judge the tier downward — a contradiction when the fed stage says a completed
# Phase 3 exists. Matched case-insensitively against the prose.
_CONTRADICTION_PHRASES = [
    "no completed phase 3",
    "no completed phase iii",
    "no dedicated phase 3",
    "no pivotal trial",
    "no pivotal phase 3",
    "no phase 3 program",
    "exploratory only",
    "phase 4 only",
    "no phase 3 on record",
    "lacks a completed phase 3",
    "without a completed phase 3",
]


# (name, stage, active_programs, first_approval, trials, expected_closure,
#  forbid_contradiction). forbid_contradiction = the stage asserts a completed Phase 3, so any
#  _CONTRADICTION_PHRASES in the prose is a FAIL.
CASES = [
    (
        "T1DM-shape (the bug): stage=Phase 3 completed, completed P2/3 + active P3s",
        "Phase 3 completed for this indication",
        "2 Phase 3 recruiting (NCT_D, NCT_E)",
        1923,  # an old generic (insulin-era) — no-approval must not read as closure
        [
            {"nct": "NCT_A", "phase": "Phase 2", "status": "COMPLETED", "title": "P2 study"},
            {"nct": "NCT_B", "phase": "Phase 4", "status": "COMPLETED", "title": "P4 study"},
            {"nct": "NCT_C", "phase": "Phase 2/Phase 3", "status": "COMPLETED",
             "title": "Pivotal P2/3"},
            {"nct": "NCT_D", "phase": "Phase 3", "status": "Recruiting", "title": "Active P3"},
            {"nct": "NCT_E", "phase": "Phase 3", "status": "Recruiting", "title": "Active P3"},
        ],
        "live",
        True,
    ),
    (
        "completed Phase 2/Phase 3 only (cocaine-shape), stage=Phase 3 completed",
        "Phase 3 completed for this indication",
        "None active",
        1880,  # cocaine-era; old, no approval ≠ closed
        [
            {"nct": "NCT02111798", "phase": "Phase 2/Phase 3", "status": "COMPLETED",
             "title": "Pivotal"},
            {"nct": "NCT00227812", "phase": "Phase 2", "status": "COMPLETED", "title": "P2"},
        ],
        "live",
        True,
    ),
    (
        "Phase 3/Phase 4 completed (has a P3 arm), stage=Phase 3 completed",
        "Phase 3 completed for this indication",
        "None active",
        2005,
        [{"nct": "NCT_A", "phase": "Phase 3/Phase 4", "status": "COMPLETED",
          "title": "P3/4 pivotal"}],
        "live",
        True,
    ),
    (
        "Phase 3 terminated for SAFETY, stage=Phase 3 terminated for cause -> CLOSED",
        "Phase 3 terminated for cause (safety/efficacy stop)",
        "None active",
        2010,
        [
            {"nct": "NCT_A", "phase": "Phase 3", "status": "Terminated",
             "title": "Pivotal P3", "why_stopped": "halted for serious adverse events"},
            {"nct": "NCT_B", "phase": "Phase 2", "status": "COMPLETED", "title": "P2"},
        ],
        "closed",
        False,
    ),
    (
        "Phase 3 terminated for LOW ENROLLMENT (operational), stage=early -> NOT closed",
        "Early-phase only, no completed pivotal readout",
        "None active",
        2012,
        [
            {"nct": "NCT_A", "phase": "Phase 3", "status": "Terminated",
             "title": "P3", "why_stopped": "terminated due to low enrollment"},
        ],
        "live",  # operational stop is not closure; live (or unknown) acceptable
        False,
    ),
    (
        "old generic, completed Phase 2 only, no approval -> NOT closed on no-approval",
        "Phase 2 completed for this indication, no Phase 3",
        "None active",
        1957,  # metformin-era
        [{"nct": "NCT_A", "phase": "Phase 2", "status": "COMPLETED", "title": "P2 readout"}],
        "live",
        False,
    ),
    (
        "active Phase 3 only (recruiting), stage=active -> live",
        "Active Phase 3 development on record for this indication",
        "1 Phase 3 recruiting (NCT_A)",
        2018,
        [{"nct": "NCT_A", "phase": "Phase 3", "status": "Recruiting", "title": "Ongoing P3"}],
        "live",
        False,
    ),
]


async def judge(case):
    _, stage, active, first_approval, trials, _, _ = case
    resp = await client.messages.create(
        model=MODEL,
        max_tokens=600,
        messages=[{"role": "user", "content": PROMPT.format(
            stage=stage,
            active_programs=active,
            first_approval=first_approval,
            trials=fmt(trials),
        )}],
    )
    text = resp.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```")[1].lstrip("json").strip()
    try:
        data = json.loads(text)
        return data.get("prose", ""), data.get("closure", "PARSE_FAIL")
    except json.JSONDecodeError:
        return "", "PARSE_FAIL"


# Closure expectations that accept more than one label.
_CLOSURE_ACCEPT = {
    "live": {"live", "unknown"},  # never accept "closed" where we expect live
    "closed": {"closed"},
    "unknown": {"live", "unknown"},
}


def closure_ok(got, expected):
    return got in _CLOSURE_ACCEPT.get(expected, {expected})


def contradicts(prose):
    low = prose.lower()
    return [p for p in _CONTRADICTION_PHRASES if p in low]


async def main():
    print(f"=== model: {MODEL} ===")
    cases = [CASES[0], CASES[1], CASES[3], CASES[4]] if _SUBSET else CASES
    all_pass = True
    for case in cases:
        name, _, _, _, _, expected_closure, forbid = case
        results = await asyncio.gather(*(judge(case) for _ in range(RUNS_PER_CASE)))
        closures = Counter(c for _, c in results)
        n_closure_ok = sum(closure_ok(c, expected_closure) for _, c in results)
        # Contradiction check only where the stage asserts a completed Phase 3.
        bad_runs = [contradicts(p) for p, _ in results] if forbid else [[] for _ in results]
        n_clean = sum(not b for b in bad_runs)

        closure_pass = n_closure_ok == RUNS_PER_CASE
        prose_pass = n_clean == RUNS_PER_CASE
        verdict = "PASS" if (closure_pass and prose_pass) else "FAIL"
        if verdict == "FAIL":
            all_pass = False
        print(f"[{verdict}] {name}")
        print(f"        closure {n_closure_ok}/{RUNS_PER_CASE} ok "
              f"(expected={expected_closure}, got={dict(closures)})")
        if forbid:
            print(f"        prose   {n_clean}/{RUNS_PER_CASE} contradiction-free")
            hits = [h for b in bad_runs for h in b]
            if hits:
                print(f"        contradictions hit: {Counter(hits)}")
    print("=== ALL PASS ===" if all_pass else "=== SOME FAILED ===")


if __name__ == "__main__":
    asyncio.run(main())
