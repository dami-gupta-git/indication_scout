"""Standalone harness: does the LLM judge the development STAGE correctly from raw trial
details alone (phase + status), applying clinical-trial conventions it already knows — WITHOUT
being handed a precomputed dev_stage?

The question this settles: can we drop the deterministic dev_stage and just say "apply the
phase conventions you know"? The hard cases are the ones that recurred this session:
  - Phase 4 ranks ABOVE Phase 3 numerically but is POST-APPROVAL, not progression. A completed
    Phase 4 alongside a Phase 2 must NOT read as "past Phase 3".
  - A completed "Phase 2/Phase 3" DOES count as a completed Phase 3 (it has a Phase 3 arm).
  - An UNKNOWN-status Phase 3 is "on record, status unknown", not a completed Phase 3.
  - A WITHDRAWN Phase 3 never ran — not "a Phase 3 on record".

Each case lists trials; the model must return one tier label. We score against EXPECTED.
Run N times per case to measure consistency (the real failure mode is drift, not ignorance).

Run: .venv/bin/python tests/harness_tests/dev_stage_judgment_harness.py
"""

import asyncio
import json
import sys
from collections import Counter

from anthropic import AsyncAnthropic

from indication_scout.config import get_settings

client = AsyncAnthropic(api_key=get_settings().anthropic_api_key)
# Model from argv[1] (e.g. "claude-haiku-4-5-20251001"); default Sonnet. argv[2] = "subset"
# runs only the 5 crux cases.
MODEL = sys.argv[1] if len(sys.argv) > 1 else "claude-sonnet-4-6"
_SUBSET = len(sys.argv) > 2 and sys.argv[2] == "subset"
RUNS_PER_CASE = 5

TIERS = [
    "completed_phase3",  # a completed Phase 3 or Phase 2/Phase 3 (or Phase 3/Phase 4) exists
    "active_phase3",  # an active/recruiting Phase 3-band trial, none completed
    "phase3_unknown_status",  # a Phase 3 on record but neither completed nor active (unknown)
    "completed_phase2",  # highest completed is Phase 2 / Phase 1-2, no Phase 3
    "exploratory_phase4_only",  # ONLY Phase 4 (post-approval), no Phase 2 or Phase 3
    "early_phase",  # Phase 1 / early only, or trials with no completed phase
    "untested",  # no trials
]

PROMPT = """You are a clinical development analyst. Given the trials below for one drug x \
indication pair, classify the development STAGE into exactly ONE tier.

Apply the clinical-trial conventions you already know. In particular:
- Phases rank Early Phase 1 < Phase 1 < Phase 1/2 < Phase 2 < Phase 2/3 < Phase 3 < Phase 4.
- BUT Phase 4 is POST-APPROVAL / off-label activity — it is NOT progression beyond Phase 3.
  A trial that is only Phase 4 (with no Phase 2 or Phase 3 on record) is exploratory, NOT a
  completed pivotal program.
- A "Phase 2/Phase 3" trial HAS a Phase 3 arm — a completed one counts as a completed Phase 3.
- A Phase 3 whose status is UNKNOWN is "on record, status unknown" — not completed, not active.
- A WITHDRAWN / no-longer-available Phase 3 never produced a program.
- Recruiting / Active, not recruiting / Not yet recruiting / Enrolling by invitation / \
Suspended = an active/ongoing program.

Tiers (choose ONE, highest applicable wins, in this priority order):
1. completed_phase3 — a COMPLETED Phase 3 or Phase 2/Phase 3 (or Phase 3/Phase 4) trial exists.
2. active_phase3 — an ACTIVE/recruiting Phase 3-band trial exists, but none completed.
3. phase3_unknown_status — a Phase 3-band trial on record with UNKNOWN status (not completed, \
not active, not withdrawn).
4. completed_phase2 — a completed Phase 2 (or Phase 1/Phase 2) exists, no Phase 3 at all.
5. exploratory_phase4_only — ONLY Phase 4 trials, no Phase 2 or Phase 3.
6. early_phase — only Phase 1 / Early Phase 1, or trials with no completed phase.
7. untested — no trials.

Trials:
{trials}

Respond with ONLY a JSON object: {{"tier": "<one_tier>", "reason": "<one short sentence>"}}"""


def fmt(trials):
    return "\n".join(
        f"- {t['nct']}: phase={t['phase']}, status={t['status']}" for t in trials
    )


# (name, trials, expected_tier). The Phase-4 trap cases are the crux.
CASES = [
    (
        "T1DM-shape: Phase4 + Phase2/3 completed (the recurring bug)",
        [
            {"nct": "NCT_A", "phase": "Phase 2", "status": "COMPLETED"},
            {"nct": "NCT_B", "phase": "Phase 4", "status": "COMPLETED"},
            {"nct": "NCT_C", "phase": "Phase 2/Phase 3", "status": "COMPLETED"},
        ],
        "completed_phase3",
    ),
    (
        "Phase4 completed + Phase2 completed, NO phase3 (must NOT read as past-phase-3)",
        [
            {"nct": "NCT_A", "phase": "Phase 2", "status": "COMPLETED"},
            {"nct": "NCT_B", "phase": "Phase 4", "status": "COMPLETED"},
        ],
        "completed_phase2",
    ),
    (
        "ONLY a completed Phase 4 (post-approval off-label)",
        [{"nct": "NCT_A", "phase": "Phase 4", "status": "COMPLETED"}],
        "exploratory_phase4_only",
    ),
    (
        "cocaine-shape: completed Phase 2/Phase 3",
        [
            {"nct": "NCT02111798", "phase": "Phase 2/Phase 3", "status": "COMPLETED"},
            {"nct": "NCT00227812", "phase": "Phase 2", "status": "COMPLETED"},
        ],
        "completed_phase3",
    ),
    (
        "recruiting Phase 3, none completed",
        [
            {"nct": "NCT_A", "phase": "Phase 3", "status": "Recruiting"},
            {"nct": "NCT_B", "phase": "Phase 2", "status": "COMPLETED"},
        ],
        "completed_phase2_or_active_phase3",  # accept either; both defensible
    ),
    (
        "unknown-status Phase 3 only",
        [{"nct": "NCT_A", "phase": "Phase 3", "status": "UNKNOWN"}],
        "phase3_unknown_status",
    ),
    (
        "withdrawn Phase 3 (never ran)",
        [{"nct": "NCT_A", "phase": "Phase 3", "status": "Withdrawn"}],
        "early_phase_or_untested",  # must NOT be phase3_unknown_status or completed_phase3
    ),
    (
        "not-yet-recruiting Phase 3",
        [{"nct": "NCT_A", "phase": "Phase 3", "status": "Not yet recruiting"}],
        "active_phase3",
    ),
    # ---- harder / messier cases ----
    (
        "Phase 3/Phase 4 completed (has a Phase 3 arm) — counts as completed Phase 3",
        [{"nct": "NCT_A", "phase": "Phase 3/Phase 4", "status": "COMPLETED"}],
        "completed_phase3",
    ),
    (
        "large mixed T1DM-like portfolio: completed P2/3 + many active P3 + P4",
        [
            {"nct": "NCT_1", "phase": "Phase 2", "status": "COMPLETED"},
            {"nct": "NCT_2", "phase": "Phase 4", "status": "COMPLETED"},
            {"nct": "NCT_3", "phase": "Phase 2/Phase 3", "status": "COMPLETED"},
            {"nct": "NCT_4", "phase": "Phase 3", "status": "Recruiting"},
            {"nct": "NCT_5", "phase": "Phase 3", "status": "Recruiting"},
            {"nct": "NCT_6", "phase": "Phase 3", "status": "Not yet recruiting"},
            {"nct": "NCT_7", "phase": "Phase 2", "status": "Recruiting"},
            {"nct": "NCT_8", "phase": "Early Phase 1", "status": "COMPLETED"},
        ],
        "completed_phase3",  # a completed P2/3 exists; active P3s are additional
    ),
    (
        "terminated Phase 3 for SAFETY, plus a completed Phase 2",
        [
            {"nct": "NCT_A", "phase": "Phase 3", "status": "Terminated (safety concerns)"},
            {"nct": "NCT_B", "phase": "Phase 2", "status": "COMPLETED"},
        ],
        # No COMPLETED Phase 3; a terminated-for-cause P3 is a closure signal, not a completed
        # program. Either reading is defensible from raw trials alone — accept both.
        "completed_phase2_or_phase3_terminated",
    ),
    (
        "ALL Phase 3 terminated for enrollment (operational), none completed",
        [
            {"nct": "NCT_A", "phase": "Phase 3", "status": "Terminated (low enrollment)"},
            {"nct": "NCT_B", "phase": "Phase 3", "status": "Terminated (slow accrual)"},
        ],
        # No completed/active P3. Operational terminations aren't a completed program.
        "early_phase_or_phase3_terminated",
    ),
    (
        "active Phase 2/3 (recruiting) + completed Phase 1",
        [
            {"nct": "NCT_A", "phase": "Phase 2/Phase 3", "status": "Active, not recruiting"},
            {"nct": "NCT_B", "phase": "Phase 1", "status": "COMPLETED"},
        ],
        "active_phase3",
    ),
    (
        "Phase 1/Phase 2 completed only (sub-pivotal)",
        [{"nct": "NCT_A", "phase": "Phase 1/Phase 2", "status": "COMPLETED"}],
        "completed_phase2",
    ),
    (
        "suspended Phase 3 only (a pause, often resumes)",
        [{"nct": "NCT_A", "phase": "Phase 3", "status": "Suspended"}],
        "active_phase3",
    ),
    (
        "many early-phase, no completed pivotal, one recruiting Phase 2",
        [
            {"nct": "NCT_A", "phase": "Phase 1", "status": "COMPLETED"},
            {"nct": "NCT_B", "phase": "Early Phase 1", "status": "COMPLETED"},
            {"nct": "NCT_C", "phase": "Phase 2", "status": "Recruiting"},
        ],
        "early_phase",  # no completed Phase 2/3; the Phase 2 is only recruiting
    ),
    (
        "completed Phase 3 + terminated Phase 3 for safety (both on record)",
        [
            {"nct": "NCT_A", "phase": "Phase 3", "status": "COMPLETED"},
            {"nct": "NCT_B", "phase": "Phase 3", "status": "Terminated (safety)"},
        ],
        # A completed Phase 3 exists — completed wins; the termination is a separate risk note.
        "completed_phase3",
    ),
    (
        "no phase field given (phase unknown) on a completed trial",
        [{"nct": "NCT_A", "phase": "", "status": "COMPLETED"}],
        "early_phase_or_untested",  # no classifiable phase → must not claim a Phase 3
    ),
]


async def judge(trials):
    resp = await client.messages.create(
        model=MODEL,
        max_tokens=200,
        messages=[{"role": "user", "content": PROMPT.format(trials=fmt(trials))}],
    )
    text = resp.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```")[1].lstrip("json").strip()
    try:
        return json.loads(text).get("tier", "PARSE_FAIL")
    except json.JSONDecodeError:
        return "PARSE_FAIL"


_ACCEPT = {
    "completed_phase2_or_active_phase3": {"completed_phase2", "active_phase3"},
    "early_phase_or_untested": {"early_phase", "untested"},
    "completed_phase2_or_phase3_terminated": {
        "completed_phase2",
        "phase3_terminated_for_cause",
        "phase3_terminated",
    },
    "early_phase_or_phase3_terminated": {
        "early_phase",
        "untested",
        "phase3_terminated_for_cause",
        "phase3_terminated",
    },
}


def ok(got, expected):
    if expected in _ACCEPT:
        return got in _ACCEPT[expected]
    return got == expected


async def main():
    print(f"=== model: {MODEL} ===")
    # Subset = the 5 crux cases (Phase-4 traps + the recurring bug + status edges).
    cases = (
        [CASES[0], CASES[1], CASES[2], CASES[9], CASES[6]] if _SUBSET else CASES
    )
    for name, trials, expected in cases:
        results = await asyncio.gather(*(judge(trials) for _ in range(RUNS_PER_CASE)))
        counts = Counter(results)
        n_ok = sum(ok(r, expected) for r in results)
        verdict = "PASS" if n_ok == RUNS_PER_CASE else (
            "FLAKY" if n_ok else "FAIL"
        )
        print(f"[{verdict}] {n_ok}/{RUNS_PER_CASE}  {name}")
        print(f"        expected={expected}  got={dict(counts)}")


if __name__ == "__main__":
    asyncio.run(main())
