"""Standalone harness for _render_active_programs (services/dev_stage): does the deterministic
"active programs" summary line describe each trial's status HONESTLY?

The label had a recurring failure mode: it collapsed statuses that are NOT "recruiting/ongoing"
into either "active" or "None active", overstating what we know.
  - NOT_YET_RECRUITING is a PLANNED program, not an ongoing one — must not read as "active".
  - UNKNOWN status is NOT confirmed inactive — "None active" over it asserts an inactivity we
    cannot confirm; it must be surfaced as "status unknown".
  - COMPLETED / TERMINATED / WITHDRAWN are known-inactive and stay silent (never listed).

Each case builds a relevant trial set and asserts the exact rendered line. Pure/deterministic —
no LLM, no network.

Run: PYTHONPATH=src python tests/harness_tests/active_programs_render_harness.py
"""

import sys

from indication_scout.models.model_clinical_trials import Trial
from indication_scout.services.dev_stage import _render_active_programs


def _t(nct: str, phase: str, status: str) -> Trial:
    return Trial(nct_id=nct, phase=phase, overall_status=status, title="x")


# (name, trials, expected_line)
CASES: list[tuple[str, list[Trial], str]] = [
    (
        "semaglutide/Parkinson: single unknown-status Phase 2 (the bug)",
        [_t("NCT03659682", "Phase 2", "Unknown status")],
        "None active; 1 on record with unknown status (NCT03659682)",
    ),
    (
        "unknown-status enum form (UNKNOWN)",
        [_t("NCT03659682", "Phase 2", "UNKNOWN")],
        "None active; 1 on record with unknown status (NCT03659682)",
    ),
    (
        "two unknown-status trials",
        [
            _t("NCT00000001", "Phase 3", "Unknown status"),
            _t("NCT00000002", "Phase 2", "Unknown"),
        ],
        "None active; 2 on record with unknown status (NCT00000001, NCT00000002)",
    ),
    (
        "unknown status alongside an ACTIVE pivotal trial -> pivotal wins, unknown not appended",
        [
            _t("NCT00000001", "Phase 3", "Recruiting"),
            _t("NCT00000002", "Phase 2", "Unknown status"),
        ],
        "1 Phase 3 active (NCT00000001)",
    ),
    (
        "only completed/terminated/withdrawn (all known-inactive) -> None active, silent",
        [
            _t("NCT00000001", "Phase 3", "Completed"),
            _t("NCT00000002", "Phase 2", "Terminated"),
            _t("NCT00000003", "Phase 1", "Withdrawn"),
        ],
        "None active",
    ),
    (
        "empty set -> None active",
        [],
        "None active",
    ),
    (
        "metformin/PCOS: single not-yet-recruiting Phase 3",
        [_t("NCT07120815", "Phase 3", "Not yet recruiting")],
        "1 Phase 3 not yet recruiting (NCT07120815)",
    ),
    (
        "recruiting + not-yet-recruiting Phase 3 split",
        [
            _t("NCT00000001", "Phase 3", "Recruiting"),
            _t("NCT07120815", "Phase 3", "Not yet recruiting"),
        ],
        "1 Phase 3 active (NCT00000001); 1 Phase 3 not yet recruiting (NCT07120815)",
    ),
    (
        "plain recruiting Phase 3 -> active",
        [_t("NCT00000001", "Phase 3", "Recruiting")],
        "1 Phase 3 active (NCT00000001)",
    ),
    (
        "active Phase 2/Phase 3 (pivotal band, not pure P3)",
        [_t("NCT03899402", "Phase 2/Phase 3", "Active, not recruiting")],
        "1 Phase 2/Phase 3 active (NCT03899402)",
    ),
    (
        "no pivotal active, only earlier-phase active -> non-pivotal line",
        [_t("NCT06374875", "Phase 4", "Recruiting")],
        "No pivotal program active; 1 non-pivotal active (NCT06374875)",
    ),
    (
        "non-pivotal active AND an unknown-status trial: active wins (unknown not surfaced here)",
        [
            _t("NCT06374875", "Phase 4", "Recruiting"),
            _t("NCT06005012", "Phase 2", "Unknown status"),
        ],
        "No pivotal program active; 1 non-pivotal active (NCT06374875)",
    ),
]


def main() -> int:
    failures = 0
    for name, trials, expected in CASES:
        got = _render_active_programs(trials)
        ok = got == expected
        tag = "PASS" if ok else "FAIL"
        print(f"[{tag}] {name}")
        if not ok:
            failures += 1
            print(f"        expected: {expected!r}")
            print(f"        got:      {got!r}")
    total = len(CASES)
    print(f"\n{total - failures}/{total} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
