"""Integration test for services/judge_interpretive — the LIVE interpretive call.

Hits real Anthropic. Mirrors the crux cases proven in scratch/{interpretive_fields,
staged_blurb,approval_input}_harness.py: handed the resolved facts, the interpretive fields
must NOT contradict the authoritative stage, and must not over-claim approval.

Uses test_cache_dir so the real cache is untouched.
"""

import logging
import re

import pytest

from indication_scout.services.judge_interpretive import judge_interpretive

logger = logging.getLogger(__name__)

# Phrases that contradict a completed/active Phase 3 stage.
_BAD = (
    "no dedicated phase 2/3",
    "no dedicated phase 2 or phase 3",
    "no phase 2/3 program",
    "no dedicated development program",
    "no formal development program",
    "no development program",
    "exploratory only",
    "phase 4 only",
    "no phase 3",
    "post-phase 2",
    "no pivotal program",
)
_FALSE_APPROVAL = (
    "approved for this indication",
    "is approved for this",
    "fda-approved for this",
)


def _asserts_phase3(stage: str) -> bool:
    s = stage.lower()
    return any(
        x in s for x in ("phase 3 completed", "active phase 3", "phase 3 development")
    )


# (label, facts, expect_phase3_consistency_checked)
_CASES = [
    (
        "T1DM: completed P3 + active P3 + related_family/T2D (the recurring contradiction)",
        dict(
            stage="Phase 3 completed for this indication",
            active_programs="Phase 3 recruiting (NCT06082063, NCT05819138)",
            literature="moderate, supports, RCT-backed / controlled",
            relationship="related_family",
            approved_indication="Type 2 Diabetes Mellitus",
        ),
    ),
    (
        "contradicts: drug failed (completed P3, literature contradicts)",
        dict(
            stage="Phase 3 completed for this indication",
            active_programs="None active",
            literature="strong, contradicts, RCT-backed / controlled",
            relationship="none",
            approved_indication=None,
        ),
    ),
    (
        "genuine Phase-4-only ('no program' language IS correct here)",
        dict(
            stage=(
                "Phase 4 exploratory only (post-approval off-label study; no dedicated "
                "development program for this indication)"
            ),
            active_programs="None active",
            literature="weak, observational",
            relationship="related_family",
            approved_indication="Type 2 Diabetes Mellitus",
        ),
    ),
    (
        "related_family but NO matched indication (must not over-claim approval)",
        dict(
            stage="Active Phase 3 development on record for this indication",
            active_programs="Phase 3 recruiting (NCT_A)",
            literature="moderate, supports",
            relationship="related_family",
            approved_indication=None,
        ),
    ),
]


@pytest.mark.parametrize("label,facts", _CASES, ids=[c[0] for c in _CASES])
async def test_judge_interpretive_no_contradiction_live(label, facts, test_cache_dir):
    j = await judge_interpretive(
        **facts, cache_dir=test_cache_dir, drug="td", indication=label
    )
    assert j is not None, f"{label}: parse failed"
    blob = " ".join([j.blocker, j.key_risk, j.verdict, j.prose]).lower()

    # (1) No phase understatement contradicting a completed/active Phase 3 stage.
    if _asserts_phase3(facts["stage"]):
        leaked = [p for p in _BAD if p in blob]
        assert not leaked, f"{label}: contradicts stage with {leaked}: {blob}"

    # (2) No false approval claim when there is no matched approved indication. Flag an
    # AFFIRMATIVE approval phrase only — a NEGATED form ("not yet approved for this indication")
    # is correct and must not trip the check.
    if facts["approved_indication"] is None:
        false_appr = [
            p
            for p in _FALSE_APPROVAL
            if p in blob
            and not re.search(r"(?:not|no|never)\b[^.]{0,15}" + re.escape(p), blob)
        ]
        assert not false_appr, f"{label}: false approval claim {false_appr}: {blob}"

    # (3) All four fields populated.
    assert j.blocker and j.key_risk and j.verdict and j.prose
