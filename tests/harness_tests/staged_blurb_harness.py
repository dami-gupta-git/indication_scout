"""End-to-end test of the STAGED blurb pipeline on real-shaped data:

  real trials + literature
      -> stage 1: judge_dev_stage  (stage tier + active_programs)        [existing service]
      -> stage 2: judge_interpretive (constraint, key_risk, assessment, prose, fed stage-1 out)
      -> assemble the full card

Proves the CHAIN (not just isolated stages) yields an internally-consistent card with no
phase contradictions. If this holds on the T1DM shape that broke every monolithic run, the
staged-synthesis redesign is validated.

Run: .venv/bin/python tests/harness_tests/staged_blurb_harness.py
"""

import asyncio
import json
import sys

from anthropic import AsyncAnthropic

from indication_scout.config import get_settings
from indication_scout.models.model_clinical_trials import Trial
from indication_scout.services.dev_stage import judge_dev_stage

client = AsyncAnthropic(api_key=get_settings().anthropic_api_key)
MODEL = sys.argv[1] if len(sys.argv) > 1 else "claude-sonnet-4-6"

# --- stage 2: interpretive call (the proven prompt from interpretive_fields_harness) ----------
INTERP_PROMPT = """You are a drug-repurposing analyst writing four interpretive fields for a \
candidate card. You are GIVEN the authoritative facts below — already decided. Interpret them; \
do NOT restate, re-derive, or contradict them.

AUTHORITATIVE FACTS (ground truth):
- Development stage: {stage}
- Active programs (what is still moving): {active_programs}
- Literature: {literature}
- Approval relationship: {approval}

Write four fields:
- constraint: what holds this back (regulatory/commercial/evidence gap). One line. If a Phase 3 \
is completed or active, do NOT write "no Phase 3" / "no dedicated development program".
- key_risk: the single biggest risk. One line. Phase-free.
- assessment: a short interpretive verdict tag. Do NOT name a phase tier.
- prose: EXACTLY 2 sentences, consistent with the stage and active programs. If literature \
direction is "contradicts", surface that the drug failed.

Respond with ONLY JSON: \
{{"constraint":"...","key_risk":"...","assessment":"...","prose":"..."}}"""

# dev_stage tier -> the authoritative stage phrase (mirrors supervisor _DEV_STAGE_PHRASE).
STAGE_PHRASE = {
    "phase3_terminated_for_cause": "Phase 3 terminated for cause (safety/efficacy stop)",
    "completed_phase3": "Phase 3 completed for this indication",
    "active_phase3": "Active Phase 3 development on record for this indication",
    "phase3_unknown_status": "Phase 3 on record, status unknown",
    "completed_phase2": "Phase 2 completed for this indication, no Phase 3",
    "exploratory_phase4_only": (
        "Phase 4 exploratory only (post-approval off-label study; no dedicated "
        "development program for this indication)"
    ),
    "early_phase": "Early-phase only, no completed pivotal readout",
    "untested": "No trials on record for this indication",
}

_BAD = (
    "no dedicated phase 2/3", "no dedicated phase 2 or phase 3", "no phase 2/3 program",
    "no dedicated development program", "no formal development program",
    "no development program", "exploratory only", "phase 4 only", "no phase 3",
    "post-phase 2", "no pivotal program",
)


async def interp(facts):
    resp = await client.messages.create(
        model=MODEL, max_tokens=400,
        messages=[{"role": "user", "content": INTERP_PROMPT.format(**facts)}],
    )
    text = resp.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```")[1].lstrip("json").strip()
    return json.loads(text)


def _t(nct, phase, status):
    return Trial(nct_id=nct, phase=phase, overall_status=status)


# Real T1DM-shape trial set (the one that broke every monolithic report).
T1DM = [
    _t("NCT05537233", "Phase 2", "COMPLETED"),
    _t("NCT05857085", "Phase 4", "COMPLETED"),
    _t("NCT05205928", "Phase 2/Phase 3", "COMPLETED"),
    _t("NCT06082063", "Phase 3", "Recruiting"),
    _t("NCT05819138", "Phase 3", "Recruiting"),
    _t("NCT06909006", "Phase 3", "Not yet recruiting"),
    _t("NCT03899402", "Phase 2/Phase 3", "Active, not recruiting"),
]

CASES = [
    ("semaglutide x T1DM (the recurring break)", "semaglutide", "type 1 diabetes mellitus",
     T1DM, "Moderate, RCT-backed controlled studies", "related_family (type 2 diabetes)"),
    ("contradicts: drug failed", "drugX", "disease Y",
     [_t("NCT_A", "Phase 3", "COMPLETED")],
     "Strong, contradicts — RCTs show no benefit", "none"),
    ("exploratory Phase 4 only", "drugZ", "disease W",
     [_t("NCT_A", "Phase 4", "COMPLETED")], "Weak, observational", "related_family"),
    ("Phase 3 terminated for SAFETY (closed)", "drugT", "disease S",
     [_t("NCT_A", "Phase 3", "Terminated (safety concerns)"),
      _t("NCT_B", "Phase 2", "COMPLETED")],
     "Moderate, mixed", "none"),
    ("completed P2 only, recruiting P2 ('no Phase 3' is ACCURATE)", "drugP", "disease Q",
     [_t("NCT_A", "Phase 2", "COMPLETED"), _t("NCT_B", "Phase 2", "Recruiting")],
     "Moderate, supports", "none"),
    ("untested, rationale only", "drugU", "disease R",
     [], "Weak, preclinical only", "none"),
    ("active P3 only, none completed", "drugA", "disease V",
     [_t("NCT_A", "Phase 3", "Recruiting"), _t("NCT_B", "Phase 2", "COMPLETED")],
     "Moderate, supports", "none"),
    ("completed P3 + active P3 + adverse safety signal", "drugS", "disease T",
     [_t("NCT_A", "Phase 3", "COMPLETED"), _t("NCT_B", "Phase 3", "Recruiting")],
     "Strong, mixed — efficacy signal but adverse safety reports", "none"),
    ("unknown-status Phase 3 only", "drugK", "disease N",
     [_t("NCT_A", "Phase 3", "UNKNOWN")], "Moderate, supports", "none"),
]


async def run_case(label, drug, indication, trials, literature, approval):
    # Stage 1 — facts (real service, no cache dir pollution: use a tmp path).
    from pathlib import Path
    import tempfile
    cache = Path(tempfile.mkdtemp())
    j = await judge_dev_stage(trials, cache, drug=drug, indication=indication)
    stage_phrase = STAGE_PHRASE[j.tier]

    # Stage 2 — interpretation, FED stage-1 output.
    facts = {
        "stage": stage_phrase, "active_programs": j.active_programs,
        "literature": literature, "approval": approval,
    }
    fields = await interp(facts)

    # Assemble + check internal consistency.
    card = {"stage": stage_phrase, "active_programs": j.active_programs, **fields}
    s = stage_phrase.lower()
    asserts_p3 = any(
        x in s for x in ("phase 3 completed", "active phase 3", "phase 3 development")
    )
    blob = " ".join(
        str(card.get(k, "")) for k in ("constraint", "key_risk", "assessment", "prose")
    ).lower()
    bad = [p for p in _BAD if p in blob] if asserts_p3 else []

    print(f"[{'PASS' if not bad else 'FAIL'}] {label}")
    print(f"    stage={card['stage']!r}  active={card['active_programs']!r}")
    if bad:
        print(f"    !! CONTRADICTION {bad}")
        for k in ("constraint", "key_risk", "assessment", "prose"):
            print(f"      {k}: {card[k]}")
    print()


async def main():
    print(f"=== model: {MODEL} ===\n")
    for c in CASES:
        await run_case(*c)


if __name__ == "__main__":
    asyncio.run(main())
