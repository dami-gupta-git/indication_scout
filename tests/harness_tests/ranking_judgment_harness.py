"""Test the new LLM-judgment supervisor RANKING (no critique_ranking guard).

The supervisor prompt now hands ranking to the LLM instead of a fixed tier ladder. This harness
feeds the REAL supervisor.txt as the system prompt plus a synthetic per-candidate label block in
the EXACT shape the supervisor sees at ranking time (dev_stage tier + completed/active Phase 3 +
contaminated-excluded count + closure verdict + literature strength/direction/design/basis), and
asks ONLY for the ranked disease order. No tools, no critique step — this isolates whether the
prompt alone produces a sensible order from the labels.

Each case lists candidates (already de-identified to invented diseases to avoid the model leaning
on real-world priors) and an `expect` predicate over the returned order.

Run: .venv/bin/python tests/harness_tests/ranking_judgment_harness.py [model]
"""

import asyncio
import json
import re
import sys
from pathlib import Path

from anthropic import AsyncAnthropic

from indication_scout.config import get_settings

client = AsyncAnthropic(api_key=get_settings().anthropic_api_key)
MODEL = sys.argv[1] if len(sys.argv) > 1 else "claude-sonnet-4-6"
RUNS_PER_CASE = 5

_PROMPTS_DIR = Path(__file__).parent.parent / "src" / "indication_scout" / "prompts"
SUPERVISOR_PROMPT = (_PROMPTS_DIR / "supervisor.txt").read_text()


def _candidate_block(c: dict) -> str:
    """Render one candidate the way the supervisor sees it (literature header + derived signals)."""
    lines = [
        f"### Candidate: {c['disease']}",
        (
            f"Literature for DRUG × {c['disease']}: {c.get('pmids', 0)} PMIDs, "
            f"strength={c['strength']}, direction={c['direction']}, "
            f"study_design={c['design']}, evidence_basis={c['basis']}."
        ),
        "DERIVED SIGNALS (authoritative facts — relevant trials only):",
        f"  highest_completed_phase: {c.get('highest_completed_phase', 'none')}",
        f"  completed_phase_3: {c.get('completed_phase_3', 'no')}",
        f"  active_phase_3: {c.get('active_phase_3', 'no')}",
        f"  relevant_phase3_terminated_for_cause: {c.get('terminated_for_cause', 'no')}",
        f"  dev_stage: {c['dev_stage']}",
    ]
    if c.get("contaminated_excluded"):
        lines.append(f"  contaminated_excluded: {c['contaminated_excluded']} trial(s)")
    lines.append(
        f"Sub-agent closure verdict (trust, do not re-judge): closure={c.get('closure', 'live')}"
    )
    return "\n".join(lines)


def _task(candidates: list[dict]) -> str:
    blocks = "\n\n".join(_candidate_block(c) for c in candidates)
    names = [c["disease"] for c in candidates]
    return (
        "You have already investigated the following candidates for drug DRUG this run "
        "(both analyze_literature and analyze_clinical_trials ran for each). Using the RANKING "
        "guidance in your instructions, rank them best-repurposing-signal first.\n\n"
        f"{blocks}\n\n"
        f"Candidates to rank: {names}\n\n"
        'Respond with ONLY a JSON object: {"order": ["<disease>", ...]} listing every candidate '
        "exactly once, best first. No prose, no fences."
    )


def first(order: list[str]) -> str:
    return order[0] if order else ""


def last(order: list[str]) -> str:
    return order[-1] if order else ""


def before(order: list[str], a: str, b: str) -> bool:
    return a in order and b in order and order.index(a) < order.index(b)


# Invented diseases (alpha/beta/gamma...) so the model ranks on the LABELS, not real-world priors.
CASES = [
    {
        "label": "drug_specific_supports beats higher-stage approved-basis",
        "candidates": [
            # Higher trial stage but literature is approved-sub-indication (strength none).
            {
                "disease": "Alphosis",
                "dev_stage": "completed_phase3",
                "completed_phase_3": "yes",
                "highest_completed_phase": "Phase 3",
                "strength": "none",
                "direction": "none",
                "design": "undetermined",
                "basis": "approved",
            },
            # Lower stage but genuine drug-specific supportive RCT literature.
            {
                "disease": "Betalgia",
                "dev_stage": "completed_phase2",
                "highest_completed_phase": "Phase 2",
                "strength": "strong",
                "direction": "supports",
                "design": "rct_or_controlled",
                "basis": "drug_specific",
            },
        ],
        # Genuine drug-specific evidence should outrank the poorly-grounded higher-stage one.
        "expect": lambda o: before(o, "Betalgia", "Alphosis"),
    },
    {
        "label": "no-abstracts-yet early signal NOT buried below contradicts",
        "candidates": [
            # Real completed Phase 2 of THIS drug, no publications yet (basis none).
            {
                "disease": "Gammatosis",
                "dev_stage": "completed_phase2",
                "highest_completed_phase": "Phase 2",
                "strength": "none",
                "direction": "none",
                "design": "undetermined",
                "basis": "none",
            },
            # Strong literature but DISPROVEN (contradicts).
            {
                "disease": "Deltemia",
                "dev_stage": "completed_phase3",
                "completed_phase_3": "yes",
                "highest_completed_phase": "Phase 3",
                "strength": "strong",
                "direction": "contradicts",
                "design": "rct_or_controlled",
                "basis": "drug_specific",
            },
        ],
        # Disproven candidate goes to the bottom; the early no-lit-yet signal outranks it.
        "expect": lambda o: before(o, "Gammatosis", "Deltemia"),
    },
    {
        "label": "contradicts + closed both at bottom",
        "candidates": [
            {
                "disease": "Epsilitis",
                "dev_stage": "completed_phase2",
                "highest_completed_phase": "Phase 2",
                "strength": "moderate",
                "direction": "supports",
                "design": "rct_or_controlled",
                "basis": "drug_specific",
            },
            {
                "disease": "Zetawasting",
                "dev_stage": "phase3_terminated_for_cause",
                "terminated_for_cause": "yes",
                "highest_completed_phase": "none",
                "strength": "weak",
                "direction": "mixed",
                "design": "observational",
                "basis": "drug_specific",
                "closure": "closed — safety termination",
            },
            {
                "disease": "Etacline",
                "dev_stage": "completed_phase3",
                "completed_phase_3": "yes",
                "highest_completed_phase": "Phase 3",
                "strength": "strong",
                "direction": "contradicts",
                "design": "rct_or_controlled",
                "basis": "drug_specific",
            },
        ],
        # The clean supportive candidate is #1; the closed and the contradicts ones sink below it.
        "expect": lambda o: first(o) == "Epsilitis"
        and before(o, "Epsilitis", "Zetawasting")
        and before(o, "Epsilitis", "Etacline"),
    },
    {
        "label": "higher completed phase wins among clean drug-specific supports",
        "candidates": [
            {
                "disease": "Thetacosis",
                "dev_stage": "completed_phase3",
                "completed_phase_3": "yes",
                "highest_completed_phase": "Phase 3",
                "strength": "moderate",
                "direction": "supports",
                "design": "rct_or_controlled",
                "basis": "drug_specific",
            },
            {
                "disease": "Iotremia",
                "dev_stage": "completed_phase2",
                "highest_completed_phase": "Phase 2",
                "strength": "moderate",
                "direction": "supports",
                "design": "rct_or_controlled",
                "basis": "drug_specific",
            },
        ],
        "expect": lambda o: before(o, "Thetacosis", "Iotremia"),
    },
    {
        "label": "class-level signal does not outrank drug-specific supports",
        "candidates": [
            {
                "disease": "Kappadrome",
                "dev_stage": "active_phase3",
                "active_phase_3": "yes",
                "highest_completed_phase": "none",
                "strength": "none",
                "direction": "none",
                "design": "undetermined",
                "basis": "class_level",
            },
            {
                "disease": "Lambdosis",
                "dev_stage": "completed_phase2",
                "highest_completed_phase": "Phase 2",
                "strength": "moderate",
                "direction": "supports",
                "design": "rct_or_controlled",
                "basis": "drug_specific",
            },
        ],
        "expect": lambda o: before(o, "Lambdosis", "Kappadrome"),
    },
]


def _parse_order(text: str) -> list[str]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return []
    try:
        return list(json.loads(m.group(0)).get("order", []))
    except json.JSONDecodeError:
        return []


async def _run_one(case: dict) -> list[str]:
    resp = await client.messages.create(
        model=MODEL,
        max_tokens=400,
        system=SUPERVISOR_PROMPT,
        messages=[{"role": "user", "content": _task(case["candidates"])}],
    )
    return _parse_order(resp.content[0].text)


async def main() -> None:
    print(f"Model: {MODEL}  |  {RUNS_PER_CASE} runs/case\n")
    total_pass = 0
    total = 0
    for case in CASES:
        results = await asyncio.gather(
            *(_run_one(case) for _ in range(RUNS_PER_CASE))
        )
        passes = sum(1 for o in results if o and case["expect"](o))
        total_pass += passes
        total += RUNS_PER_CASE
        flag = "OK " if passes == RUNS_PER_CASE else "!! "
        print(f"{flag}{case['label']}: {passes}/{RUNS_PER_CASE}")
        if passes < RUNS_PER_CASE:
            for o in results:
                print(f"      order={o}")
    print(f"\nTOTAL: {total_pass}/{total}")


if __name__ == "__main__":
    asyncio.run(main())
