"""Harness for the candidate-card writer: does it contradict the report's own literature section?

The card writer (judge_interpretive) receives literature as a one-line label and a closure sentence written by the
trials step, which never sees the literature but asserts that "no literature reporting efficacy failure has been
identified". Two cards from the 2026-11-16 runs contradicted the sections below them:

- bupropion x cocaine dependence: card called the hypothesis "RCT-supported"; the literature says the RCTs were null.
- semaglutide x heart failure: card said there was no pivotal readout; the literature describes the STEP-HFpEF
  randomized trials, which the registry search missed because the registry tags them only as obesity and diabetes.

The fix (2026-11-16) labels each fact by source, passes the literature summary's text alongside its label, and limits
the trials step's closure sentence to what the registry shows. Each case runs the production prompt twice: with the
inputs the cards were written from (no summary, closure sentence asserting no literature failure) and with the
restructured inputs. Before the fix the old prompt passed 0/5 (cocaine) and 3/5 (heart failure); the restructured
version passed 5/5 on both. A grader call answers a yes/no question per case. Run with:

    .venv/bin/python tests/harness_tests/interpretive_reconcile_harness.py [runs]
"""

import asyncio
import logging
import sys

from pydantic import BaseModel

from indication_scout.services.judge_interpretive import (
    _INTERP_PROMPT,
    _parse_interpretive,
)
from indication_scout.services.llm import (
    parse_last_json_object,
    query_llm,
    query_small_llm,
)

logger = logging.getLogger(__name__)

_RELATIONSHIP_NONE = "no notable relationship to an approved indication"



class CardCase(BaseModel):
    name: str
    production: dict[str, str]  # inputs the 2026-11-16 cards were written from
    candidate: dict[str, str]  # restructured inputs
    fail_question: str


_COCAINE_SUMMARY = (
    "The evidence for bupropion in cocaine dependence is mixed, with controlled clinical trials generally failing to "
    "show a significant overall benefit versus placebo (PMID: 18551884; PMID: 36920397), while some subgroup and "
    "contingency-management combination analyses suggest potential benefit in specific populations (PMID: 16461866; "
    "PMID: 36920397). Supporting evidence comes primarily from a small uncontrolled case series (PMID: 11916368) and "
    "preclinical rat data (PMID: 39881654), which are insufficient to offset the null findings from the "
    "higher-quality randomized controlled trials."
)
_HF_SUMMARY = (
    "The totality of evidence from multiple large randomized controlled trials, prespecified subgroup analyses, pooled "
    "analyses, and meta-analyses consistently supports semaglutide as beneficial in heart failure, particularly HFpEF. "
    "In the STEP-HFpEF and STEP-HFpEF DM trials, semaglutide significantly improved KCCQ scores, body weight, 6-minute "
    "walk distance, and hierarchical composite endpoints versus placebo (PMIDs 37622681, 38587233, 38599221), and a "
    "pooled analysis across four RCTs demonstrated that semaglutide reduced the composite of cardiovascular death or "
    "worsening heart failure events by 31% in HFpEF patients (PMID 39222642). The single contradicting meta-analysis "
    "(PMID 41272444) found no significant effect on heart failure hospitalizations in its pooled MACE-focused "
    "analysis, but this is outweighed by the consistent, direct, and more granular evidence from dedicated heart "
    "failure trials and larger pooled analyses showing significant reductions in HF events."
)

_CASES: tuple[CardCase, ...] = (
    CardCase(
        name="bupropion x cocaine dependence",
        production={
            "stage": "Phase 2/Phase 3 completed for this indication (no standalone Phase 3)",
            "active_programs": "None active",
            "literature": "strong, mixed, RCT-backed / controlled, ⚠️ safety signal reported for this indication",
            "literature_summary": "none",
            "trial_evidence": "2 relevant trial(s), complete review of 6 registry query match(es)",
            "closure": (
                "LIVE — not closed (No relevant Phase 3 termination for safety or benefit:risk reasons, and no "
                "literature reporting efficacy failure for this indication has been identified.)"
            ),
            "terminations": "none reported",
        },
        candidate={
            "stage": "Phase 2/Phase 3 completed for this indication (no standalone Phase 3)",
            "active_programs": "None active",
            "literature": "strong, mixed, RCT-backed / controlled, ⚠️ safety signal reported for this indication",
            "literature_summary": _COCAINE_SUMMARY,
            "trial_evidence": "2 relevant trial(s), complete review of 6 registry query match(es)",
            "closure": (
                "LIVE — not closed (registry trials only: no relevant Phase 3 termination for safety or "
                "benefit:risk reasons.)"
            ),
            "terminations": "none reported",
        },
        fail_question=(
            "Does the card describe the efficacy evidence for bupropion in cocaine dependence as positive, supported "
            "or RCT-supported, or say that no efficacy failure has been found?"
        ),
    ),
    CardCase(
        name="semaglutide x heart failure",
        production={
            "stage": "Early-phase only, no completed pivotal readout",
            "active_programs": "None active",
            "literature": "strong, supports, RCT-backed / controlled",
            "literature_summary": "none",
            "trial_evidence": "1 relevant trial(s), complete review of 9 registry query match(es)",
            "closure": (
                "LIVE — not closed (No relevant Phase 3 termination for safety or benefit:risk concerns, and no "
                "literature reporting efficacy failure for this indication has been identified.)"
            ),
            "terminations": "none reported",
        },
        candidate={
            "stage": "Early-phase only, no completed pivotal readout",
            "active_programs": "None active",
            "literature": "strong, supports, RCT-backed / controlled",
            "literature_summary": _HF_SUMMARY,
            "trial_evidence": "1 relevant trial(s), complete review of 9 registry query match(es)",
            "closure": (
                "LIVE — not closed (registry trials only: no relevant Phase 3 termination for safety or "
                "benefit:risk concerns.)"
            ),
            "terminations": "none reported",
        },
        fail_question=(
            "Does the card say or imply that semaglutide lacks large or pivotal randomized trial results in heart "
            "failure, without also saying that the published literature reports large randomized trials?"
        ),
    ),
)

_GRADER_PROMPT = """Read this candidate card and answer one question about it.

Card:
{card}

Question: {question}

Respond with ONLY one JSON object: {{"reason": "<one sentence>", "answer": "yes"|"no"}}"""


async def _write_card(prompt_template: str, facts: dict[str, str]) -> str | None:
    prompt = prompt_template.format(
        relationship=_RELATIONSHIP_NONE,
        approved_indication="none",
        **facts,
    )
    judgment = _parse_interpretive(await query_llm(prompt))
    if judgment is None:
        return None
    return (
        f"Constraint: {judgment.blocker}\nKey risk: {judgment.key_risk}\n"
        f"Assessment: {judgment.verdict}\nProse: {judgment.prose}"
    )


async def _fails(card: str, question: str) -> bool | None:
    parsed = parse_last_json_object(
        await query_small_llm(_GRADER_PROMPT.format(card=card, question=question))
    )
    if not isinstance(parsed, dict) or parsed.get("answer") not in ("yes", "no"):
        return None
    return parsed["answer"] == "yes"


async def _run_one(prompt_template: str, facts: dict[str, str], question: str) -> tuple[str | None, bool | None]:
    card = await _write_card(prompt_template, facts)
    if card is None:
        return None, None
    return card, await _fails(card, question)


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    if runs < 1:
        raise ValueError("runs must be at least 1")

    jobs = []
    for case in _CASES:
        for variant, template, facts in (
            ("production", _INTERP_PROMPT, case.production),
            ("candidate", _INTERP_PROMPT, case.candidate),
        ):
            for _ in range(runs):
                jobs.append((case, variant, _run_one(template, facts, case.fail_question)))
    results = await asyncio.gather(*(job[2] for job in jobs))

    candidate_ok = True
    for case in _CASES:
        for variant in ("production", "candidate"):
            outcomes = [
                result for job, result in zip(jobs, results, strict=True) if job[0] is case and job[1] == variant
            ]
            passed = sum(fail is False for _, fail in outcomes)
            if variant == "candidate":
                candidate_ok = candidate_ok and passed == runs
            logger.info("%s [%s]: %d/%d pass", case.name, variant, passed, runs)
            logger.info("  example card:\n%s", outcomes[0][0])
    if not candidate_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
