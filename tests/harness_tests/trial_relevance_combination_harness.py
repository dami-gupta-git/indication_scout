"""Does the trial relevance gate count a trial that tests THIS drug plus one other named agent, and still drop a fixed-dose product?

Bug (2026-11-25): metformin x Duchenne muscular dystrophy reported zero relevant trials. Three completed trials give
metformin together with L-citrulline or L-arginine against placebo (NCT01995032 Phase 3, NCT02018731, NCT02516085). The
combination-only rule excluded them because no arm gave metformin alone.

The rule now excludes only a trial whose experimental intervention is one coformulated product with no arm that isolates
this drug. A regimen the investigators designed as this drug plus one other named agent counts, with or without a
monotherapy arm.

Controls: a trial with a metformin-alone arm (NCT04268563), a bupropion trial with a monotherapy arm (NCT04167306), a
trial that treats a different target in the same patients (NCT00720161 treats obesity, not the dystrophy), and every case
in trial_relevance_own_effect_harness.py, so the earlier multi-drug-programme fixes still hold. Two of those cases
(NCT03029702 and NCT05394142) already failed on the prompt before this change.

Feeds the production clinical_trials.txt prompt (or a candidate prompt file given as the second argument) and the production
row formatter with the trials' real registry records. Run with:

    .venv/bin/python tests/harness_tests/trial_relevance_combination_harness.py [runs] [prompt_path]
"""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from trial_relevance_own_effect_harness import (  # noqa: E402
    _CASES as _OWN_EFFECT_CASES,
    _COLUMNS,
    _PRODUCTION_PROMPT,
    PairCase,
    TrialCase,
    _classify,
    _task,
)

from indication_scout.agents._trial_formatting import _format_trial_row  # noqa: E402
from indication_scout.constants import DEFAULT_CACHE_DIR  # noqa: E402
from indication_scout.data_sources.clinical_trials import ClinicalTrialsClient  # noqa: E402

logger = logging.getLogger(__name__)

_CASES: tuple[PairCase, ...] = (
    PairCase(
        drug="metformin",
        disease="duchenne muscular dystrophy",
        approved=["type 2 diabetes mellitus"],
        trials=[
            TrialCase(
                nct="NCT01995032",
                relevant=True,
                note="metformin + L-citrulline vs placebo, Phase 3, no monotherapy arm",
            ),
            TrialCase(
                nct="NCT02018731",
                relevant=False,
                note="Becker muscular dystrophy, a sibling disease of the candidate: excluded by the sibling rule, not the combination rule",
            ),
            TrialCase(
                nct="NCT02516085",
                relevant=True,
                note="metformin + L-arginine, single arm, Phase 1",
            ),
            TrialCase(
                nct="NCT00720161",
                relevant=False,
                note="metformin for obesity and insulin resistance in children with a motor deficit",
            ),
        ],
    ),
    PairCase(
        drug="metformin",
        disease="polycystic ovary syndrome",
        approved=["type 2 diabetes mellitus"],
        trials=[
            TrialCase(
                nct="NCT04268563",
                relevant=True,
                note="sitagliptin/metformin product vs metformin alone vs sitagliptin vs placebo",
            ),
        ],
    ),
    PairCase(
        drug="bupropion",
        disease="alcohol use disorder",
        approved=[
            "major depressive disorder",
            "seasonal affective disorder",
            "smoking cessation",
        ],
        trials=[
            TrialCase(
                nct="NCT04167306",
                relevant=True,
                note="COMB: varenicline and bupropion, in combination and alone",
            ),
        ],
    ),
)


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    if runs < 1:
        raise ValueError("runs must be at least 1")
    prompt_path = Path(sys.argv[2]) if len(sys.argv) > 2 else _PRODUCTION_PROMPT
    system_prompt = prompt_path.read_text()
    logger.info("Prompt: %s", prompt_path)

    cases = _CASES + _OWN_EFFECT_CASES
    async with ClinicalTrialsClient(cache_dir=DEFAULT_CACHE_DIR) as client:
        tasks = []
        for case in cases:
            trials = [await client.get_trial(t.nct) for t in case.trials]
            tasks.append(_task(case, [_format_trial_row(t, _COLUMNS) for t in trials]))

    results = await asyncio.gather(
        *(_classify(system_prompt, task) for task in tasks for _ in range(runs))
    )
    all_ok = True
    for index, case in enumerate(cases):
        answers = results[index * runs : (index + 1) * runs]
        for trial in case.trials:
            expected = "relevant" if trial.relevant else "contaminated"
            passed = sum(answer.get(trial.nct) == expected for answer in answers)
            all_ok = all_ok and passed == runs
            logger.info(
                "%s %s x %s (%s): expected %s, %d/%d",
                trial.nct,
                case.drug,
                case.disease,
                trial.note,
                expected,
                passed,
                runs,
            )
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
