"""Does the trial relevance gate count only trials from which THIS drug's own effect can be read?

Bug (2026-11-16): bupropion x bipolar disorder counted three trials in which bupropion is one option among many, and
one of them (NCT05973786, recruiting Phase 3) set bipolar's stage to "active Phase 3":
- NCT00012558 STEP-BD: a treatment programme listing eight drugs and three psychotherapies, no arm groups reported.
- NCT05973786, NCT05603104: the bipolar arm is "switch to one of the following combinations" over seven drugs.
Also a strategy trial, NCT03029702 (metformin x gestational diabetes): both arms give insulin, glyburide and metformin
and differ only in how the drug is chosen.

Controls are multi-drug trials whose arms isolate the drug: one arm differs from a comparison arm by THIS drug.

Feeds the production clinical_trials.txt prompt (or a candidate prompt file given as the second argument) and the
production row formatter with the trials' real registry records, batched per drug x disease. Run with:

    .venv/bin/python tests/harness_tests/trial_relevance_own_effect_harness.py [runs] [prompt_path]
"""

import asyncio
import logging
import sys
from pathlib import Path

from pydantic import BaseModel

from indication_scout.agents._trial_formatting import _format_trial_row
from indication_scout.constants import DEFAULT_CACHE_DIR
from indication_scout.data_sources.clinical_trials import ClinicalTrialsClient
from indication_scout.services.llm import parse_last_json_object, query_llm

logger = logging.getLogger(__name__)

_PRODUCTION_PROMPT = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "indication_scout"
    / "prompts"
    / "clinical_trials.txt"
)

# The columns the production search view shows the relevance gate.
_COLUMNS = (
    "nct_id",
    "phase",
    "status",
    "interventions",
    "arms",
    "mesh",
    "brief_summary",
    "title",
)


class TrialCase(BaseModel):
    nct: str
    relevant: bool
    note: str


class PairCase(BaseModel):
    drug: str
    disease: str
    approved: list[str]
    trials: list[TrialCase]


_CASES: tuple[PairCase, ...] = (
    PairCase(
        drug="bupropion",
        disease="bipolar disorder",
        approved=[
            "major depressive disorder",
            "seasonal affective disorder",
            "smoking cessation",
        ],
        trials=[
            TrialCase(
                nct="NCT00012558",
                relevant=True,
                note="STEP-BD programme, eight drugs, no arms reported",
            ),
            TrialCase(
                nct="NCT05973786",
                relevant=True,
                note="bipolar arm: switch to one of seven drugs",
            ),
            TrialCase(
                nct="NCT05603104",
                relevant=False,
                note="one of seven drugs, but also enrols major depressive disorder, an approved indication",
            ),
        ],
    ),
    PairCase(
        drug="metformin",
        disease="gestational diabetes",
        approved=["type 2 diabetes mellitus"],
        trials=[
            TrialCase(
                nct="NCT03029702",
                relevant=False,
                note="strategy trial, both arms insulin/glyburide/metformin",
            ),
        ],
    ),
    PairCase(
        drug="metformin",
        disease="polycystic ovary syndrome",
        approved=["type 2 diabetes mellitus"],
        trials=[
            TrialCase(
                nct="NCT05394142",
                relevant=True,
                note="SPIOMET vs SPIO isolates metformin",
            ),
            TrialCase(nct="NCT06170463", relevant=True, note="metformin-alone arm"),
            TrialCase(
                nct="NCT06405178",
                relevant=True,
                note="metformin + letrozole vs control",
            ),
        ],
    ),
    PairCase(
        drug="metformin",
        disease="obesity disorder",
        approved=["type 2 diabetes mellitus"],
        trials=[
            TrialCase(nct="NCT01351753", relevant=True, note="metformin arm vs placebo")
        ],
    ),
    PairCase(
        drug="semaglutide",
        disease="type 1 diabetes mellitus",
        approved=["type 2 diabetes mellitus", "obesity"],
        trials=[
            TrialCase(
                nct="NCT03899402",
                relevant=True,
                note="insulin + semaglutide vs insulin",
            )
        ],
    ),
    PairCase(
        drug="sildenafil",
        disease="infertility",
        approved=["erectile dysfunction", "pulmonary arterial hypertension"],
        trials=[
            TrialCase(
                nct="NCT05753098",
                relevant=True,
                note="sildenafil + clomiphene vs clomiphene alone",
            )
        ],
    ),
)


def _task(case: PairCase, rows: list[str]) -> str:
    return (
        f"DRUG FACT — drug under analysis: {case.drug}\n"
        f"DRUG FACT — FDA-approved indications of this drug: {', '.join(case.approved)}\n\n"
        f"Repurposing candidate (the broad indication under investigation): {case.disease}\n\n"
        "Shown trials to classify (every one needs a verdict):\n"
        + "\n".join(rows)
        + "\n\n"
        'Respond with ONLY a JSON object: {"verdicts": [{"nct": "...", "drug_role": '
        '"studied"|"comparator"|"background"|"absent", "verdict": "relevant"|"contaminated"}, ...]} '
        "— one per shown trial."
    )


async def _classify(system_prompt: str, task: str) -> dict[str, str]:
    parsed = parse_last_json_object(await query_llm(task, system=system_prompt))
    if not isinstance(parsed, dict):
        return {}
    return {
        str(v.get("nct")): str(v.get("verdict", "")).strip().lower()
        for v in parsed.get("verdicts", [])
        if isinstance(v, dict)
    }


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    if runs < 1:
        raise ValueError("runs must be at least 1")
    prompt_path = Path(sys.argv[2]) if len(sys.argv) > 2 else _PRODUCTION_PROMPT
    system_prompt = prompt_path.read_text()
    logger.info("Prompt: %s", prompt_path)

    async with ClinicalTrialsClient(cache_dir=DEFAULT_CACHE_DIR) as client:
        tasks = []
        for case in _CASES:
            trials = [await client.get_trial(t.nct) for t in case.trials]
            tasks.append(_task(case, [_format_trial_row(t, _COLUMNS) for t in trials]))

    results = await asyncio.gather(
        *(_classify(system_prompt, task) for task in tasks for _ in range(runs))
    )
    all_ok = True
    for index, case in enumerate(_CASES):
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
