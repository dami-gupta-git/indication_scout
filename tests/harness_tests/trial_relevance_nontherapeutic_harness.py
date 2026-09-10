"""Test whether the relevance gate rejects trials whose STUDIED object is not the drug.

Anchor bug (metformin, 2026-09-07 run): NCT02440893 ("Understanding the Effect of Metformin on
Corus CAD") was tagged RELEVANT under coronary artery disorder and CONTAMINATED under
cardiovascular disorder. Metformin is not a registered intervention at all — the sole intervention
is a diagnostic test, and the primary outcome is whether metformin shifts the test's gene-expression
score. The same run counted NCT03122769 (an 11C-metformin PET tracer study) as heart-failure
evidence; its only intervention is of type Radiation.

TEST 0 asks whether this drug is the studied/experimental agent, but its clauses only cover the case
where ANOTHER DRUG is the subject. Nothing covers a trial whose subject is a diagnostic test, a
device, or a tracer, so the verdict falls back on the title — and both titles read as metformin
trials.

Feeds the REAL clinical_trials.txt as the system prompt and the REAL row formatter, so the rows are
shaped exactly as production sends them (including intervention types). Every trial is a real
ClinicalTrials.gov record; interventions, phases and summaries are transcribed from the v2 API.

Batched per candidate disease, matching production: each disease is classified in its own call, so a
contaminated verdict cannot be earned on a disease mismatch the gate would have caught anyway.

Run: .venv/bin/python tests/harness_tests/trial_relevance_nontherapeutic_harness.py [model]
"""

import asyncio
import json
import re
import sys
from pathlib import Path

from anthropic import AsyncAnthropic

from indication_scout.agents._trial_formatting import _format_trial_row
from indication_scout.config import get_settings
from indication_scout.models.model_clinical_trials import Intervention, Trial

client = AsyncAnthropic(api_key=get_settings().anthropic_api_key)
MODEL = sys.argv[1] if len(sys.argv) > 1 else "claude-sonnet-4-6"
RUNS = 8

_PROMPTS_DIR = Path(__file__).parents[2] / "src" / "indication_scout" / "prompts"
SYSTEM_PROMPT = (_PROMPTS_DIR / "clinical_trials.txt").read_text()

# Production columns for the completed-trials classification view.
_COLUMNS = ("nct_id", "phase", "interventions", "title", "brief_summary")

DRUG = "metformin"
APPROVED = ["type 2 diabetes mellitus"]

# (nct, phase, title, summary, [(intervention_type, intervention_name)], expected_verdict, note)
_CASES = [
    (
        "coronary artery disorder",
        [
            (
                "NCT02440893",
                "Not Applicable",
                "Understanding the Effect of Metformin on Corus CAD (or ASGES)",
                "The study goal was to understand the effect of Metformin on Age/Sex/Gene "
                'Expression Score (ASGES) or Corus CAD (henceforth "Corus") in pre-diabetic '
                "patients who are medication naive. This study provided data to determine if the "
                "Corus CAD (ASGES) signature was different in pre-diabetic patients when metformin "
                "was newly prescribed and taken.",
                [("Diagnostic Test", "Corus CAD (ASGES)")],
                "contaminated",
                "studied object is the diagnostic test; metformin is not an intervention",
            ),
            (
                "NCT02226510",
                "Phase 4",
                "MetfoRmin and Its Effects on Left Ventricular Hypertrophy in Normotensive "
                "Patients With Coronary Artery Disease",
                "Thickening of the heart muscle (left ventricle) known medically as Left "
                "Ventricular Hypertrophy (LVH) is very common in patients with heart disease. LVH "
                "may be seen in normotensive patients where factors such as obesity and insulin "
                "resistance are present.",
                [("Drug", "Metformin XL"), ("Drug", "Placebo")],
                "relevant",
                "control — metformin is the studied drug",
            ),
            (
                "NCT01438723",
                "Phase 4",
                "The Metformin in Coronary Artery Bypass Graft (CABG) (MetCAB) Trial",
                "In patients with a myocardial infarction, occlusion of a coronary artery induces "
                "myocardial ischemia and cell death. Reperfusion itself can also damage myocardial "
                "tissue and contribute to the final infarct size.",
                [("Drug", "Metformin"), ("Drug", "Placebo")],
                "relevant",
                "control — metformin is the studied drug",
            ),
        ],
    ),
    (
        "heart failure",
        [
            (
                "NCT03122769",
                "Phase 1",
                "Cardiac Uptake of Metformin, Visualized by Positron Emission Tomography",
                "The purpose of the study is to evaluate if metformin is taken up into the failing "
                "myocardium. The aim of the study is to investigate if metformin is taken up in "
                "heart failure using a novel 11C-metformin tracer and positron emission tomography "
                "(PET).",
                [("Radiation", "11C-metformin")],
                "contaminated",
                "tracer biodistribution study; no therapeutic arm, no clinical endpoint",
            ),
            (
                "NCT04549415",
                "Phase 4",
                "The Influence of Metformin on Chronic Heart Failure Clinical Course in Patients "
                "With Prediabetes",
                "Prediabetes is a predictor of high cardiovascular mortality. Insulin resistance is "
                "one of the crucial mechanisms for the development and progression of chronic heart "
                "failure (CHF).",
                [
                    ("Drug", "Metformin Hydrochloride"),
                    ("Other", "lifestyle modification"),
                ],
                "relevant",
                "control — non-drug intervention co-listed, metformin still the studied drug",
            ),
        ],
    ),
    (
        "atrial fibrillation",
        [
            (
                "NCT04625946",
                "Phase 4",
                "Metformin as an Adjunctive Therapy to Catheter Ablation in Atrial Fibrillation",
                "This clinical trial is being done to determine if metformin, a drug which is "
                "normally used in diabetes, can reduce atrial fibrillation in patients who are "
                "having an ablation for atrial fibrillation (AF). It is anticipated that the "
                "participants treated in the metformin arm will have greater freedom from recurrent "
                "atrial arrhythmias after ablation.",
                [
                    ("Drug", "Metformin"),
                    ("Behavioral", "Recommendations for lifestyle modification."),
                    ("Device", "AliveCor"),
                ],
                "relevant",
                "over-rejection control — device and behavioral arms must not exclude it",
            ),
        ],
    ),
]


def _row(nct: str, phase: str, title: str, summary: str, interventions) -> str:
    trial = Trial(
        nct_id=nct,
        phase=phase,
        title=title,
        brief_summary=summary,
        interventions=[
            Intervention(intervention_type=t, intervention_name=n)
            for t, n in interventions
        ],
    )
    return _format_trial_row(trial, _COLUMNS)


def _task(candidate: str, trials: list) -> str:
    rows = "\n".join(_row(t[0], t[1], t[2], t[3], t[4]) for t in trials)
    return (
        f"DRUG FACT — drug under analysis: {DRUG}\n"
        f"DRUG FACT — FDA-approved indications of this drug: {', '.join(APPROVED)}\n\n"
        f"Repurposing candidate (the broad indication under investigation): {candidate}\n\n"
        f"Shown trials to classify (every one needs a verdict):\n{rows}\n\n"
        'Respond with ONLY a JSON object: {"verdicts": [{"nct": "...", '
        '"verdict": "relevant"|"contaminated"}, ...]} — one per shown trial.'
    )


def _parse(text: str) -> dict[str, str]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    return {
        v.get("nct"): (v.get("verdict") or "").strip().lower()
        for v in data.get("verdicts", [])
        if isinstance(v, dict)
    }


async def _run_one(candidate: str, trials: list) -> dict[str, str]:
    resp = await client.messages.create(
        model=MODEL,
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _task(candidate, trials)}],
    )
    return _parse(resp.content[0].text)


async def main() -> None:
    print(f"Model: {MODEL}  |  {RUNS} runs per candidate\n")
    total_hits = 0
    total_slots = 0
    for candidate, trials in _CASES:
        results = await asyncio.gather(
            *(_run_one(candidate, trials) for _ in range(RUNS))
        )
        print(f"--- candidate: {candidate}")
        for nct, _phase, title, _summary, _ints, expected, note in trials:
            hits = sum(1 for r in results if r.get(nct) == expected)
            flag = "OK " if hits == RUNS else "!! "
            print(f"{flag}{nct} (expect {expected}): {hits}/{RUNS}  — {title[:52]}")
            print(f"      {note}")
            total_hits += hits
            total_slots += RUNS
        print()
    print(f"TOTAL: {total_hits}/{total_slots}")


if __name__ == "__main__":
    asyncio.run(main())
