"""Signal-ablation harness for per-trial relevance tagging.

Question: which trial fields let the LLM reliably + completely tag each trial
RELEVANT vs CONTAMINATED for sildenafil × (systemic) hypertension? The registry
search recalls PAH/PH trials (a distinct disease sildenafil is already approved
for) plus other-drug trials (sitaxsentan etc.). MeSH is known unreliable — it
tags an Alzheimer's trial "PAH" and an ED trial "Hypertension".

Two conditions, one batched LLM call each (forced per-trial verdict):
  A. mesh-only   — current baseline; expected to under-perform
  B. title+interventions+summary — candidate fix

Scored vs hand labels: COVERAGE (every trial tagged?) and ACCURACY (tags right?).

Run: .venv/bin/python tests/harness_tests/trial_relevance_harness.py
"""

import asyncio
import json
from pathlib import Path

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

load_dotenv()
DATA = json.loads((Path(__file__).parent / "data" / "sild_htn_trials.json").read_text())
client = AsyncAnthropic()

# Ground truth: R = relevant to SYSTEMIC hypertension; C = contaminated
# (pulmonary hypertension / PAH / other disease / other drug). Hand-labeled from
# title + interventions + brief_summary.
LABELS: dict[str, str] = {
    "NCT05039086": "C",  # Alzheimer's repurposing (MeSH says PAH — misleading)
    "NCT00666198": "C",  # pulmonary, long-term surveillance
    "NCT01266265": "C",  # PAH therapies AE surveillance
    "NCT02565030": "C",  # CTEPH (pulmonary)
    "NCT03364244": "C",  # Revatio pediatric PAH
    "NCT00000623": "C",  # thalassemia network
    "NCT00303459": "C",  # PAH (COMPASS-2)
    "NCT00644605": "C",  # PAH
    "NCT00159861": "C",  # PAH
    "NCT00159887": "C",  # PAH
    "NCT00150358": "C",  # erectile dysfunction (MeSH says Hypertension)
    "NCT00159913": "C",  # pediatric PAH
    "NCT00159874": "C",  # pediatric PAH
    "NCT01365585": "C",  # PAH
    "NCT02891850": "C",  # PAH (riociguat)
    "NCT00862043": "C",  # secondary PH from valve disease
    "NCT00517933": "C",  # IPF
    "NCT00223717": "R",  # supine hypertension / autonomic failure (systemic-adjacent)
    "NCT02484807": "C",  # PAH
    "NCT01392638": "R",  # resistant hypertensives — systemic
    "NCT00323297": "C",  # PAH
    "NCT01043627": "C",  # PAH (SITAR)
    "NCT05546125": "C",  # PAH
    "NCT00433329": "C",  # PAH
    "NCT01757782": "C",  # PPHN newborns
    "NCT01055405": "C",  # COPD-associated PH
    "NCT02378649": "C",  # post-MV-surgery PH
    "NCT01726049": "C",  # HFpEF + PH
    "NCT01548950": "C",  # PAH-CHD
    "NCT02620995": "R",  # hypertensive men w/ ED — systemic HTN cohort
    "NCT00454207": "C",  # PAH
    "NCT01156636": "C",  # PH in diastolic HF
    "NCT01409122": "C",  # other drug (sodium nitrite), healthy subjects
    "NCT00334490": "C",  # post-cardiac-surgery PH
    "NCT00617305": "C",  # PAH (ATHENA-1)
    "NCT00946114": "C",  # PAH
    "NCT00599235": "R",  # exercise capacity in hypertension — systemic
    "NCT02595541": "C",  # postoperative PH
    "NCT00359736": "C",  # IPF
    "NCT00872170": "C",  # thalassemia + PH
    "NCT04704440": "C",  # PH + hypoxia exercise
    "NCT04697862": "C",  # PH + hypoxia exercise
    "NCT01181284": "C",  # PAH (MELISSA)
    "NCT04715113": "C",  # PH + hypoxia exercise
    "NCT04706546": "C",  # PH exercise
    "NCT00808912": "C",  # athletic performance / air pollution (not HTN therapy)
    "NCT01391104": "C",  # PAH exercise tests
    "NCT01392469": "C",  # PAH PK (imatinib)
    "NCT00352482": "C",  # IPF + PH
    "NCT01044693": "R",  # supine hypertension / autonomic failure
    "NCT02060487": "C",  # PAH mortality
    "NCT00795639": "C",  # sitaxsentan (other drug), PAH
    "NCT00796666": "C",  # sitaxsentan (other drug), PAH
    "NCT00430716": "C",  # PAH
    "NCT02284737": "C",  # PAH (PADN)
    "NCT00492531": "C",  # sickle cell + PH
    "NCT00302211": "C",  # PAH (VISION)
    "NCT00853112": "C",  # other drug (PF-00489791), PAH
    "NCT00586794": "C",  # Eisenmenger PAH
    "NCT00327080": "C",  # HIV-associated PH
    "NCT01244620": "C",  # sitaxsentan/tadalafil DDI (other drugs)
    "NCT00145938": "C",  # portopulmonary hypertension
    "NCT00742014": "C",  # RV contractility, PAH context
    "NCT00133679": "C",  # diaphragmatic hernia (PPHN)
    "NCT00718185": "C",  # PK/PD only, pulmonary
    "NCT00796510": "C",  # sitaxsentan (other drug), PAH
    "NCT01409031": "C",  # PPHN newborns
    "NCT02435303": "C",  # PAH after MV surgery (SUPERIOR)
}

SYSTEM = (
    "You are a clinical-trials relevance classifier. The candidate repurposing "
    "indication is SYSTEMIC HYPERTENSION (ordinary high blood pressure). The "
    "registry search over-recalls: it returns trials for PULMONARY hypertension / "
    "PAH (a DISTINCT disease) and trials whose primary drug is NOT sildenafil. "
    "For EACH trial id given, output a verdict: 'relevant' if it studies SILDENAFIL "
    "for SYSTEMIC hypertension, else 'contaminated'. You MUST return a verdict for "
    "every id — omit none. Return JSON: {\"verdicts\": [{\"nct\": str, \"verdict\": "
    "\"relevant\"|\"contaminated\"}]}."
)


def _render(condition: str) -> str:
    lines = []
    for r in DATA:
        if condition == "mesh":
            lines.append(f"{r['nct']} | phase {r['phase']} | mesh: {'; '.join(r['mesh']) or '(none)'}")
        else:  # rich
            lines.append(
                f"{r['nct']} | phase {r['phase']} | drugs: {'; '.join(r['interv']) or '(none)'} "
                f"| {r['title']} | {r['summary']}"
            )
    return "\n".join(lines)


async def run(condition: str) -> dict:
    prompt = f"Classify these {len(DATA)} trials:\n\n{_render(condition)}"
    resp = await client.messages.create(
        model="claude-opus-4-8",
        max_tokens=4000,
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```")[1].lstrip("json").strip()
    verdicts = {v["nct"]: v["verdict"] for v in json.loads(text)["verdicts"]}

    all_ncts = {r["nct"] for r in DATA}
    tagged = set(verdicts)
    coverage = len(tagged & all_ncts) / len(all_ncts)
    scored = [n for n in all_ncts if n in verdicts]
    correct = sum(
        (verdicts[n] == "relevant") == (LABELS[n] == "R") for n in scored
    )
    accuracy = correct / len(scored) if scored else 0.0
    missed = sorted(all_ncts - tagged)
    wrong = sorted(
        n for n in scored if (verdicts[n] == "relevant") != (LABELS[n] == "R")
    )
    return {
        "condition": condition,
        "coverage": coverage,
        "accuracy": accuracy,
        "untagged": missed,
        "mistagged": [(n, verdicts[n], LABELS[n]) for n in wrong],
    }


async def main() -> None:
    for condition in ("mesh", "rich"):
        res = await run(condition)
        print(f"\n=== condition: {res['condition']} ===")
        print(f"coverage: {res['coverage']:.0%}  ({len(DATA)} trials)")
        print(f"accuracy: {res['accuracy']:.0%}")
        print(f"untagged ({len(res['untagged'])}): {res['untagged']}")
        print(f"mistagged ({len(res['mistagged'])}): {res['mistagged']}")


if __name__ == "__main__":
    asyncio.run(main())
