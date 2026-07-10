"""Test the THERAPEUTIC-INTENT clause of the per-trial relevance gate (TEST 2).

Anchor bug: sildenafil × systemic hypertension, NCT02620995 ("Sildenafil on Penile Vascular
Function in Hypertensive Men With Erectile Dysfunction") was tagged RELEVANT to systemic
hypertension. ED is sildenafil's approved indication; hypertension only names the POPULATION, not
the treatment target -> CONTAMINATION. The gate now has a therapeutic-intent clause in TEST 2.

Feeds the REAL clinical_trials.txt as the system prompt and a synthetic batch: the NCT02620995
shape + a genuine systemic-HTN trial (relevant control) + a PAH distinct-disease trial
(contaminated) + an ED-only trial (contaminated, approved indication).

Run: .venv/bin/python tests/harness_tests/trial_relevance_intent_harness.py [model]
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
RUNS = 8

_PROMPTS_DIR = Path(__file__).parent.parent / "src" / "indication_scout" / "prompts"
SYSTEM_PROMPT = (_PROMPTS_DIR / "clinical_trials.txt").read_text()

DRUG = "sildenafil"
CANDIDATE = "hypertension"  # systemic
APPROVED = ["pulmonary arterial hypertension", "erectile dysfunction"]

TRIALS = [
    {
        "nct": "NCT02620995",  # the anchor bug
        "title": "Sildenafil on Penile Vascular Function in Hypertensive Men With Erectile Dysfunction",
        "summary": "Effect of sildenafil on penile vascular function in men with erectile "
        "dysfunction who also have hypertension.",
        "drug": "sildenafil",
        "expected": "contaminated",  # target = ED (approved); hypertension only the population
    },
    {
        "nct": "NCT01000010",
        "title": "Sildenafil for Lowering Blood Pressure in Resistant Systemic Hypertension",
        "summary": "Randomized trial of sildenafil to reduce ambulatory blood pressure in adults "
        "with resistant essential (systemic) hypertension.",
        "drug": "sildenafil",
        "expected": "relevant",  # genuinely targets systemic hypertension
    },
    {
        "nct": "NCT01000011",
        "title": "Sildenafil in Pulmonary Arterial Hypertension (WHO Group I)",
        "summary": "Sildenafil to improve exercise capacity in PAH.",
        "drug": "sildenafil",
        "expected": "contaminated",  # PAH = approved, distinct disease (TEST 1/2)
    },
    {
        "nct": "NCT01000012",
        "title": "Sildenafil for Erectile Dysfunction",
        "summary": "Sildenafil in men with erectile dysfunction.",
        "drug": "sildenafil",
        "expected": "contaminated",  # ED = approved indication
    },
]


def _task() -> str:
    rows = "\n".join(
        f"- {t['nct']}: drugs=[{t['drug']}], title=\"{t['title']}\", summary=\"{t['summary']}\""
        for t in TRIALS
    )
    return (
        f"DRUG FACT — drug under analysis: {DRUG}\n"
        f"DRUG FACT — FDA-approved indications of this drug: {', '.join(APPROVED)}\n\n"
        f"Repurposing candidate (the broad indication under investigation): {CANDIDATE} (systemic)\n\n"
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


async def _run_one() -> dict[str, str]:
    resp = await client.messages.create(
        model=MODEL,
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _task()}],
    )
    return _parse(resp.content[0].text)


async def main() -> None:
    print(f"Model: {MODEL}  |  {RUNS} runs\n")
    results = await asyncio.gather(*(_run_one() for _ in range(RUNS)))
    per = {t["nct"]: 0 for t in TRIALS}
    for v in results:
        for t in TRIALS:
            if v.get(t["nct"]) == t["expected"]:
                per[t["nct"]] += 1
    for t in TRIALS:
        p = per[t["nct"]]
        flag = "OK " if p == RUNS else "!! "
        print(f"{flag}{t['nct']} (expect {t['expected']}): {p}/{RUNS}  — {t['title'][:48]}")
    total = sum(per.values())
    print(f"\nTOTAL: {total}/{len(TRIALS) * RUNS}")


if __name__ == "__main__":
    asyncio.run(main())
