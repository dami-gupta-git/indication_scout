"""Test TEST 1 (approved-subtype) of the per-trial relevance gate after the clarity refactor +
multi-condition rule.

Anchor bug: semaglutide × NAFLD, NCT04639414 ("combined active treatment in type 2 diabetes with
NASH") flipped from contaminated (correct — NASH is the approved subtype) to relevant across runs.
The compound "T2DM with NASH" condition let the model latch on T2DM and miss the NASH subtype. The
refactor adds: a trial contaminates if ANY listed condition is the approved subtype.

Feeds the REAL clinical_trials.txt as the system prompt and a synthetic batch (the NCT04639414
shape + regression anchors: severity-qualifier NASH, broad NAFLD kept, T1D sibling kept, PAH
distinct, NSCLC minority-biomarker kept, wrong-drug). Approved = "MASH (NASH) with fibrosis".

Run: .venv/bin/python tests/harness_tests/trial_relevance_test1_harness.py [model]
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

DRUG = "semaglutide"
CANDIDATE = "non-alcoholic fatty liver disease"
APPROVED = ["MASH (NASH) with moderate-to-advanced fibrosis", "type 2 diabetes mellitus", "obesity"]

# Each trial: nct, condition/title/summary, drug, expected verdict.
TRIALS = [
    {
        "nct": "NCT04639414",  # the anchor bug
        "title": "Combined Active Treatment in Type 2 Diabetes With NASH",
        "summary": "Semaglutide-containing regimen in patients with type 2 diabetes and NASH.",
        "drug": "semaglutide",
        "expected": "contaminated",  # NASH = approved subtype, despite T2DM co-listed
    },
    {
        "nct": "NCT01000001",
        "title": "Semaglutide in NASH With Stage 2-3 Fibrosis",
        "summary": "Once-weekly semaglutide in biopsy-confirmed NASH.",
        "drug": "semaglutide",
        "expected": "contaminated",  # bare NASH = approved subtype (severity qualifier ignored)
    },
    {
        "nct": "NCT01000002",
        "title": "Semaglutide Across the Broad NAFLD Spectrum Including Simple Steatosis",
        "summary": "Semaglutide in NAFLD patients without requiring steatohepatitis histology.",
        "drug": "semaglutide",
        "expected": "relevant",  # broad NAFLD umbrella = the repurposing candidate
    },
    {
        "nct": "NCT01000003",
        "title": "Semaglutide for Type 1 Diabetes",
        "summary": "Adjunct semaglutide in adults with type 1 diabetes.",
        "drug": "semaglutide",
        "expected": "contaminated",  # distinct disease from the NAFLD candidate (TEST 2)
    },
    {
        "nct": "NCT01000004",
        "title": "Efinopegdutide vs Semaglutide Comparator in NAFLD",
        "summary": "Efinopegdutide is the studied drug; semaglutide is the active comparator.",
        "drug": "efinopegdutide",
        "expected": "contaminated",  # wrong studied drug (TEST 2)
    },
]


def _task() -> str:
    approved_line = ", ".join(APPROVED)
    rows = "\n".join(
        f"- {t['nct']}: drugs=[{t['drug']}], title=\"{t['title']}\", summary=\"{t['summary']}\""
        for t in TRIALS
    )
    return (
        f"DRUG FACT — drug under analysis: {DRUG}\n"
        f"DRUG FACT — FDA-approved indications of this drug: {approved_line}\n\n"
        f"Repurposing candidate (the broad indication under investigation): {CANDIDATE}\n\n"
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
    per_trial_pass = {t["nct"]: 0 for t in TRIALS}
    for verdicts in results:
        for t in TRIALS:
            if verdicts.get(t["nct"]) == t["expected"]:
                per_trial_pass[t["nct"]] += 1
    for t in TRIALS:
        p = per_trial_pass[t["nct"]]
        flag = "OK " if p == RUNS else "!! "
        print(f"{flag}{t['nct']} (expect {t['expected']}): {p}/{RUNS}  — {t['title'][:50]}")
    total = sum(per_trial_pass.values())
    print(f"\nTOTAL: {total}/{len(TRIALS) * RUNS}")


if __name__ == "__main__":
    asyncio.run(main())
