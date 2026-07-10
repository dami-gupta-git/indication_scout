"""Standalone harness: does an ISOLATED call judge DRUG-SPECIFIC literature strength correctly —
not inflating to "strong" when the strong RCTs are for OTHER drugs in the same class?

The bug (snapshot semaglutide_2026-06-14_19-41-15.md, Parkinson): synthesize set
strength="strong" while its own prose said "no direct clinical evidence for semaglutide in
Parkinson's disease" — the strong RCTs are lixisenatide / exenatide / NLY01, and the one
semaglutide abstract is for depression. Strength must grade THIS DRUG's evidence; class-level
(other-drug) evidence is surfaced as evidence_basis="class_level", never as drug strength.

This harness pulls the REAL abstracts (by PMID) from the pgvector pubmed_abstracts table — the
same text synthesize saw — and runs the isolated call. Crux cases:
  - Parkinson (the bug): class-level GLP-1 RCTs + 1 off-topic semaglutide(depression) abstract
    → evidence_basis="class_level", strength NOT "strong".
  - T1DM: two genuine semaglutide RCTs → drug_specific, strength strong/moderate.
  - NASH: semaglutide RCTs, mixed fibrosis → drug_specific, direction mixed.

Run N times per case to catch drift. Gate before any wiring.

Run: .venv/bin/python tests/harness_tests/literature_strength_harness.py [model]
"""

import asyncio
import json
import sys
from collections import Counter

from sqlalchemy import create_engine, text

from anthropic import AsyncAnthropic

from indication_scout.config import get_settings

settings = get_settings()
client = AsyncAnthropic(api_key=settings.anthropic_api_key)
engine = create_engine(settings.database_url)
MODEL = sys.argv[1] if len(sys.argv) > 1 else "claude-sonnet-4-6"
RUNS_PER_CASE = 5

_BASIS = ("drug_specific", "class_level", "none")
_STRENGTH = ("strong", "moderate", "weak", "none")

# Mirrors the planned _LITERATURE_STRENGTH_PROMPT. Strength/direction grade THIS DRUG's evidence
# only; a different drug in the same class is class-level context, NOT drug strength.
PROMPT = """You are a biomedical evidence analyst. Grade the literature evidence for repurposing \
ONE drug to treat ONE disease, using ONLY the abstracts below.

Drug: {drug}
Disease: {disease}

CRITICAL — strength and direction grade evidence for THIS EXACT DRUG only:
- An abstract about a DIFFERENT drug — even one in the same mechanistic class (e.g. another GLP-1
  receptor agonist) — is NOT direct evidence for this drug. It is class-level context.
- An abstract about this drug but a DIFFERENT disease is NOT relevant evidence for this pair.
- THERAPEUTIC INTENT: evidence that the drug was studied IN patients who HAVE this disease but
  FOR a different condition (a comorbidity, smoking cessation, weight loss, etc.) is NOT evidence
  that the drug TREATS this disease. It does not support the repurposing hypothesis. Set
  evidence_basis="none" in that case (there is no relevant evidence for treating this disease),
  even though the abstracts are about this drug and mention this disease.
- evidence_basis:
  - "drug_specific": at least one abstract reports clinical/preclinical evidence for THIS drug
    used to TREAT THIS disease.
  - "class_level": the disease-relevant evidence is for OTHER drugs in the class; there is no
    direct evidence for this drug in this disease.
  - "none": no relevant evidence for treating this disease with this drug — neither drug-specific
    nor class-level, OR the only this-drug evidence is for a different condition in this
    population (the therapeutic-intent case above).
- strength grades DRUG-SPECIFIC evidence quantity/quality only:
  - "strong": multiple drug-specific clinical studies (RCTs, large cohorts) for THIS drug in THIS
    disease. NEVER "strong" when evidence_basis is "class_level" or "none".
  - "moderate": small drug-specific clinical studies, case series, or strong drug-specific
    preclinical data.
  - "weak": drug-specific case reports only, or drug-specific in-vitro/animal data only.
  - "none": no drug-specific evidence (set this whenever evidence_basis != "drug_specific").
- direction (of the drug-specific evidence): "supports" | "contradicts" | "mixed" | "none". When
  evidence_basis != "drug_specific", direction is "none".
- is_observational: true if the relevant drug-specific clinical evidence is exclusively
  observational; false if at least one drug-specific RCT/controlled trial; null if no relevant
  drug-specific clinical evidence.

Abstracts:
{abstracts}

Respond with ONLY a JSON object:
{{"evidence_basis": "drug_specific"|"class_level"|"none", \
"strength": "strong"|"moderate"|"weak"|"none", \
"direction": "supports"|"contradicts"|"mixed"|"none", \
"is_observational": true|false|null, "reason": "<one short sentence>"}}"""


def fetch_abstracts(pmids):
    with engine.connect() as c:
        rows = c.execute(
            text(
                "SELECT pmid, title, abstract FROM pubmed_abstracts WHERE pmid = ANY(:p)"
            ),
            {"p": pmids},
        ).fetchall()
    by_pmid = {r[0]: (r[1], r[2]) for r in rows}
    # Preserve the given order; skip any missing.
    return [
        f"PMID: {p}\nTitle: {by_pmid[p][0]}\nAbstract: {by_pmid[p][1]}"
        for p in pmids
        if p in by_pmid
    ]


# (name, drug, disease, pmids, expected_basis, strength_must_not_be).
CASES = [
    (
        "Parkinson (the bug): class-level GLP-1 RCTs + off-topic semaglutide",
        "semaglutide",
        "Parkinson Disease",
        ["38598572", "23728174", "28781108", "38101901", "41218611"],
        "class_level",
        {"strong"},  # must NOT be strong — no drug-specific Parkinson evidence
    ),
    (
        "T1DM: two genuine semaglutide RCTs",
        "semaglutide",
        "Type 1 Diabetes Mellitus",
        ["40550013", "41144928"],
        "drug_specific",
        set(),
    ),
    (
        "NASH: semaglutide RCTs, mixed fibrosis",
        "semaglutide",
        "Non-alcoholic Steatohepatitis",
        ["40305708", "33185364", "36934740"],
        "drug_specific",
        set(),
    ),
    (
        # THERAPEUTIC INTENT: bupropion RCTs ARE in schizophrenia patients, but FOR smoking
        # cessation / weight — not treating schizophrenia. Must be evidence_basis="none", NOT
        # drug_specific/supports (the prose itself says "does not support it as a direct
        # antipsychotic treatment").
        "Schizophrenia (intent trap): bupropion in schizophrenia patients for smoking/weight",
        "bupropion",
        "Schizophrenia",
        ["12079730", "11694208", "15876899", "34735098", "17632223"],
        "none",
        {"strong", "moderate"},  # no treats-schizophrenia evidence → not strong/moderate
    ),
    # ---- other drugs ----
    (
        "gefitinib x CRC: real RCTs showing gefitinib inactive (drug-specific, contradicts)",
        "gefitinib",
        "Colorectal Cancer",
        ["16361624", "16062074", "18667394", "20204674"],
        "drug_specific",
        set(),
    ),
    (
        "sildenafil x Raynaud: real sildenafil RCTs (drug-specific)",
        "sildenafil",
        "Raynaud Disease",
        ["21360507", "28281457", "30383134", "23443723"],
        "drug_specific",
        set(),
    ),
    # ---- cross-drug class-level TRAPS: real abstracts of ANOTHER same-class drug, queried
    #      under a sibling drug name. The evidence is class-level for the queried drug, never
    #      drug-specific — must NOT be "strong".
    (
        "erlotinib x CRC (TRAP): fed gefitinib RCTs — class-level EGFR-TKI, not erlotinib",
        "erlotinib",
        "Colorectal Cancer",
        ["16361624", "16062074", "18667394", "20204674"],
        "class_level",
        {"strong"},
    ),
    (
        "tadalafil x Raynaud (TRAP): fed sildenafil RCTs — class-level PDE5i, not tadalafil",
        "tadalafil",
        "Raynaud Disease",
        ["21360507", "28281457", "30383134", "23443723"],
        "class_level",
        {"strong"},
    ),
]


async def judge(drug, disease, abstracts):
    resp = await client.messages.create(
        model=MODEL,
        max_tokens=400,
        messages=[
            {
                "role": "user",
                "content": PROMPT.format(
                    drug=drug, disease=disease, abstracts="\n\n".join(abstracts)
                ),
            }
        ],
    )
    text_out = resp.content[0].text.strip()
    if text_out.startswith("```"):
        text_out = text_out.split("```")[1].lstrip("json").strip()
    try:
        d = json.loads(text_out)
        return d.get("evidence_basis", "PARSE_FAIL"), d.get("strength", "PARSE_FAIL")
    except json.JSONDecodeError:
        return "PARSE_FAIL", "PARSE_FAIL"


async def main():
    print(f"=== model: {MODEL} ===")
    all_pass = True
    for name, drug, disease, pmids, exp_basis, bad_strength in CASES:
        abstracts = fetch_abstracts(pmids)
        if len(abstracts) != len(pmids):
            print(f"[SKIP] {name}: only {len(abstracts)}/{len(pmids)} abstracts in DB")
            all_pass = False
            continue
        results = await asyncio.gather(
            *(judge(drug, disease, abstracts) for _ in range(RUNS_PER_CASE))
        )
        basis_counts = Counter(b for b, _ in results)
        strength_counts = Counter(s for _, s in results)
        n_basis_ok = sum(b == exp_basis for b, _ in results)
        n_strength_ok = sum(s not in bad_strength for _, s in results)
        ok = n_basis_ok == RUNS_PER_CASE and n_strength_ok == RUNS_PER_CASE
        if not ok:
            all_pass = False
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        print(
            f"        basis {n_basis_ok}/{RUNS_PER_CASE} (exp={exp_basis}, "
            f"got={dict(basis_counts)})"
        )
        if bad_strength:
            print(
                f"        strength {n_strength_ok}/{RUNS_PER_CASE} not-in {bad_strength} "
                f"(got={dict(strength_counts)})"
            )
        else:
            print(f"        strength got={dict(strength_counts)}")
    print("=== ALL PASS ===" if all_pass else "=== SOME FAILED ===")


if __name__ == "__main__":
    asyncio.run(main())
