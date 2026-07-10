"""B3 gate: can the synthesize/grading call ALSO emit an accurate `is_animal_only` flag from the
abstracts it already reads — no new PubMed/MeSH fetch?

Context (humira run, 2026-07-10): adalimumab×asthma ranked #1 despite its only supporting study
being a murine OVA model (PMID 24882395); the ranking critic couldn't demote it because the
FACT/ranking string has no animal-vs-human descriptor. Fix option B3: since synthesize already reads
the abstract text in-context, have it additionally return `is_animal_only`. This harness tests
whether that added field is RELIABLE before we wire it into the model + retrieval path.

is_animal_only semantics under test:
  - true  : ALL relevant drug-specific studies are animal/in-vitro (no human clinical/observational).
  - false : at least one relevant HUMAN study (RCT, cohort, case series, observational).
  - null  : no relevant drug-specific evidence to grade (don't fabricate).

Pulls REAL abstracts from the pgvector pubmed_abstracts table (same text synthesize saw). N runs per
case to catch drift. If this passes, B3 is viable and the plan's animal-only item is a one-field add
to synthesize.txt + EvidenceSummary. If it drifts, fall back to B1 (MeSH fetch).

Run: .venv/bin/python tests/harness_tests/animal_only_synthesis_harness.py [model]
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

# Extends the literature_strength grading prompt with an is_animal_only field. Kept minimal and
# self-contained so this gate stands alone; if it passes, the wording folds into synthesize.txt.
PROMPT = """You are a biomedical evidence analyst. Grade the literature evidence for repurposing \
ONE drug to treat ONE disease, using ONLY the abstracts below.

Drug: {drug}
Disease: {disease}

Judge the RELEVANT drug-specific evidence (this exact drug, treating this disease). Then classify
whether that relevant evidence is animal/in-vitro only:
- is_animal_only:
  - true  : every relevant study for THIS drug in THIS disease is a non-human model — animal
    (mouse/rat/etc.) or in-vitro/cell studies — with NO human clinical or observational data.
  - false : at least one relevant study for THIS drug in THIS disease is in humans (RCT, cohort,
    case series, or other observational human data).
  - null  : there is no relevant drug-specific evidence to grade.
Do not infer humans from the disease name alone — a mouse MODEL of a human disease is still animal.

Abstracts:
{abstracts}

Respond with ONLY a JSON object:
{{"is_animal_only": true|false|null, "reason": "<one short sentence naming the study species/type>"}}"""


def fetch_abstracts(pmids):
    with engine.connect() as c:
        rows = c.execute(
            text(
                "SELECT pmid, title, abstract FROM pubmed_abstracts WHERE pmid = ANY(:p)"
            ),
            {"p": pmids},
        ).fetchall()
    by_pmid = {r[0]: (r[1], r[2]) for r in rows}
    return [
        f"PMID: {p}\nTitle: {by_pmid[p][0]}\nAbstract: {by_pmid[p][1]}"
        for p in pmids
        if p in by_pmid
    ]


# (name, drug, disease, pmids, expected_is_animal_only).
CASES = [
    (
        "asthma (the bug): sole study is a murine OVA anti-TNF model → animal-only",
        "adalimumab",
        "Asthma",
        ["24882395"],
        True,
    ),
    (
        "T1DM: human semaglutide RCTs → not animal-only",
        "semaglutide",
        "Type 1 Diabetes Mellitus",
        ["40550013", "41144928"],
        False,
    ),
    (
        "Raynaud: human sildenafil RCTs → not animal-only",
        "sildenafil",
        "Raynaud Disease",
        ["21360507", "28281457"],
        False,
    ),
    (
        "CRC: human gefitinib Phase II trial → not animal-only",
        "gefitinib",
        "Colorectal Cancer",
        ["16361624"],
        False,
    ),
    (
        # Asthma really has TWO murine adalimumab studies (24882395 + 27262379 OVA mouse models),
        # no human data → still animal-only. Guards against a second animal abstract flipping it.
        "asthma (two murine studies): both OVA mouse models → animal-only",
        "adalimumab",
        "Asthma",
        ["24882395", "27262379"],
        True,
    ),
    (
        # MIXED trap (real same-drug/same-disease pair): a murine adalimumab-PsA model (26445328)
        # + a human adalimumab-PsA trial (16200601). One relevant HUMAN study must flip it to
        # FALSE — the mouse abstract must not drag it to animal-only.
        "MIXED trap: murine PsA model + human adalimumab PsA trial → NOT animal-only",
        "adalimumab",
        "Psoriatic Arthritis",
        ["26445328", "16200601"],
        False,
    ),
    # ---- 5 additional discriminators ----
    (
        # RELEVANCE (wrong-drug) guard: 39252097 is an INFLIXIMAB case report, not adalimumab. No
        # relevant adalimumab evidence → is_animal_only must be null (not fabricated true/false).
        "wrong-drug guard: infliximab T1D case report queried as adalimumab → null",
        "adalimumab",
        "Type 1 Diabetes Mellitus",
        ["39252097"],
        None,
    ),
    (
        # Human retrospective COHORT → false.
        "human cohort: semaglutide in renal-failure patients → NOT animal-only",
        "semaglutide",
        "Renal Failure",
        ["39025300"],
        False,
    ),
    (
        # RELEVANCE (wrong-drug/disease) guard: these are OTHER GLP-1 drugs in type-2 mouse models,
        # not semaglutide in T1D. No relevant drug-specific evidence → null, NOT animal-only=true.
        "wrong-pair guard: other-GLP-1 mouse studies queried as semaglutide/T1D → null",
        "semaglutide",
        "Type 1 Diabetes Mellitus",
        ["26724516", "24534256"],
        None,
    ),
    (
        # RELEVANCE guard: a human RCT for a DIFFERENT drug+disease (sildenafil-Raynaud) is NOT
        # relevant adalimumab-asthma evidence, so it cannot make asthma "not animal-only". The only
        # RELEVANT adalimumab-asthma study is murine → true. (Was the mis-built case earlier;
        # now asserted with the correct expectation.)
        "relevance guard: murine adalimumab-asthma + off-pair human RCT → still animal-only",
        "adalimumab",
        "Asthma",
        ["27262379", "21360507"],
        True,
    ),
    (
        # Human PsA trial ALONE (no animal abstract) → false. Baseline that a pure-human single
        # study reads false.
        "human only: adalimumab PsA trial alone → NOT animal-only",
        "adalimumab",
        "Psoriatic Arthritis",
        ["16200601"],
        False,
    ),
]


async def judge(drug, disease, abstracts):
    resp = await client.messages.create(
        model=MODEL,
        max_tokens=300,
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
        return d.get("is_animal_only", "PARSE_FAIL"), d.get("reason", "")
    except json.JSONDecodeError:
        return "PARSE_FAIL", ""


async def main():
    print(f"=== model: {MODEL}  |  {RUNS_PER_CASE} runs/case ===\n")
    all_pass = True
    for name, drug, disease, pmids, expected in CASES:
        abstracts = fetch_abstracts(pmids)
        if len(abstracts) != len(pmids):
            print(f"[SKIP] {name}: only {len(abstracts)}/{len(pmids)} abstracts in DB\n")
            all_pass = False
            continue
        results = await asyncio.gather(
            *(judge(drug, disease, abstracts) for _ in range(RUNS_PER_CASE))
        )
        vals = [v for v, _ in results]
        counts = Counter(vals)
        n_ok = sum(v == expected for v in vals)
        ok = n_ok == RUNS_PER_CASE
        all_pass = all_pass and ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        print(f"        is_animal_only {n_ok}/{RUNS_PER_CASE} (exp={expected}, got={dict(counts)})")
        if not ok:
            for v, r in results:
                if v != expected:
                    print(f"          miss: {v} — {r}")
        print()
    print("=== ALL PASS ===" if all_pass else "=== SOME FAILED ===")


if __name__ == "__main__":
    asyncio.run(main())
