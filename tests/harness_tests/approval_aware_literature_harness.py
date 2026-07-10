"""Harness: APPROVAL-AWARE literature strength.

Tests the planned extension to judge_literature_strength — given the drug's APPROVED indications,
papers studying an APPROVED sub-indication of a broad candidate must NOT count toward the
candidate's strength (they are already-approved evidence, not repurposing evidence for the broader
term). The literature analogue of the trial relevance gate dropping approved-sub-indication trials.

The case that motivated it (bupropion x "mood disorder"): "mood disorder" is a broad umbrella over
APPROVED sub-indications MDD and SAD. A MeSH-ancestor search pulls in MDD/SAD papers; if they count
toward strength, the broad candidate looks better-evidenced than its genuinely-broader (e.g.
bipolar) evidence warrants.

What we assert per case:
  - approved_basis: when the ONLY relevant drug-specific abstracts are about APPROVED
    sub-indications, evidence_basis must be "approved" (new value) — NOT drug_specific — and
    strength must drop out of {strong, moderate}.
  - control (no approved list / genuinely-broader evidence): unchanged drug_specific grading.

Real abstracts pulled by PMID from the pgvector pubmed_abstracts table (same text the pipeline
sees). Run N times to catch drift; gate before wiring.

Run: .venv/bin/python tests/harness_tests/approval_aware_literature_harness.py [model]
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

_BASIS = ("drug_specific", "approved", "class_level", "none")
_STRENGTH = ("strong", "moderate", "weak", "none")

# Planned approval-aware prompt: the existing drug-specific rubric PLUS an approved-indication
# exclusion. Papers about an APPROVED sub-indication of the candidate are already-approved
# evidence (evidence_basis="approved"), not repurposing evidence for the broader candidate.
PROMPT = """You are a biomedical evidence analyst. Grade the literature evidence for repurposing \
ONE drug to treat ONE disease, using ONLY the abstracts below.

Drug: {drug}
Disease: {disease}
Drug's FDA-approved indications: {approved_indications}

CRITICAL — strength/direction grade evidence for THIS EXACT DRUG and the repurposing of it for \
THIS candidate disease:
- An abstract about a DIFFERENT drug (even same mechanistic class) is class-level context, NOT \
direct evidence for this drug.
- An abstract about this drug but a DIFFERENT, unrelated disease is not relevant evidence.
- APPROVED-INDICATION EXCLUSION: the candidate disease may be a BROAD term that includes one of \
the drug's APPROVED indications listed above (e.g. candidate "mood disorder" includes approved \
"major depressive disorder"). An abstract studying THIS DRUG for an APPROVED sub-indication is \
ALREADY-APPROVED evidence, NOT evidence for repurposing the drug to the broader candidate — do \
NOT count it toward strength/direction. Only abstracts about the candidate's NON-approved scope \
(e.g. bipolar disorder, dysthymia for "mood disorder") count as repurposing evidence. \
SEVERITY/STAGE QUALIFIER: ignore a severity/stage/biomarker qualifier on the approved indication — \
if the approval is "X with <severity/stage>" (e.g. "MASH with fibrosis"), abstracts on the bare \
disease X or near-synonyms (e.g. "NASH"/"steatohepatitis") ARE the approved sub-indication and are \
EXCLUDED; only the broad NAFLD/simple-steatosis spectrum counts. A severity grade is NOT a minority \
biomarker.

evidence_basis:
- "drug_specific": >=1 abstract reports evidence for THIS drug in the candidate's NON-approved \
scope.
- "approved": the only relevant this-drug abstracts study an APPROVED sub-indication — there is no \
repurposing evidence for the broader candidate's uncovered scope.
- "class_level": the only relevant evidence is for another drug in the class.
- "none": no relevant evidence.

When evidence_basis is "approved", "class_level", or "none", strength must be "weak" or "none" \
(there is no drug-specific repurposing evidence to grade as strong/moderate) and direction "none".

Abstracts:
{abstracts}

Respond with ONLY a JSON object:
{{"evidence_basis": "drug_specific"|"approved"|"class_level"|"none", \
"strength": "strong"|"moderate"|"weak"|"none", \
"direction": "supports"|"contradicts"|"mixed"|"none", \
"reason": "<one short sentence>"}}"""


def fetch_abstracts(pmids):
    with engine.connect() as c:
        rows = c.execute(
            text("SELECT pmid, title, abstract FROM pubmed_abstracts WHERE pmid = ANY(:p)"),
            {"p": pmids},
        ).fetchall()
    by_pmid = {r[0]: (r[1], r[2]) for r in rows}
    return [
        f"PMID: {p}\nTitle: {by_pmid[p][0]}\nAbstract: {by_pmid[p][1]}"
        for p in pmids
        if p in by_pmid
    ]


# (name, drug, disease, approved_indications, pmids, expected_basis, strength_must_not_be)
# PMIDs are REAL ids present in pgvector (pulled from live sildenafil/bupropion runs).
CASES = [
    # ---- sildenafil x hypertension: the PAH contamination case ----
    (
        # 4 of these 5 hypertension-query papers are PULMONARY hypertension (the APPROVED PAH
        # indication); only the lead-induced one is systemic. With approved={PAH}, the relevant
        # evidence is approved → basis "approved", NOT a drug_specific moderate (today's bug).
        "sildenafil x hypertension, approved={PAH}: PAH papers => 'approved', not moderate",
        "sildenafil",
        "Hypertension",
        ["pulmonary arterial hypertension", "erectile dysfunction"],
        ["33983650", "33226665", "32634428", "39487423"],
        "approved",
        {"strong", "moderate"},
    ),
    (
        # NUANCE: even with NO approved list, these papers are about PULMONARY hypertension, a
        # DISTINCT disease from the systemic-hypertension candidate — so the model marks them
        # not-relevant on disease grounds alone (basis "approved"/"none", never drug_specific
        # moderate). Documents that PAH-vs-systemic is caught by disease mismatch too, not only
        # the approved-list rule. Either approved or none is acceptable here; must NOT be moderate.
        "sildenafil x hypertension, approved=(none): PAH papers are still not systemic-HTN evidence",
        "sildenafil",
        "Hypertension",
        [],
        ["33983650", "33226665", "32634428", "39487423"],
        {"approved", "none"},  # either acceptable; binding assert is NOT moderate/strong
        {"strong", "moderate"},
    ),
    # ---- bupropion x mood disorder: the MDD/SAD contamination case ----
    (
        # MDD-focused papers; MDD is APPROVED, so for the broad "mood disorder" candidate this is
        # already-approved evidence, not repurposing of the broader term.
        "bupropion x mood disorder, approved={MDD,SAD}: MDD papers => 'approved', not strong",
        "bupropion",
        "Mood Disorder",
        ["major depressive disorder", "seasonal affective disorder", "smoking cessation"],
        ["31301615", "25124683"],
        "approved",
        {"strong", "moderate"},
    ),
    (
        # CONTROL: the bipolar paper is the candidate's genuinely NON-approved scope → drug_specific.
        "bupropion x mood disorder, approved={MDD,SAD}: bipolar paper => drug_specific (control)",
        "bupropion",
        "Mood Disorder",
        ["major depressive disorder", "seasonal affective disorder", "smoking cessation"],
        ["2856918"],
        "drug_specific",
        set(),
    ),
    (
        # CONTROL: ADHD is NOT an approved bupropion indication → exclusion must NOT fire.
        "bupropion x ADHD, approved={MDD,SAD,smoking}: no overlap => drug_specific (control)",
        "bupropion",
        "Attention Deficit Hyperactivity Disorder",
        ["major depressive disorder", "seasonal affective disorder", "smoking cessation"],
        ["30097390", "24259638", "26601963"],
        "drug_specific",
        set(),
    ),
    # ---- held-out: semaglutide x NAFLD, approved={MASH} ----
    (
        # NAFLD is the broad umbrella; MASH/NASH is the APPROVED subtype. These papers center on
        # NAFLD/NASH semaglutide evidence. With approved={MASH}, the NASH-specific evidence is
        # already-approved; only genuinely-broader (simple-steatosis NAFLD) evidence would be
        # repurposing. Accept 'approved' (papers are MASH-scoped) — must NOT be strong.
        "semaglutide x NAFLD, approved={MASH}: NASH papers => 'approved', not strong",
        "semaglutide",
        "Non-alcoholic Fatty Liver Disease",
        ["metabolic dysfunction-associated steatohepatitis (MASH)", "type 2 diabetes mellitus", "obesity"],
        ["37717295", "37899788", "37355043"],
        "approved",
        {"strong"},
    ),
    # ---- held-out CONTROL: semaglutide x T1DM, approved={T2DM} — sibling, NOT covered ----
    (
        # T1DM is a SIBLING of approved T2DM, not a sub-indication of it. The approved-exclusion
        # must NOT fire — these are genuine T1DM repurposing papers → drug_specific.
        "semaglutide x T1DM, approved={T2DM,obesity}: sibling, exclusion must NOT fire (control)",
        "semaglutide",
        "Type 1 Diabetes Mellitus",
        ["type 2 diabetes mellitus", "obesity", "chronic weight management"],
        ["40550013", "38444317", "39745353"],
        "drug_specific",
        set(),
    ),
    # ---- held-out: empagliflozin x kidney disease, approved={CKD} ----
    (
        # "Kidney disease" is the broad umbrella; CHRONIC kidney disease IS an approved
        # empagliflozin indication. These papers are all empagliflozin-in-CKD → already-approved
        # evidence for the broad candidate, not repurposing. With approved={CKD} → 'approved'.
        "empagliflozin x kidney disease, approved={CKD,T2DM}: CKD papers => 'approved', not strong",
        "empagliflozin",
        "Kidney Disease",
        ["chronic kidney disease", "type 2 diabetes mellitus", "heart failure"],
        ["36331190", "39453837", "38061371"],
        "approved",
        {"strong"},
    ),
    (
        # CONTROL: SAME CKD papers, NO approved list. Without the approved hint, empagliflozin-in-CKD
        # IS genuine drug-specific evidence for the kidney-disease pair (CKD is the candidate scope).
        # Proves the exclusion is DRIVEN by the approved list, not invented.
        "empagliflozin x kidney disease, approved=(none): CKD papers => drug_specific (control)",
        "empagliflozin",
        "Kidney Disease",
        [],
        ["36331190", "39453837", "38061371"],
        "drug_specific",
        set(),
    ),
    # ---- held-out CONTROL: metformin x PCOS — no approval overlap at all ----
    (
        # PCOS is NOT a metformin-approved indication and is unrelated to approved T2DM. The
        # approved list must NOT trigger any exclusion → genuine drug_specific PCOS evidence.
        "metformin x PCOS, approved={T2DM}: unrelated to approval => drug_specific (control)",
        "metformin",
        "Polycystic Ovary Syndrome",
        ["type 2 diabetes mellitus"],
        ["12196859", "17940431", "34740275"],
        "drug_specific",
        set(),
    ),
    # ---- held-out: semaglutide x cardiovascular disease, approved={CV risk reduction} ----
    (
        # Semaglutide IS approved to reduce CV risk. "Cardiovascular disease" as a broad candidate
        # overlaps that approval; these CV-outcomes papers are the approved-CV evidence, not
        # repurposing for the broader CVD term. With the CV approval listed → 'approved', not strong.
        "semaglutide x CVD, approved={CV risk reduction}: CV-outcome papers => 'approved', not strong",
        "semaglutide",
        "Cardiovascular Disease",
        ["cardiovascular risk reduction", "type 2 diabetes mellitus", "obesity"],
        ["27633186", "37952131", "40162642"],
        "approved",
        {"strong"},
    ),
]


async def _one_call(content):
    # 900 tokens: some cases reason step-by-step before the JSON and were truncating at 400
    # (stop_reason=max_tokens → no closing brace → parse fail).
    resp = await client.messages.create(
        model=MODEL,
        max_tokens=900,
        messages=[{"role": "user", "content": content}],
    )
    return resp.content[0].text.strip() if resp.content else ""


async def judge(drug, disease, approved, abstracts, _retries=3):
    content = PROMPT.format(
        drug=drug,
        disease=disease,
        approved_indications=", ".join(approved) if approved else "(none)",
        abstracts="\n\n".join(abstracts),
    )
    # Retry on empty/unparseable output — the batched gather occasionally returns empty under
    # load (529/overloaded), which is transport flakiness, not a prompt failure.
    txt = ""
    for attempt in range(_retries):
        try:
            txt = await _one_call(content)
        except Exception:
            txt = ""
        if txt:
            break
        await asyncio.sleep(1.5 * (attempt + 1))
    if txt.startswith("```"):
        txt = txt.split("```")[1].lstrip("json").strip()
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        # Tolerant fallback: take the LAST balanced {...} block (model may emit prose first).
        end = txt.rfind("}")
        if end != -1:
            depth = 0
            for i in range(end, -1, -1):
                if txt[i] == "}":
                    depth += 1
                elif txt[i] == "{":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(txt[i : end + 1])
                        except json.JSONDecodeError:
                            break
        return {}


async def main():
    print(f"=== model: {MODEL}  ({RUNS_PER_CASE} runs/case) ===\n")
    n_pass = 0
    for name, drug, disease, approved, pmids, exp_basis, bad_strength in CASES:
        abstracts = fetch_abstracts(pmids)
        if len(abstracts) < len(pmids):
            print(f"[SKIP] {name}\n        missing abstracts: got {len(abstracts)}/{len(pmids)}")
            continue
        runs = await asyncio.gather(
            *(judge(drug, disease, approved, abstracts) for _ in range(RUNS_PER_CASE))
        )
        # exp_basis may be a single str or a set of acceptable values.
        accept = {exp_basis} if isinstance(exp_basis, str) else set(exp_basis)
        basis_votes = Counter(r.get("evidence_basis", "PARSE_FAIL") for r in runs)
        bad_hits = sum(1 for r in runs if r.get("strength") in bad_strength)
        basis_ok = basis_votes.most_common(1)[0][0] in accept
        ok = basis_ok and bad_hits == 0
        n_pass += ok
        verdict = "PASS" if ok else "FAIL"
        print(f"[{verdict}] {name}")
        print(f"        basis={dict(basis_votes)} (expected {exp_basis}); "
              f"strength-in-{sorted(bad_strength) or '∅'}: {bad_hits}/{RUNS_PER_CASE}")
        if not ok:
            print(f"        sample: {json.dumps(runs[0])[:240]}")
    print(f"\n=== {n_pass}/{len(CASES)} cases pass ===")


if __name__ == "__main__":
    asyncio.run(main())
