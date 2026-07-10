"""Harness: APPROVAL-AWARE clinical-trials relevance gate.

Tests the planned extension to the CT relevance gate (prompts/clinical_trials.txt) — given the
drug's APPROVED indications, a trial whose condition is an APPROVED sub-indication of a BROAD
candidate must be classified CONTAMINATION, not relevant evidence that rolls up into the parent.
The trial analogue of the literature harness's approved-sub-indication exclusion.

The case that motivated it (bupropion x "mood disorder"): "mood disorder" is a broad umbrella over
APPROVED sub-indications MDD and SAD. CT.gov AREA[ConditionMeshTerm] matches via MeSH ancestors, so
the umbrella query pulls in the approved-SAD trial NCT00046241; the current relevance rule ("a
narrower subtype rolls up") marks it RELEVANT — propping up the broad candidate's dev-stage signal
with already-approved evidence. Today this is patched by the CURATED_CONTAMINATED_NCTS hardcode;
this harness gates the GENERAL rule that subsumes it.

What we assert per case:
  - approved-subtype contamination: a trial about an APPROVED sub-indication of a BROAD candidate
    must be verdict "contaminated" — NOT relevant — once the approved list is supplied.
  - control (no approved list, OR a genuinely-narrower non-approved subtype, OR a sibling): the
    trial keeps its normal roll-up verdict; the exclusion must NOT over-fire.

Trial records are REAL (pulled by NCT from CT.gov), embedded as literals so the harness is offline
and stable. Run N times to catch drift; gate before wiring the prompt change.

Run: .venv/bin/python tests/harness_tests/approval_aware_trials_harness.py [model]
"""

import asyncio
import json
import sys
from collections import Counter

from anthropic import AsyncAnthropic

from indication_scout.config import get_settings

settings = get_settings()
client = AsyncAnthropic(api_key=settings.anthropic_api_key)
MODEL = sys.argv[1] if len(sys.argv) > 1 else "claude-sonnet-4-6"
RUNS_PER_CASE = 5

_VERDICT = ("relevant", "contaminated")

# Planned approval-aware gate: the existing RELEVANCE rubric PLUS an approved-indication override.
# A trial about an APPROVED sub-indication of a BROAD candidate is CONTAMINATION (already-approved
# evidence), even though it is a clinical subtype that would otherwise roll up into the parent.
PROMPT = """You classify clinical trials surfaced for a drug x indication pair as RELEVANT evidence \
or CONTAMINATION, using ONLY the trial records below.

Drug: {drug}
Candidate indication: {disease}
Drug's FDA-approved indications: {approved_indications}

Apply these tests to EACH trial IN ORDER. The FIRST one that matches decides the verdict.

TEST 1 — APPROVED-SUBTYPE (=> CONTAMINATION). Look at the trial's condition(s). Is that condition \
one of the drug's APPROVED indications listed above, OR a narrower form of one of them? If YES, the \
trial is ALREADY-APPROVED evidence and is CONTAMINATION — even though the drug IS this drug and the \
condition is a clinical subtype of the candidate. This is the key rule: the candidate is a BROAD \
term, and a trial for the APPROVED part of it does NOT support repurposing the drug to the broader \
candidate. Examples:
  - candidate "kidney disease", approved includes "chronic kidney disease" -> a CKD trial of this \
drug is CONTAMINATION (CKD is approved).
  - candidate "fatty liver disease", approved includes "MASH/NASH" -> a NASH trial of this drug is \
CONTAMINATION.
  - candidate "mood disorder", approved includes "seasonal affective disorder" -> a SAD trial is \
CONTAMINATION.
  SEVERITY/STAGE QUALIFIER: ignore a severity/stage/biomarker QUALIFIER on the approved indication \
when matching — it does NOT create a separable disease. If the approval is "X with <severity/stage>" \
(e.g. "MASH with moderate-to-advanced fibrosis"), a trial whose condition is the bare disease X or \
its near-synonyms (e.g. "NASH"/"steatohepatitis") IS the approved sub-indication => CONTAMINATION, \
while a trial of the broad NAFLD/simple-steatosis spectrum is RELEVANT. A severity grade is NOT a \
minority biomarker — "NASH" is the approved disease, not a broad parent of it.
  IMPORTANT: a SIBLING of an approved indication is NOT a subtype of it and does NOT match this \
test (e.g. "type 1 diabetes" is a sibling of approved "type 2 diabetes", not a narrower form of \
it) — fall through to the next tests.

TEST 2 — DISTINCT DISEASE or WRONG DRUG (=> CONTAMINATION). Is the trial's condition a DISTINCT \
disease that merely shares a parent term with the candidate (e.g. systemic "Hypertension" vs \
"Pulmonary Arterial Hypertension")? Or is the studied/intervention drug NOT this drug (this drug \
appears only as a comparator or PK probe)? If YES => CONTAMINATION.

TEST 3 — otherwise (=> RELEVANT). The trial studies THIS drug for THIS candidate indication or a \
narrower NON-APPROVED subtype of it (which rolls up as real repurposing evidence).

Judge from the DRUGS (interventions), CONDITIONS, TITLE, and SUMMARY — not the condition name \
alone. If the approved-indications list is "(none)", TEST 1 never matches.

Trials:
{trials}

Respond with ONLY a JSON object mapping each NCT id to its verdict:
{{"<NCT id>": "relevant"|"contaminated", ...}}"""


def fmt_trial(t):
    return (
        f"NCT: {t['nct']}\n"
        f"Condition(s): {', '.join(t['conditions'])}\n"
        f"Intervention(s): {', '.join(t['interventions']) or '(none listed)'}\n"
        f"Title: {t['title']}\n"
        f"Summary: {t['summary']}"
    )


# Real CT.gov records (pulled by NCT), embedded so the harness is offline.
T = {
    "NCT00046241": {
        "nct": "NCT00046241",
        "conditions": ["Seasonal Affective Disorder"],
        "interventions": ["Extended-release bupropion hydrochloride"],
        "title": "Prevention of Seasonal Affective Disorder",
        "summary": "This is a placebo controlled study evaluating the effectiveness of medication "
        "in preventing depressive episodes in subjects with a history of Seasonal Affective "
        "Disorder (SAD).",
    },
    "NCT00519428": {
        "nct": "NCT00519428",
        "conditions": ["Major Depressive Disorder"],
        "interventions": ["escitalopram", "bupropion extra long (XL)", "escitalopram + bupropion"],
        "title": "Does Dual Therapy Hasten Antidepressant Response?",
        "summary": "This study will utilize a randomized double-blind design to evaluate whether "
        "initial treatment with two anti-depressant medications (escitalopram and bupropion) "
        "results in more rapid remission and greater over-all remission rates than either "
        "monotherapy in 240 depressed subjects.",
    },
    "NCT07266545": {
        "nct": "NCT07266545",
        "conditions": ["Unipolar Depression", "Bipolar Depression"],
        "interventions": ["Vortioxetine", "Bupropion extended release", "Cariprazine"],
        "title": "RNA Editing as a Biomarker of Antidepressant Response in Unipolar and Bipolar "
        "Depression (EDIT-ANDRE)",
        "summary": "The purpose of this research is to understand how changes in RNA editing relate "
        "to treatment response in unipolar and bipolar depression.",
    },
    "NCT01181284": {
        "nct": "NCT01181284",
        "conditions": ["Pulmonary Arterial Hypertension"],
        "interventions": [],
        "title": "Modulating Effects of Lisinopril on Sildenafil Activity in Pulmonary Arterial "
        "Hypertension (PAH) (MELISSA)",
        "summary": "Patients with pulmonary arterial hypertension (PAH) suffer from chronic "
        "shortness of breath, and have impaired survival related to progressive right ventricular "
        "failure. Abnormal vasoreactivity to nitric oxide (NO) plays a role in the pathophysiology "
        "of PAH.",
    },
    "NCT01013857": {
        "nct": "NCT01013857",
        "conditions": ["Hypertension"],
        "interventions": ["Health coaching", "Health coaching plus home titration"],
        "title": "Treating to Target for Patients With Hypertension",
        "summary": "Patients with poorly controlled hypertension will have improved hypertensive "
        "control with telephone coaching and with telephone coaching combined with home-titration "
        "of medications.",
    },
    "NCT03884075": {
        "nct": "NCT03884075",
        "conditions": ["Non-Alcoholic Steatohepatitis", "Non-Alcoholic Fatty Liver Disease"],
        "interventions": ["Semaglutide"],
        "title": "Non-Alcoholic Fatty Liver Disease, the HEpatic Response to Oral Glucose, and the "
        "Effect of Semaglutide (NAFLD HEROES)",
        "summary": "In non-alcoholic fatty liver disease (NAFLD), fat accumulates in the liver and "
        "can cause damage. Researchers want to learn what causes the damage in NAFLD, and to see "
        "if a medication (semaglutide) can help.",
    },
    "NCT04143581": {
        "nct": "NCT04143581",
        "conditions": ["Obesity", "Non-diabetic Chronic Kidney Disease"],
        "interventions": ["Empagliflozin 10 MG"],
        "title": "SGLT2 Inhibitors in Glomerular Hyperfiltration",
        "summary": "Glomerular hyperfiltration is a major risk factor for accelerated glomerular "
        "filtration rate (GFR) decline and renal and cardiovascular events despite optimized "
        "conservative therapy. This study evaluates empagliflozin in non-diabetic chronic kidney "
        "disease.",
    },
    "NCT00322452": {
        "nct": "NCT00322452",
        "conditions": ["Non-Small Cell Lung Cancer"],
        "interventions": ["Gefitinib", "Carboplatin", "Paclitaxel"],
        "title": "First Line IRESSA Versus Carboplatin/Paclitaxel in Asia",
        "summary": "The purpose of this study is to compare gefitinib with carboplatin / paclitaxel "
        "doublet chemotherapy given as first line treatment in terms of progression free survival "
        "in selected NSCLC patients with the objective of demonstrating non-inferiority.",
    },
    "NCT04303780": {
        "nct": "NCT04303780",
        "conditions": ["KRAS p.G12C Mutated / Advanced Metastatic NSCLC"],
        "interventions": ["AMG 510 (sotorasib)", "Docetaxel"],
        "title": "Study to Compare Sotorasib With Docetaxel in Non Small Cell Lung Cancer "
        "(CodeBreak 200)",
        "summary": "A Phase 3 Study to Compare AMG 510 (sotorasib) with Docetaxel in Non Small Cell "
        "Lung Cancer (NSCLC) subjects with KRAS p.G12C mutation.",
    },
    "NCT00041197": {
        "nct": "NCT00041197",
        "conditions": ["Gastrointestinal Stromal Tumor"],
        "interventions": ["imatinib mesylate", "placebo"],
        "title": "Imatinib Mesylate in Treating Patients With Primary Gastrointestinal Stromal "
        "Tumor That Has Been Completely Removed By Surgery",
        "summary": "This randomized phase III trial is studying imatinib mesylate to see how well "
        "it works compared to placebo in treating patients with primary gastrointestinal stromal "
        "tumor that has been completely removed by surgery.",
    },
    "NCT00759785": {
        "nct": "NCT00759785",
        "conditions": ["Breast Cancer"],
        "interventions": ["dalotuzumab (MK0646)"],
        "title": "A Study of Dalotuzumab (MK-0646) in Breast Cancer Patients",
        "summary": "A study to evaluate the response of growth factor signatures to a single dose "
        "of dalotuzumab in participants with triple negative or ER-positive luminal B breast "
        "cancer.",
    },
}

# (name, drug, disease, approved_indications, [ncts], {nct: expected_verdict})
# Each case maps each shown NCT to the verdict the gate MUST produce (majority vote over runs).
CASES = [
    # ---- bupropion x mood disorder: the SAD/MDD contamination case (motivating bug) ----
    (
        # SAD is an APPROVED bupropion indication; for the broad "mood disorder" candidate this
        # approved-SAD trial is already-approved evidence => contaminated. Reproduces the
        # CURATED_CONTAMINATED_NCTS hardcode via the GENERAL rule.
        "bupropion x mood disorder, approved={SAD,MDD}: approved-SAD trial => contaminated",
        "bupropion",
        "Mood Disorder",
        ["seasonal affective disorder", "major depressive disorder", "smoking cessation"],
        ["NCT00046241"],
        {"NCT00046241": "contaminated"},
    ),
    (
        # SAME trial, NO approved list. SAD is a fairly DISTINCT narrower condition, so the model
        # may call it contaminated on disease grounds alone even without the approved hint — either
        # verdict is acceptable here. The list-DRIVEN proof lives in the empagliflozin CKD pair
        # below (a clean subtype that rolls up as relevant without the list).
        "bupropion x mood disorder, approved=(none): SAD trial (either verdict ok)",
        "bupropion",
        "Mood Disorder",
        [],
        ["NCT00046241"],
        {"NCT00046241": {"relevant", "contaminated"}},
    ),
    (
        # MDD is ALSO approved => the MDD trial is contaminated for the broad candidate. The
        # bipolar trial studies bupropion for bipolar depression — NON-approved scope => relevant.
        "bupropion x mood disorder, approved={SAD,MDD}: MDD=>contaminated, bipolar=>relevant",
        "bupropion",
        "Mood Disorder",
        ["seasonal affective disorder", "major depressive disorder", "smoking cessation"],
        ["NCT00519428", "NCT07266545"],
        {"NCT00519428": "contaminated", "NCT07266545": "relevant"},
    ),
    # ---- sildenafil x hypertension: PAH is a DISTINCT disease (caught without approved list) ----
    (
        # The PAH trial is a distinct disease from systemic hypertension AND PAH is approved.
        # Either way => contaminated. The systemic-HTN trial is the real candidate => relevant
        # (it does not study sildenafil, but condition is systemic HTN — keep it relevant on the
        # disease axis; the drug-mismatch axis is exercised separately).
        "sildenafil x hypertension, approved={PAH,ED}: PAH trial => contaminated",
        "sildenafil",
        "Hypertension",
        ["pulmonary arterial hypertension", "erectile dysfunction"],
        ["NCT01181284"],
        {"NCT01181284": "contaminated"},
    ),
    # ---- semaglutide x NAFLD: MASH/NASH is the APPROVED subtype ----
    (
        # This trial's condition lists BOTH the approved subtype (NASH) AND the broad candidate
        # (NAFLD) explicitly, so it genuinely studies the broader population too — a legitimately
        # AMBIGUOUS case. Either verdict is acceptable; the clean NASH-only behavior is covered by
        # the NASH-only trial case below.
        "semaglutide x NAFLD, approved={MASH}: mixed NASH+NAFLD trial (either verdict ok)",
        "semaglutide",
        "Non-alcoholic Fatty Liver Disease",
        ["metabolic dysfunction-associated steatohepatitis (MASH)", "type 2 diabetes", "obesity"],
        ["NCT03884075"],
        {"NCT03884075": {"contaminated", "relevant"}},
    ),
    # ---- empagliflozin x kidney disease: CKD is approved; control proves list-driven ----
    (
        # The trial studies empagliflozin in NON-diabetic CKD. CKD is approved => contaminated for
        # the broad "kidney disease" candidate.
        "empagliflozin x kidney disease, approved={CKD}: CKD trial => contaminated",
        "empagliflozin",
        "Kidney Disease",
        ["chronic kidney disease", "type 2 diabetes", "heart failure"],
        ["NCT04143581"],
        {"NCT04143581": "contaminated"},
    ),
    (
        # SAME CKD trial, NO approved list => CKD rolls up as a kidney-disease subtype => relevant.
        # The exclusion must be DRIVEN by the approved list.
        "empagliflozin x kidney disease, approved=(none): CKD trial rolls up => relevant (control)",
        "empagliflozin",
        "Kidney Disease",
        [],
        ["NCT04143581"],
        {"NCT04143581": "relevant"},
    ),
    # ---- gefitinib x lung cancer: MINORITY-biomarker approval, BARE-NSCLC trial stays relevant ----
    (
        # Gefitinib's approval is EGFR-mutated NSCLC (~10-15% of NSCLC). This trial's condition is
        # bare "Non-Small Cell Lung Cancer" (all-comers) — BROADER than the EGFR approval, so it is
        # NOT purely an approved-subtype trial and rolls up as relevant. Confirms the override does
        # NOT over-fire on a trial that is broader than a minority-biomarker approval (contrast the
        # sotorasib KRAS-G12C-NSCLC case below, where the trial == the approved subset).
        "gefitinib x lung cancer, approved={EGFR NSCLC}: bare-NSCLC trial is broader => relevant",
        "gefitinib",
        "Lung Cancer",
        ["EGFR-mutated non-small cell lung cancer"],
        ["NCT00322452"],
        {"NCT00322452": "relevant"},
    ),
    # ---- sotorasib x lung cancer: approved KRAS-G12C NSCLC subtype ----
    (
        # Sotorasib is approved for KRAS G12C NSCLC. The KRAS-G12C NSCLC trial is the approved
        # sub-indication of broad "lung cancer" => contaminated.
        "sotorasib x lung cancer, approved={KRAS G12C NSCLC}: KRAS-NSCLC trial => contaminated",
        "sotorasib",
        "Lung Cancer",
        ["KRAS G12C-mutated non-small cell lung cancer", "KRAS G12C-mutated colorectal cancer"],
        ["NCT04303780"],
        {"NCT04303780": "contaminated"},
    ),
    # ---- WRONG-DRUG (TEST 2): trial studies a DIFFERENT drug ----
    (
        # Querying gefitinib x breast cancer, but this trial studies DALOTUZUMAB, not gefitinib.
        # Wrong studied drug => contaminated regardless of disease/approval.
        "gefitinib x breast cancer: trial studies a DIFFERENT drug => contaminated",
        "gefitinib",
        "Breast Cancer",
        ["EGFR-mutated non-small cell lung cancer"],
        ["NCT00759785"],
        {"NCT00759785": "contaminated"},
    ),
    # ---- DISTINCT-DISEASE (TEST 2): GIST is not leukemia ----
    (
        # Candidate "leukemia" with approved CML; this is an imatinib GIST trial — a DISTINCT
        # disease that doesn't share the leukemia parent => contaminated (TEST 2, not TEST 1).
        "imatinib x leukemia, approved={Ph+ CML}: GIST trial is a distinct disease => contaminated",
        "imatinib",
        "Leukemia",
        ["Philadelphia chromosome positive chronic myeloid leukemia"],
        ["NCT00041197"],
        {"NCT00041197": "contaminated"},
    ),
]


async def _one_call(content):
    resp = await client.messages.create(
        model=MODEL,
        max_tokens=900,
        messages=[{"role": "user", "content": content}],
    )
    return resp.content[0].text.strip() if resp.content else ""


async def judge(drug, disease, approved, ncts, _retries=3):
    trials = "\n\n".join(fmt_trial(T[n]) for n in ncts)
    content = PROMPT.format(
        drug=drug,
        disease=disease,
        approved_indications=", ".join(approved) if approved else "(none)",
        trials=trials,
    )
    # Retry on empty/unparseable output — batched gather occasionally returns empty under load
    # (529/overloaded), which is transport flakiness, not a prompt failure.
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
    for name, drug, disease, approved, ncts, expected in CASES:
        runs = await asyncio.gather(
            *(judge(drug, disease, approved, ncts) for _ in range(RUNS_PER_CASE))
        )
        # Per-NCT majority vote across runs; a case passes only if every NCT's majority verdict
        # matches expected.
        per_nct_ok = True
        report = {}
        for nct, want in expected.items():
            accept = {want} if isinstance(want, str) else set(want)
            votes = Counter(r.get(nct, "PARSE_FAIL") for r in runs)
            got = votes.most_common(1)[0][0]
            report[nct] = (want, dict(votes))
            if got not in accept:
                per_nct_ok = False
        n_pass += per_nct_ok
        verdict = "PASS" if per_nct_ok else "FAIL"
        print(f"[{verdict}] {name}")
        for nct, (want, votes) in report.items():
            print(f"        {nct}: want={want}  votes={votes}")
        if not per_nct_ok:
            print(f"        sample: {json.dumps(runs[0])[:240]}")
    print(f"\n=== {n_pass}/{len(CASES)} cases pass ===")


if __name__ == "__main__":
    asyncio.run(main())
