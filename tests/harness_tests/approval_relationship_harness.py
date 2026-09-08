"""Test the proposed UPSTREAM 4-way approval-relationship classifier.

Today get_fda_approved_disease_mapping returns a bool per candidate ("approved?"). The plan is to
widen that single label-grounded LLM call to return one of four labels per candidate, decided once
and carried as data — instead of the supervisor re-deriving demotion in free-text prose each run
(the source of the T1DM/NAFLD flip-flopping).

Labels:
  approved          — same condition, synonym, OR a narrower CHILD/subset of a labeled
                      indication (patients already covered)              → DROP upstream
  combination_only  — labeled only as part of a combo product           → demote
  contaminated      — broader candidate that contains a supplied approved indication
                                                                         → KEEP ranked + suppress tables
  none              — sibling / related-family / unrelated candidate     → KEEP, rank normally

Only "approved" removes a candidate. "contaminated" and "none" are BOTH kept and ranked — the only
difference is whether the trial tables are trustworthy. This kills the "demoted a real candidate"
class of bugs (systemic HTN for sildenafil, NAFLD, T1DM all stay ranked).

Run: .venv/bin/python tests/harness_tests/approval_relationship_harness.py [model]
"""

import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

from anthropic import AsyncAnthropic

from indication_scout.config import get_settings
from indication_scout.services.approval_check import _parse_approval_decisions
from indication_scout.services.llm import parse_last_json_object

client = AsyncAnthropic(api_key=get_settings().anthropic_api_key)
# Parse args: optional positional MODEL, and optional `--only <substr>` to filter CASES.
_ONLY = None
_args = sys.argv[1:]
if "--only" in _args:
    _i = _args.index("--only")
    _ONLY = _args[_i + 1].lower() if _i + 1 < len(_args) else None
    del _args[_i : _i + 2]
MODEL = _args[0] if _args else "claude-sonnet-4-6"
RUNS_PER_CASE = 5

LABELS = ("approved", "combination_only", "contaminated", "none")

# Each rule is one line followed by its own example. The examples name RA/arthritis/osteoarthritis,
# CML/leukemia/AML, hyperlipidemia/homozygous-FH, stroke/AF, NSCLC/lung-cancer, gonorrhea — so the
# harness CASES below use DIFFERENT drugs and diseases to keep the test honest (no example→answer
# leak).
PROMPT = (
    Path(__file__).parents[2]
    / "src"
    / "indication_scout"
    / "prompts"
    / "extract_fda_approval_single.txt"
).read_text()


# (case name, label text, {candidate: expected_label})
# Drugs/diseases chosen to NOT overlap the prompt's examples (RA/arthritis/OA/gout, CML/leukemia/
# AML, hyperlipidemia/homozygous-FH, stroke/AF, NSCLC/lung-cancer, gonorrhea). Honest held-out test.
CASES = [
    (
        "false contamination — baricitinib: unrelated inflammatory bowel diseases stay none",
        "OLUMIANT (baricitinib) is indicated for moderately to severely active rheumatoid "
        "arthritis, COVID-19 in hospitalized adults requiring supplemental oxygen, and severe "
        "alopecia areata.",
        {
            "rheumatoid arthritis": "approved",
            "arthritis": "contaminated",
            "inflammatory bowel disease": "none",
            "ulcerative colitis": "none",
            "crohn disease": "none",
        },
    ),
    (
        "false contamination — tofacitinib: indications of other JAK inhibitors stay none",
        "XELJANZ (tofacitinib) is indicated for rheumatoid arthritis, psoriatic arthritis, "
        "ankylosing spondylitis, polyarticular course juvenile idiopathic arthritis, and "
        "ulcerative colitis.",
        {
            "rheumatoid arthritis": "approved",
            "ulcerative colitis": "approved",
            "alopecia areata": "none",
            "atopic dermatitis": "none",
            "crohn disease": "none",
        },
    ),
    (
        "false contamination — secukinumab: unrelated inflammatory bowel diseases stay none",
        "COSENTYX (secukinumab) is indicated for plaque psoriasis, psoriatic arthritis, "
        "ankylosing spondylitis, non-radiographic axial spondyloarthritis, and hidradenitis "
        "suppurativa.",
        {
            "plaque psoriasis": "approved",
            "inflammatory bowel disease": "none",
            "ulcerative colitis": "none",
            "crohn disease": "none",
            "rheumatoid arthritis": "none",
        },
    ),
    (
        "false contamination — dupilumab: unrelated immune diseases stay none",
        "DUPIXENT (dupilumab) is indicated for atopic dermatitis, asthma, chronic rhinosinusitis "
        "with nasal polyposis, eosinophilic esophagitis, and prurigo nodularis.",
        {
            "atopic dermatitis": "approved",
            "dermatitis": "contaminated",
            "inflammatory bowel disease": "none",
            "rheumatoid arthritis": "none",
            "psoriasis": "none",
        },
    ),
    (
        "false contamination — rituximab: off-label and immune neighbors stay none",
        "RITUXAN (rituximab) is indicated for non-Hodgkin lymphoma, chronic lymphocytic leukemia, "
        "rheumatoid arthritis, granulomatosis with polyangiitis, and microscopic polyangiitis.",
        {
            "rheumatoid arthritis": "approved",
            "vasculitis": "contaminated",
            "multiple sclerosis": "none",
            "systemic lupus erythematosus": "none",
            "inflammatory bowel disease": "none",
        },
    ),
    (
        "infliximab: severity stripped = approved; 'IBD' parent = contaminated",
        "REMICADE (infliximab) is indicated for the treatment of moderately to severely active "
        "Crohn's disease in adults with an inadequate response to conventional therapy.",
        {
            "crohn disease": "approved",  # bare disease; severity qualifier stripped
            "inflammatory bowel disease": "contaminated",  # parent: Crohn's + UC — KEEP, suspect counts
            "ulcerative colitis": "none",  # sibling IBD subtype, not covered — KEEP
            "celiac disease": "none",  # unrelated GI disease — KEEP
        },
    ),
    (
        "vemurafenib: MINORITY biomarker → broader melanoma is contaminated, not approved",
        "ZELBORAF (vemurafenib) is indicated for the treatment of patients with unresectable or "
        "metastatic melanoma with a BRAF V600E mutation.",
        {
            "BRAF V600E-mutated melanoma": "approved",  # the approved biomarker subset
            "melanoma": "contaminated",  # BRAF V600E is ~40-50% — broader, suspect counts — KEEP
            "uveal melanoma": "none",  # distinct sibling, not covered — KEEP
            "basal cell carcinoma": "none",  # unrelated skin cancer — KEEP
        },
    ),
    (
        "doxycycline: organism-named label phrasing = synonym match",
        "VIBRAMYCIN (doxycycline) is indicated for infections caused by Borrelia burgdorferi "
        "(Lyme disease) and for urethritis caused by Chlamydia trachomatis.",
        {
            "lyme disease": "approved",  # = Borrelia burgdorferi infection
            "chlamydia": "approved",  # = Chlamydia trachomatis urethritis
            "bacterial infection": "contaminated",  # broad parent over the labeled infections — KEEP
            "tuberculosis": "none",  # unrelated infection — KEEP
        },
    ),
    (
        # NOTE: the "reduce the risk of hospitalization for HF" risk-reduction framing makes bare
        # "heart failure" a genuinely borderline approved-vs-contaminated call (narrow approval in a
        # T2DM+CVD population vs broad HF target). That edge is unresolved and not what this harness
        # guards, so heart failure is omitted here; the CKD cause-subtype rows are the kept signal.
        "dapagliflozin: CKD cause-subtypes approved; PKD distinct kept",
        "FARXIGA (dapagliflozin) is indicated for the treatment of chronic kidney disease.",
        {
            "chronic kidney disease": "approved",  # directly labeled
            # Diabetic CKD IS the approved CKD population (on-label) → approved/drop.
            "diabetic nephropathy": "approved",
            "diabetic kidney disease": "approved",
            # PKD is a DISTINCT disease entity that happens to cause CKD — its own trial population,
            # not "CKD that is on-label". Kept as a distinct candidate.
            "polycystic kidney disease": "none",
        },
    ),
    (
        "naltrexone: combo-only for obesity; alcohol dependence approved as monotherapy",
        "REVIA (naltrexone hydrochloride) is indicated for the treatment of alcohol dependence. "
        "The fixed-dose combination naltrexone/bupropion (CONTRAVE) is indicated for chronic "
        "weight management; naltrexone alone is NOT indicated for obesity.",
        {
            "alcohol dependence": "approved",
            "obesity": "combination_only",  # only via Contrave combo
            "cocaine dependence": "none",  # distinct substance disorder — KEEP
            "opioid use disorder": "none",  # not on this label — KEEP
        },
    ),
    (
        "tolvaptan: approved for BROAD 'renal dysfunction'; narrower 'kidney failure' dropped",
        "SAMSCA (tolvaptan) is indicated for the treatment of renal dysfunction (impaired kidney "
        "function of any degree) in adults.",
        {
            "renal dysfunction": "approved",  # the approved broad indication itself
            "kidney failure": "approved",  # narrower CHILD (severe end) — already covered, DROP
            "end-stage renal disease": "approved",  # narrower child synonym — DROP
            "nephrotic syndrome": "none",  # distinct disease — KEEP
        },
    ),
    (
        "bupropion: lay 'depression' is a synonym (approved); 'mood disorder' is a true umbrella",
        "WELLBUTRIN (bupropion hydrochloride) is indicated for the treatment of major depressive "
        "disorder (MDD). ZYBAN (bupropion hydrochloride) is indicated as an aid to smoking "
        "cessation treatment.",
        {
            "major depressive disorder": "approved",
            "depression": "approved",  # lay synonym of MDD — DROP, not a broad umbrella
            "depressive disorder": "approved",  # near-synonym of MDD — DROP
            "mood disorder": "contaminated",  # true umbrella: MDD + bipolar + dysthymia — KEEP, suspect
            "bipolar disorder": "none",  # distinct mood disorder, not covered — KEEP
            "nicotine dependence": "approved",  # = smoking cessation (Zyban)
        },
    ),
    (
        # Held-out (empagliflozin not used in the prompt examples). The DKD regression: a cause-
        # named subtype of approved broad CKD is ON-LABEL → approved (NOT none), since the
        # candidate's patients fall within the approved CKD population. T2DM is the qualifier
        # population (none); heart failure approved.
        "empagliflozin: CKD cause-subtypes approved (on-label); T2DM qualifier none",
        "JARDIANCE (empagliflozin) is indicated for the treatment of chronic kidney disease, and "
        "to reduce the risk of cardiovascular death in adults with type 2 diabetes mellitus and "
        "established cardiovascular disease, and for heart failure.",
        {
            "chronic kidney disease": "approved",
            "diabetic nephropathy": "approved",  # cause-subtype of approved CKD — on-label
            "diabetic kidney disease": "approved",
            "heart failure": "approved",
            "type 2 diabetes mellitus": "none",  # qualifier population only — KEEP
            # Glomerulonephritis is a DISTINCT kidney disease (immune-mediated), its own entity that
            # causes CKD — not "on-label CKD". Kept as a distinct candidate.
            "glomerulonephritis": "none",
        },
    ),
    (
        # Subtype-vs-sibling-vs-distinct boundary on a held-out drug (dupilumab, not in the prompt
        # examples). Approved broad "atopic dermatitis" → "pediatric atopic dermatitis" is the SAME
        # disease in a sub-population (on-label → approved); "eczema" is a lay synonym (approved);
        # "contact dermatitis" is a DISTINCT dermatitis (different cause, kept); "psoriasis" is an
        # unrelated inflammatory skin disease (kept).
        "dupilumab: AD subtype on-label; distinct/unrelated dermatoses kept",
        "DUPIXENT (dupilumab) is indicated for the treatment of moderate-to-severe atopic "
        "dermatitis in adults and pediatric patients.",
        {
            "atopic dermatitis": "approved",
            "pediatric atopic dermatitis": "approved",  # sub-population of the same disease — on-label
            "eczema": "approved",  # lay synonym of atopic dermatitis
            "contact dermatitis": "none",  # distinct dermatitis (different cause) — KEEP
            "psoriasis": "none",  # unrelated inflammatory skin disease — KEEP
        },
    ),
    (
        # HF subtype boundary (carvedilol, held-out). Approved broad "heart failure" → "HFrEF" is a
        # subtype within the approved HF population (on-label → approved); "HFpEF" historically had
        # distinct/limited evidence but is still HF — accept either approved or contaminated;
        # "cardiomyopathy" is a broader/parent category (contaminated); "atrial fibrillation" is a
        # distinct arrhythmia (kept).
        "carvedilol: HFrEF subtype on-label; distinct arrhythmia kept",
        "COREG (carvedilol) is indicated for the treatment of heart failure (mild to severe) of "
        "ischemic or cardiomyopathic origin.",
        {
            "heart failure": "approved",
            "heart failure with reduced ejection fraction": "approved",  # HFrEF — on-label HF subtype
            "atrial fibrillation": "none",  # distinct arrhythmia — KEEP
            "hypertension": "none",  # not on this label snippet — KEEP
        },
    ),
    (
        # Dry and wet AMD are distinct sibling diseases. The general classifier therefore returns
        # none. Any verified registry-query collision belongs in the exact curated table.
        "aflibercept: wet and dry AMD remain distinct sibling diseases",
        "EYLEA (aflibercept) is indicated for the treatment of neovascular (wet) age-related "
        "macular degeneration (AMD) and diabetic macular edema.",
        {
            "dry age-related macular degeneration": "none",
            "diabetic macular edema": "approved",  # on-label
            "retinitis pigmentosa": "none",  # distinct retinal disease, no collision — KEEP clean
        },
    ),
]


APPROVED_INDICATIONS_BY_DRUG = {
    "baricitinib": ["rheumatoid arthritis", "COVID-19", "severe alopecia areata"],
    "tofacitinib": [
        "rheumatoid arthritis",
        "psoriatic arthritis",
        "ankylosing spondylitis",
        "polyarticular course juvenile idiopathic arthritis",
        "ulcerative colitis",
    ],
    "secukinumab": [
        "plaque psoriasis",
        "psoriatic arthritis",
        "ankylosing spondylitis",
        "non-radiographic axial spondyloarthritis",
        "hidradenitis suppurativa",
    ],
    "dupilumab": [
        "atopic dermatitis",
        "asthma",
        "chronic rhinosinusitis with nasal polyposis",
        "eosinophilic esophagitis",
        "prurigo nodularis",
    ],
    "rituximab": [
        "non-Hodgkin lymphoma",
        "chronic lymphocytic leukemia",
        "rheumatoid arthritis",
        "granulomatosis with polyangiitis",
        "microscopic polyangiitis",
    ],
    "infliximab": ["Crohn disease"],
    "vemurafenib": ["BRAF V600E-mutated melanoma"],
    "doxycycline": ["Lyme disease", "chlamydia"],
    "dapagliflozin": ["chronic kidney disease"],
    "naltrexone": ["alcohol dependence", "chronic weight management"],
    "tolvaptan": ["renal dysfunction"],
    "bupropion": ["major depressive disorder", "nicotine dependence"],
    "empagliflozin": [
        "chronic kidney disease",
        "cardiovascular risk reduction",
        "heart failure",
    ],
    "carvedilol": ["heart failure"],
    "aflibercept": [
        "neovascular age-related macular degeneration",
        "diabetic macular edema",
    ],
}


def approved_indications_for(case_name: str) -> list[str]:
    """Return the explicit approved-indication fixture for one case."""
    case_lower = case_name.lower()
    for drug, approved_indications in APPROVED_INDICATIONS_BY_DRUG.items():
        if drug in case_lower:
            return approved_indications
    raise KeyError(f"No approved-indication fixture for {case_name!r}")


async def classify(
    label_texts: str,
    candidates: list[str],
    approved_indications: list[str],
) -> dict[str, str]:
    resp = await client.messages.create(
        model=MODEL,
        max_tokens=500,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": PROMPT.format(
                    label_texts=label_texts,
                    approved_indications=json.dumps(approved_indications),
                    candidate_diseases=json.dumps(candidates),
                ),
            }
        ],
    )
    text = resp.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```")[1].lstrip("json").strip()
    parsed = parse_last_json_object(text)
    if parsed is None:
        return {}

    return _parse_approval_decisions(parsed, candidates, approved_indications)


def grade(got, expected):
    """Return list of (candidate, expected, got) mismatches."""
    misses = []
    for cand, exp in expected.items():
        g = got.get(cand, "MISSING")
        if g not in LABELS and g != "MISSING":
            g = f"INVALID:{g}"
        if g != exp:
            misses.append((cand, exp, g))
    return misses


def drug_of(case_name):
    return case_name.split(":")[0].strip()


async def main():
    print(f"=== model: {MODEL}  ({RUNS_PER_CASE} runs/case) ===\n")
    # `--only <substr>` (parsed at import) filters CASES by name (e.g. --only bupropion).
    cases = [c for c in CASES if _ONLY is None or _ONLY in c[0].lower()]
    rows = []  # (drug, disease, model_label, correct_label, ok)
    for name, label, expected in cases:
        cands = list(expected.keys())
        approved_indications = approved_indications_for(name)
        runs = await asyncio.gather(
            *(
                classify(label, cands, approved_indications)
                for _ in range(RUNS_PER_CASE)
            )
        )
        drug = drug_of(name)
        for cand, exp in expected.items():
            votes = Counter(r.get(cand, "MISSING") for r in runs)
            model_label, _ = votes.most_common(1)[0]
            # Show split if the model wasn't unanimous.
            disp = (
                model_label
                if len(votes) == 1
                else "/".join(f"{lbl}×{c}" for lbl, c in votes.most_common())
            )
            rows.append((drug, cand, disp, exp, model_label == exp))

    w_drug = max(len(r[0]) for r in rows)
    w_dis = max(len(r[1]) for r in rows)
    w_mod = max(len(r[2]) for r in rows)
    w_cor = max(len(r[3]) for r in rows)
    print(
        f"{'DRUG':<{w_drug}}  {'DISEASE':<{w_dis}}  {'MODEL':<{w_mod}}  {'CORRECT':<{w_cor}}  OK"
    )
    print("-" * (w_drug + w_dis + w_mod + w_cor + 12))
    for drug, dis, mod, cor, ok in rows:
        print(
            f"{drug:<{w_drug}}  {dis:<{w_dis}}  {mod:<{w_mod}}  {cor:<{w_cor}}  {'✓' if ok else '✗'}"
        )
    n_ok = sum(1 for r in rows if r[4])
    print(f"\n=== {n_ok}/{len(rows)} disease labels match (majority vote) ===")


if __name__ == "__main__":
    asyncio.run(main())
