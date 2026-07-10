"""Standalone harness: feed a model ONLY the enriched registry facts (title +
stop-reason + generic status + phase + literature) and see if it judges the known cases
correctly — WITHOUT the supervisor pipeline. Proves whether fact-enrichment is enough.

Run: .venv/bin/python tests/harness_tests/fact_judge_harness.py
"""

import asyncio
import glob
import json
import os
from pathlib import Path

from anthropic import AsyncAnthropic

from indication_scout.agents.clinical_trials.clinical_trials_output import (
    ClinicalTrialsOutput,
)
from indication_scout.agents.clinical_trials.clinical_trials_tools import (
    _classify_stop_reason,
)

PROMPT = (Path(__file__).parent / "prompts" / "fact_judge_prompt.txt").read_text()
client = AsyncAnthropic()

# Drugs known to be generic/off-patent (for the "no NDA != failure" cue).
GENERIC = {"bupropion", "metformin", "duloxetine", "gefitinib", "imatinib", "sildenafil",
           "methotrexate", "raloxifene", "riluzole"}

# Expected outcome per case — what a fully-informed analyst should conclude.
EXPECTED = {
    ("bupropion", "attention deficit hyperactivity disorder"):
        "LIVE (real Phase 3/4, generic so no NDA — not closed)",
    ("bupropion", "cocaine dependence"):
        "NCT01077024 contaminated (smoking trial); not a real Phase 3 for cocaine",
    ("bupropion", "fibromyalgia"):
        "NCT04747314 contaminated (low-back-pain title); no real trial for fibromyalgia",
    ("baricitinib", "systemic lupus erythematosus"):
        "CLOSED — two Phase 3 terminated for safety / benefit:risk",
    ("baricitinib", "psoriasis"):
        "LIVE — Phase 2 only, no negative signal",
}

CASES = [
    # Adjacency / contamination cases — the point of feeding MeSH conditions:
    ("sildenafil", ["hypertension", "pulmonary hypertension"]),  # parent vs child + sitaxsentan
    ("methotrexate", ["sarcoma"]),                            # over-exclusion check
    ("bupropion", ["cocaine dependence", "fibromyalgia"]),    # different-disease contam
    ("baricitinib", ["systemic lupus erythematosus"]),        # genuine closure must survive
]

# Fresh cases: expected outcome left as a NOTE (these are genuinely uncertain — that's
# the point of testing them).
EXPECTED.update({
    ("semaglutide", "non-alcoholic fatty liver disease"):
        "? check if the Phase 2/3 trial is actually a PCOS trial (contamination)",
    ("sildenafil", "hypertension"):
        "? the terminated Phase 3s are sitaxsentan (ANOTHER drug) — must NOT close sildenafil on them",
    ("sildenafil", "pulmonary hypertension"):
        "? same sitaxsentan terminations — sildenafil itself has real completed P3 for PAH",
    ("empagliflozin", "diabetic nephropathy"):
        "? is the terminated P3 a real safety/efficacy stop or operational?",
    ("metformin", "polycystic ovary syndrome"):
        "LIVE — 13 real Phase 3, generic, no negative signal",
    ("methotrexate", "sarcoma"):
        "? Phase 3 titled Non-Hodgkin's — check contamination",
    ("imatinib", "glioblastoma multiforme"):
        "? 2 completed Phase 3, no approval — live or genuinely failed?",
})


def _latest(drug: str) -> str:
    return sorted(glob.glob(f"test_reports/{drug}_2026-06-13_*.json"))[-1]


def _mesh(t):
    return "; ".join(m.term for m in (t.mesh_conditions or [])) or "(no mesh)"


def _fmt_trials(trials, with_stop=False):
    lines = []
    for t in trials:
        if with_stop:
            lines.append(
                f"  {t.nct_id} | {t.phase} | {_classify_stop_reason(t.why_stopped)} "
                f"| mesh: {_mesh(t)} | {t.title}"
            )
        else:
            lines.append(f"  {t.nct_id} | {t.phase} | mesh: {_mesh(t)} | {t.title}")
    return "\n".join(lines) or "  (none)"


async def _query_mesh(disease):
    from indication_scout.services.disease_helper import resolve_mesh_id
    r = await resolve_mesh_id(disease)
    return f"{r[1]} ({r[0]})" if r else "unresolved"


async def _build_prompt(drug, disease, f):
    ct = ClinicalTrialsOutput(**(f.get("clinical_trials") or {}))
    es = (f.get("literature") or {}).get("evidence_summary") or {}
    ap = ct.approval
    comp = [t for t in (ct.completed.trials if ct.completed else [])]
    term = [t for t in (ct.terminated.trials if ct.terminated else [])]
    return PROMPT.format(
        drug=drug,
        generic=str(drug.lower() in GENERIC),
        indication=disease,
        query_mesh=await _query_mesh(disease),
        lit_strength=es.get("strength"),
        lit_studies=es.get("study_count"),
        is_approved=ap.is_approved if ap else None,
        label_found=ap.label_found if ap else None,
        completed=_fmt_trials(comp),
        terminated=_fmt_trials(term, with_stop=True),
    )


async def _judge(model, prompt):
    resp = await client.messages.create(
        model=model, max_tokens=1200, temperature=0,
        system="Output ONLY the JSON object. No preamble, no prose before or after.",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


async def main():
    models = {"sonnet": "claude-sonnet-4-6", "opus": "claude-opus-4-6"}
    for drug, dz_list in CASES:
        d = json.load(open(_latest(drug)))
        by_dz = {f["disease"].lower(): f for f in d["disease_findings"]}
        for disease in dz_list:
            f = by_dz.get(disease)
            if not f:
                continue
            prompt = await _build_prompt(drug, disease, f)
            print("=" * 80)
            print(f"{drug} × {disease}")
            print(f"  EXPECTED: {EXPECTED.get((drug, disease))}")
            for label, model in models.items():
                out = await _judge(model, prompt)
                try:
                    j = json.loads(out[out.index("{"):out.rindex("}") + 1])
                    print(f"  [{label}] phase={j.get('highest_real_completed_phase')!r} "
                          f"verdict={j.get('verdict')!r}")
                    print(f"          reasoning: {j.get('reasoning')}")
                except Exception:
                    print(f"  [{label}] (unparsed) {out[:200]}")


if __name__ == "__main__":
    os.chdir(Path(__file__).parent.parent)
    asyncio.run(main())
