"""Harness: can an LLM reliably tell a DISEASE-SPECIFIC safety signal from a generic drug-wide one?

Motivation: the safety pass fetches drug-level + disease-scoped adverse-event abstracts. We want a
per-candidate "disease-specific safety" flag for the ranking summary table — but only if the
classification is reliable. First pass (a loose prompt) scored 3/6, over-calling disease-specific
for generic harms merely STUDIED in a disease's patients (aspirin/warfarin bleeding). This harness
iterates on the prompt against a labeled case set.

Each case: (drug, disease, expected disease_specific). Abstracts come from the live safety_search
(drug-level [Majr] + disease-scoped), so the harness tests the real input the pipeline would give.

Run: .venv/bin/python tests/harness_tests/disease_specific_safety_harness.py [runs]
"""

import asyncio
import sys
from collections import Counter
from pathlib import Path

from indication_scout.constants import DEFAULT_CACHE_DIR
from indication_scout.data_sources.chembl import resolve_drug_name
from indication_scout.services.llm import parse_last_json_object, query_llm
from indication_scout.services.retrieval import RetrievalService

RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
_PROMPT = (
    Path(__file__).parent / "prompts" / "disease_specific_safety_prompt.txt"
).read_text()

# (drug, disease, expected harm_reported_for_indication). The reframed question: does an abstract
# report a safety finding for the drug IN THIS INDICATION's context (not "is it unique to the
# disease"). True when a disease-context harm is reported; False when the disease-context papers are
# efficacy-only or there is no disease-context safety paper.
CASES: list[tuple[str, str, bool]] = [
    # Harm reported in the indication's context.
    ("rofecoxib", "colorectal cancer", True),      # CV thrombotic events in adenoma-prevention trials
    ("thalidomide", "multiple myeloma", True),     # VTE reported in myeloma treatment
    ("natalizumab", "multiple sclerosis", True),   # PML reported in MS therapy
    ("bevacizumab", "colorectal cancer", True),    # GI perforation reported in colorectal use
    ("warfarin", "atrial fibrillation", True),     # bleeding reported in AF anticoagulation
    ("metformin", "type 2 diabetes", True),        # lactic acidosis reported in T2D use
    # No disease-context safety finding expected (efficacy-only or thin AE literature for the pair).
    ("rofecoxib", "migraine", False),              # migraine trials are efficacy; CV signal is not migraine-context
    ("sildenafil", "erectile dysfunction", False), # ED literature is efficacy-dominated
]


async def _classify(svc: RetrievalService, drug: str, disease: str) -> list[bool | None]:
    chembl_id = await resolve_drug_name(drug, DEFAULT_CACHE_DIR)
    abstracts = await svc.safety_search(chembl_id, disease=disease)
    formatted = "\n\n".join(
        f"PMID {a.pmid}: {a.title}. {(a.abstract or '')[:250]}" for a in abstracts[:15]
    )
    prompt = _PROMPT.format(drug=drug, disease=disease, abstracts=formatted)
    out: list[bool | None] = []
    for _ in range(RUNS):
        resp = await query_llm(prompt)
        data = parse_last_json_object(resp) or {}
        out.append(data.get("harm_reported_for_indication"))
    return out


async def main() -> None:
    svc = RetrievalService(cache_dir=DEFAULT_CACHE_DIR)
    correct = 0
    for drug, disease, expect in CASES:
        verdicts = await _classify(svc, drug, disease)
        modal = Counter(verdicts).most_common(1)[0][0]
        ok = modal == expect
        correct += ok
        mark = "OK " if ok else "XX "
        print(
            f"{mark}{drug:12s} x {disease:20s} expect={str(expect):5s} "
            f"modal={str(modal):5s} runs={verdicts}"
        )
    print(f"\n{correct}/{len(CASES)} correct (modal across {RUNS} runs)")


if __name__ == "__main__":
    asyncio.run(main())
