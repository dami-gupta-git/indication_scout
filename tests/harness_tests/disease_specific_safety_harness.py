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
import logging

from indication_scout.constants import DEFAULT_CACHE_DIR
from indication_scout.data_sources.chembl import resolve_drug_name
from indication_scout.services.retrieval import RetrievalService

logger = logging.getLogger(__name__)

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
    # Labeled False as "ED literature is efficacy-dominated" until the retrieved corpus was read:
    # it contains a randomised crossover trial of sildenafil FOR ED that stopped recruitment after
    # three of six men with multiple system atrophy dropped their blood pressure severely an hour
    # post-dose (PMID 11511713). Attributed, and it halted the study — a flag is correct.
    ("sildenafil", "erectile dysfunction", True),
    # No disease-context safety finding expected (efficacy-only or thin AE literature for the pair).
    # Labeled True on metformin's lactic-acidosis reputation until the retrieved corpus was read:
    # the two lactic-acidosis papers it returns are the Cochrane review concluding there is NO
    # evidence of increased risk versus other anti-hyperglycemics (PMID 20091535). No flag is the
    # correct answer for this evidence.
    ("metformin", "type 2 diabetes", False),
    ("rofecoxib", "migraine", False),              # migraine trials are efficacy; CV signal is not migraine-context
    # The 2026-09-08 sildenafil run flagged ischemic stroke off PMID 19717023: a 12-patient
    # uncontrolled study whose primary outcome was a counted safety event (one sudden death) and
    # whose own conclusion was that the drug appeared safe. An event the authors record without
    # attributing it to the drug is not a confirmed harm.
    ("sildenafil", "ischemic stroke", False),
]


async def _classify(svc: RetrievalService, drug: str, disease: str) -> bool | None:
    chembl_id = await resolve_drug_name(drug, DEFAULT_CACHE_DIR)
    safety_results = await svc.safety_search(chembl_id, disease=disease)
    verdict, _, _ = await svc.classify_indication_harm(
        chembl_id,
        disease,
        safety_results.disease_scoped,
    )
    return verdict


async def main() -> None:
    svc = RetrievalService(cache_dir=DEFAULT_CACHE_DIR)
    correct = 0
    for drug, disease, expect in CASES:
        verdict = await _classify(svc, drug, disease)
        ok = verdict == expect
        correct += ok
        mark = "OK " if ok else "XX "
        logger.info(
            "%s%-12s x %-20s expect=%-5s verdict=%-5s",
            mark,
            drug,
            disease,
            expect,
            verdict,
        )
    logger.info("%d/%d correct", correct, len(CASES))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
