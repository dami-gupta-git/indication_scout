"""Integration test for services/literature_strength.judge_literature_strength — the LIVE LLM
drug-specific strength judgment.

Hits real Anthropic. Abstracts are REAL PubMed text (embedded inline so the test is
self-contained — no DB dependency), the same evidence that produced the Parkinson card-vs-prose
bug. Crux shapes mirror scratch/literature_strength_harness.py:
- Parkinson: GLP-1 class RCTs (lixisenatide / exenatide / NLY01) + one off-topic semaglutide
  (depression) abstract → evidence_basis="class_level", strength NOT "strong".
- T1DM: two genuine semaglutide RCTs → drug_specific, strength NOT "none".

Uses test_cache_dir so the real cache is untouched.
"""

import logging

from indication_scout.services.literature_strength import judge_literature_strength

logger = logging.getLogger(__name__)

# Real abstracts — class-level GLP-1 RCTs for Parkinson's (NOT semaglutide) + one off-topic
# semaglutide abstract (depression). This is the exact shape that inflated the card to "strong".
_PARKINSON_ABSTRACTS = [
    {
        "pmid": "38598572",
        "title": "Trial of Lixisenatide in Early Parkinson's Disease.",
        "abstract": (
            "BACKGROUND: Lixisenatide, a glucagon-like peptide-1 receptor agonist used for "
            "the treatment of diabetes, has shown neuroprotective properties in a mouse model "
            "of Parkinson's disease. METHODS: In this phase 2, double-blind, randomized, "
            "placebo-controlled trial, we assessed the effect of lixisenatide on the "
            "progression of motor disability in persons with Parkinson's disease."
        ),
    },
    {
        "pmid": "23728174",
        "title": "Exenatide and the treatment of patients with Parkinson's disease.",
        "abstract": (
            "Exenatide is a type 2 diabetes treatment that has been shown to have "
            "neuroprotective/neurorestorative properties in preclinical models of "
            "neurodegeneration. Using a single-blind trial design, we assessed exenatide in "
            "patients with moderate Parkinson's disease."
        ),
    },
    {
        "pmid": "28781108",
        "title": (
            "Exenatide once weekly versus placebo in Parkinson's disease: a randomised, "
            "double-blind, placebo-controlled trial."
        ),
        "abstract": (
            "Exenatide, a glucagon-like peptide-1 (GLP-1) receptor agonist, has "
            "neuroprotective effects in preclinical models of Parkinson's disease. In this "
            "single-centre, randomised, double-blind, placebo-controlled trial, patients with "
            "moderate Parkinson's disease were randomly assigned to receive subcutaneous "
            "exenatide or placebo."
        ),
    },
    {
        "pmid": "38101901",
        "title": (
            "Safety, tolerability, and efficacy of NLY01 in early untreated Parkinson's "
            "disease: a randomised, double-blind, placebo-controlled trial."
        ),
        "abstract": (
            "We sought to test the safety and efficacy of NLY01 - a brain-penetrant, "
            "pegylated, longer-lasting version of exenatide (a glucagon-like peptide-1 "
            "receptor agonist) - in early untreated Parkinson's disease."
        ),
    },
    {
        "pmid": "41218611",
        "title": (
            "Semaglutide for the treatment of cognitive dysfunction in major depressive "
            "disorder: A randomized clinical trial."
        ),
        "abstract": (
            "This was a 16-week, randomized, double-blind, placebo-controlled, parallel-group "
            "trial evaluating semaglutide for cognitive dysfunction in adults with major "
            "depressive disorder (MDD). It does not concern Parkinson's disease."
        ),
    },
]

# Real abstracts — two genuine semaglutide RCTs in type 1 diabetes (drug-specific).
_T1DM_ABSTRACTS = [
    {
        "pmid": "40550013",
        "title": "Semaglutide in Adults with Type 1 Diabetes and Obesity.",
        "abstract": (
            "Once-weekly semaglutide is approved for type 2 diabetes and obesity. In this "
            "26-week, double-blind trial, we randomly assigned 72 adults with type 1 diabetes "
            "using an automated insulin delivery system and a body mass index of 30 or higher "
            "to receive once-weekly semaglutide or placebo."
        ),
    },
    {
        "pmid": "41144928",
        "title": (
            "Changes to insulin requirements over time with semaglutide in adults with type 1 "
            "diabetes on insulin pump therapy: A post-hoc analysis of a double-blinded, "
            "randomised, crossover trial."
        ),
        "abstract": (
            "This is a post-hoc analysis of a double-blinded, randomised, crossover trial "
            "assessing semaglutide versus placebo during automated insulin delivery in adults "
            "with type 1 diabetes on insulin pump therapy."
        ),
    },
]


async def test_class_level_parkinson_is_not_strong(test_cache_dir):
    """The Parkinson bug: GLP-1 class RCTs (other drugs) + an off-topic semaglutide abstract →
    evidence_basis='class_level' and strength NOT 'strong' (no direct semaglutide evidence).
    """
    s = await judge_literature_strength(
        _PARKINSON_ABSTRACTS,
        drug="semaglutide",
        indication="Parkinson Disease",
        cache_dir=test_cache_dir,
    )
    assert s is not None
    assert (
        s.evidence_basis == "class_level"
    ), f"expected class_level, got {s.evidence_basis!r}"
    assert s.strength != "strong", f"class-level evidence graded strong: {s!r}"
    # the parser enforces none for class_level
    assert s.strength == "none"
    assert s.direction == "none"


async def test_drug_specific_t1dm_is_not_none(test_cache_dir):
    """Two genuine semaglutide RCTs in T1DM → evidence_basis='drug_specific' with real
    drug-specific strength (not 'none')."""
    s = await judge_literature_strength(
        _T1DM_ABSTRACTS,
        drug="semaglutide",
        indication="Type 1 Diabetes Mellitus",
        cache_dir=test_cache_dir,
    )
    assert s is not None
    assert (
        s.evidence_basis == "drug_specific"
    ), f"expected drug_specific, got {s.evidence_basis!r}"
    assert s.strength != "none", f"genuine semaglutide RCTs graded none: {s!r}"
    assert s.direction in {"supports", "mixed", "contradicts"}


async def test_empty_abstracts_returns_none(test_cache_dir):
    """No abstracts → None (the caller keeps the synthesize values)."""
    s = await judge_literature_strength(
        [], drug="semaglutide", indication="Parkinson Disease", cache_dir=test_cache_dir
    )
    assert s is None
