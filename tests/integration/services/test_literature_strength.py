"""Integration test for the combined RetrievalService.synthesize call — the LIVE LLM that reads
the abstracts once and emits a single EvidenceSummary (per-PMID verdicts + drug-specific
strength/direction/basis + prose). Replaces the retired judge_literature_strength judgment; the
merged synthesize prompt must carry every rule the judge used.

Hits real Anthropic. Abstracts are REAL PubMed text (embedded inline so the test is
self-contained — no DB dependency), the same evidence that produced the Parkinson card-vs-prose
bug. Crux shapes mirror scratch/literature_strength_harness.py:
- Parkinson: GLP-1 class RCTs (lixisenatide / exenatide / NLY01) + one off-topic semaglutide
  (depression) abstract → evidence_basis="class_level", strength NOT "strong".
- T1DM: two genuine semaglutide RCTs → drug_specific, strength NOT "none".

get_all_drug_names is patched to the known drug name so the test isolates the merged-prompt
behavior (no OT/ChEMBL network); the ChEMBL ID is a sentinel. Uses test_cache_dir so the real
cache is untouched.
"""

import logging
from unittest.mock import AsyncMock, patch

import pytest

from indication_scout.services.retrieval import AbstractResult, RetrievalService

logger = logging.getLogger(__name__)


def _to_abstracts(dicts: list[dict]) -> list[AbstractResult]:
    """Wrap the inline {pmid,title,abstract} dicts as AbstractResults synthesize accepts."""
    return [
        AbstractResult(
            pmid=d["pmid"], title=d["title"], abstract=d["abstract"], similarity=0.9
        )
        for d in dicts
    ]


async def _synthesize(
    abstracts: list[dict],
    *,
    drug: str,
    indication: str,
    cache_dir,
    approved_indications: list[str] | None = None,
):
    """Run the live combined synthesize over inline abstracts, patching name resolution to `drug`."""
    svc = RetrievalService(cache_dir)
    with patch(
        "indication_scout.services.retrieval.get_all_drug_names",
        new=AsyncMock(return_value=[drug]),
    ):
        return await svc.synthesize(
            "CHEMBL_SENTINEL",
            indication,
            _to_abstracts(abstracts),
            approved_indications=approved_indications,
        )


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


# Real abstracts — the paired baricitinib SLE phase-3 program: BRAVE-I met its primary endpoint,
# BRAVE-II failed its primary endpoint. The per-PMID directional verdict must split them by their
# OWN result, not lump the program as all-contradicting (the BRAVE-I bug).
_SLE_BRAVE_ABSTRACTS = [
    {
        "pmid": "36848918",
        "title": "Baricitinib for systemic lupus erythematosus (SLE-BRAVE-I): a phase 3 trial.",
        "abstract": (
            "In this phase 3, double-blind, randomised, placebo-controlled trial, baricitinib "
            "4 mg met the primary endpoint: SRI-4 response at week 52 was achieved by 57% of "
            "patients vs 46% on placebo (odds ratio 1.57, 95% CI 1.09-2.27; p=0.016). No key "
            "secondary endpoints were met."
        ),
    },
    {
        "pmid": "36848919",
        "title": "Baricitinib for systemic lupus erythematosus (SLE-BRAVE-II): a phase 3 trial.",
        "abstract": (
            "In this phase 3, double-blind, randomised, placebo-controlled trial, baricitinib "
            "did not meet the primary endpoint: SRI-4 response at week 52 with baricitinib 4 mg "
            "was 47% vs 46% on placebo (odds ratio 1.07, 95% CI 0.75-1.53). No secondary "
            "endpoints were met."
        ),
    },
]

# Real abstracts — metformin × hepatic steatosis, the ARM-ATTRIBUTION case the regex guard could
# not handle: the benefit belongs to a COMPARATOR (ipragliflozin), not metformin, so the abstract
# is CONTRADICTING for metformin even though it contains "significantly improved". Only the
# per-PMID direction sub-call attributes the benefit to the right drug.
_STEATOSIS_ARM_ATTRIBUTION_ABSTRACTS = [
    {
        "pmid": "39806556",
        "title": "Ipragliflozin versus metformin on hepatic steatosis indices (sub-analysis).",
        "abstract": (
            "In a sub-analysis of a randomized controlled study, ipragliflozin significantly "
            "improved hepatic steatosis indices (FLI, HSI, NAFLD-LFS) and the liver fibrosis "
            "index (APRI) compared with metformin, which showed minimal change from baseline."
        ),
    },
    {
        "pmid": "19811343",
        "title": "Metformin in NAFLD: a double-blind placebo-controlled trial.",
        "abstract": (
            "In a randomized, double-blind, placebo-controlled trial of 48 NAFLD patients, "
            "metformin for 6 months showed no significant difference from placebo in "
            "histologically or CT-assessed liver steatosis, NAS score, or liver transaminases."
        ),
    },
]


def _assert_self_consistent(s):
    """The NEW consistency invariants the merge guarantees by construction: prose/key_findings
    cite only relevant PMIDs; supporting/contradicting are a subset of relevant with no overlap;
    study_count agrees with the relevant set size."""
    import re as _re

    relevant = set(s.relevant_pmids)
    assert set(s.supporting_pmids) <= relevant, "supporting cites a contaminated PMID"
    assert (
        set(s.contradicting_pmids) <= relevant
    ), "contradicting cites a contaminated PMID"
    # supporting∩contradicting is ALLOWED now: a PMID with verdict "mixed" appears in BOTH lists by
    # design (it carries evidence in each direction). The only invariant is that any such overlap
    # is the mixed set — i.e. still within relevant (checked above) and never in contaminated.
    assert set(s.relevant_pmids).isdisjoint(
        s.contaminated_pmids
    ), "relevant∩contaminated"
    # neutral PMIDs are relevant but in NEITHER directional list.
    assert set(s.neutral_pmids) <= relevant, "neutral cites a contaminated PMID"
    assert not (set(s.neutral_pmids) & set(s.supporting_pmids)), "neutral∩supporting"
    assert not (
        set(s.neutral_pmids) & set(s.contradicting_pmids)
    ), "neutral∩contradicting"
    # Orphan-PMID invariant (the class of bug the metformin-CVD regression exposed: prose cited
    # PMIDs the rollup dropped): each `key_findings` bullet must NOT cite a PMID outside the
    # relevant set. The merge owns this by construction (one author); assert it directly since the
    # deterministic orphan-trim was removed. Exercised on the drug-disease cases below (not
    # metformin specifically — the invariant is drug-agnostic).
    # NOTE: the `summary` prose is deliberately exempt — the prompt permits it to cite a
    # CONTAMINATED PMID solely to explain WHY it was excluded (e.g. "the RCTs were for sibling
    # drug X, not this drug"), which is exactly the class-level/approved disclaimer case.
    for kf in s.key_findings:
        cited = set(_re.findall(r"PMID:\s*(\d+)", kf))
        assert cited <= relevant, (
            f"key_finding cites non-relevant PMID(s) {cited - relevant}: {kf}"
        )


@pytest.mark.approval_aware
async def test_class_level_parkinson_is_not_strong(test_cache_dir):
    """The Parkinson bug: GLP-1 class RCTs (other drugs) + an off-topic semaglutide abstract →
    evidence_basis='class_level' and strength NOT 'strong' (no direct semaglutide evidence).
    """
    s = await _synthesize(
        _PARKINSON_ABSTRACTS,
        drug="semaglutide",
        indication="Parkinson Disease",
        cache_dir=test_cache_dir,
    )
    assert (
        s.evidence_basis == "class_level"
    ), f"expected class_level, got {s.evidence_basis!r}"
    # the deterministic strength cap forces none for a non-drug_specific basis
    assert s.strength == "none", f"class-level evidence graded {s.strength!r}"
    assert s.direction == "none"
    _assert_self_consistent(s)


@pytest.mark.approval_aware
async def test_drug_specific_t1dm_is_not_none(test_cache_dir):
    """Two genuine semaglutide RCTs in T1DM → evidence_basis='drug_specific' with real
    drug-specific strength (not 'none')."""
    s = await _synthesize(
        _T1DM_ABSTRACTS,
        drug="semaglutide",
        indication="Type 1 Diabetes Mellitus",
        cache_dir=test_cache_dir,
    )
    assert (
        s.evidence_basis == "drug_specific"
    ), f"expected drug_specific, got {s.evidence_basis!r}"
    assert s.strength != "none", f"genuine semaglutide RCTs graded none: {s!r}"
    assert s.direction in {"supports", "mixed", "contradicts"}
    _assert_self_consistent(s)


@pytest.mark.approval_aware
async def test_paired_program_splits_positive_and_negative_trials(test_cache_dir):
    """The BRAVE-I/II bug: a positive trial (endpoint met) must land in supporting even when its
    sibling failed; the negative trial stays contradicting-only. Per-PMID directional verdict +
    the positive-finding guard own this split (built in code from the verdict map)."""
    s = await _synthesize(
        _SLE_BRAVE_ABSTRACTS,
        drug="baricitinib",
        indication="Systemic Lupus Erythematosus",
        cache_dir=test_cache_dir,
    )
    # BRAVE-I met its primary endpoint -> must appear in supporting (it may also be in
    # contradicting if graded mixed for missing secondaries — that is acceptable).
    assert "36848918" in s.supporting_pmids, (
        f"BRAVE-I (positive) not in supporting: {s.supporting_pmids}"
    )
    # BRAVE-II is a clean primary-endpoint failure -> contradicting, and must NOT be in supporting.
    assert "36848919" in s.contradicting_pmids, (
        f"BRAVE-II (negative) not in contradicting: {s.contradicting_pmids}"
    )
    assert "36848919" not in s.supporting_pmids, (
        f"BRAVE-II (clean failure) leaked into supporting: {s.supporting_pmids}"
    )
    # Both are this-drug-this-disease evidence with one positive and one negative -> mixed overall.
    assert s.direction == "mixed", f"expected mixed, got {s.direction!r}"
    _assert_self_consistent(s)


@pytest.mark.approval_aware
async def test_comparator_benefit_not_credited_to_drug(test_cache_dir):
    """Arm-attribution (the regex guard could not do this): an abstract where a COMPARATOR
    (ipragliflozin) 'significantly improved' steatosis while metformin 'showed minimal change' must
    be CONTRADICTING for metformin — the benefit belongs to the other drug. The flat-negative
    metformin RCT is also contradicting. So neither lands in supporting and the pair is contradicts."""
    s = await _synthesize(
        _STEATOSIS_ARM_ATTRIBUTION_ABSTRACTS,
        drug="metformin",
        indication="Hepatic Steatosis",
        cache_dir=test_cache_dir,
    )
    # The comparator-benefit abstract must NOT be credited to metformin.
    assert "39806556" not in s.supporting_pmids, (
        f"comparator's benefit credited to metformin: {s.supporting_pmids}"
    )
    assert "39806556" in s.contradicting_pmids
    # The flat-negative metformin RCT is contradicting too.
    assert "19811343" in s.contradicting_pmids
    assert "19811343" not in s.supporting_pmids
    assert s.direction == "contradicts", f"expected contradicts, got {s.direction!r}"
    _assert_self_consistent(s)


# ---- Approval-aware exclusion (PLAN_approval_aware_relevance §C) ----
# Real abstracts (pgvector pubmed_abstracts), the same PMIDs validated 11/11 in
# scratch/approval_aware_literature_harness.py. A paper about an APPROVED sub-indication of a broad
# candidate is evidence_basis="approved" (excluded from repurposing strength), NOT drug_specific.

# Two bupropion-in-MDD RCTs. MDD is an APPROVED bupropion indication; under the broad "mood
# disorder" candidate this is already-approved evidence, not repurposing of the broader term.
_BUPROPION_MDD_ABSTRACTS = [
    {
        "pmid": "31301615",
        "title": (
            "Efficacy and safety of bupropion hydrochloride extended-release versus escitalopram "
            "oxalate in Chinese patients with major depressive disorder: Results from a "
            "randomized, double-blind, non-inferiority trial."
        ),
        "abstract": (
            "BACKGROUND: This study evaluated the non-inferiority of bupropion extended-release "
            "(XL) compared to escitalopram for acute-phase treatment of Chinese patients with "
            "major depressive disorder (MDD). METHODS: This randomized (1:1), double-blind, "
            "active-control study included patients with MDD (DSM-IV) (N=538) treated with "
            "bupropion XL or escitalopram. Primary outcome was mean change from baseline in "
            "HAMD-17 total score at Week 8. RESULTS: The response rate was 69.6% versus 72.9% for "
            "bupropion XL versus escitalopram. CONCLUSION: The efficacy of bupropion XL was "
            "non-inferior to escitalopram in Chinese patients with MDD."
        ),
    },
    {
        "pmid": "25124683",
        "title": (
            "The effect of bupropion XL and escitalopram on memory and functional outcomes in "
            "adults with major depressive disorder: results from a randomized controlled trial."
        ),
        "abstract": (
            "In this study we sought to determine the effect of escitalopram and bupropion XL on "
            "memory and psychosocial function in Major Depressive Disorder (MDD). Forty-one "
            "individuals with MDD were enrolled in an 8-week, double-blind, randomized controlled "
            "comparative trial of bupropion XL and escitalopram. Treatment with either drug was "
            "associated with improvement in memory and psychosocial function in adults with MDD."
        ),
    },
]

# Bupropion in bipolar/cyclic mood disorder — the candidate's genuinely NON-approved scope (bipolar
# is NOT an approved bupropion indication), so the approved-exclusion must NOT fire here.
_BUPROPION_BIPOLAR_ABSTRACTS = [
    {
        "pmid": "2856918",
        "title": (
            "Bupropion in the long-term treatment of cyclic mood disorders: mood stabilizing "
            "effects."
        ),
        "abstract": (
            "The effects of bupropion HCl in treating 11 patients with bipolar or schizoaffective "
            "disorder were examined in an open trial. All patients were maintained on bupropion "
            "alone or in combination with low-dose neuroleptics or anxiolytics for 1 year or more, "
            "with little or no relapse and few side effects."
        ),
    },
]

_BUPROPION_MOOD_APPROVED = [
    "major depressive disorder",
    "seasonal affective disorder",
    "smoking cessation",
]


@pytest.mark.approval_aware
async def test_bupropion_mdd_papers_under_mood_disorder_are_approved_not_strong(
    test_cache_dir,
):
    """Bupropion-in-MDD RCTs under the broad 'mood disorder' candidate, with MDD in the approved
    list → evidence_basis='approved' (already-approved evidence), strength capped (not
    strong/moderate). This is the leak the plan fixes."""
    s = await _synthesize(
        _BUPROPION_MDD_ABSTRACTS,
        drug="bupropion",
        indication="Mood Disorder",
        cache_dir=test_cache_dir,
        approved_indications=_BUPROPION_MOOD_APPROVED,
    )
    assert (
        s.evidence_basis == "approved"
    ), f"expected approved, got {s.evidence_basis!r}"
    # the strength cap forces strength/direction to none for the approved basis
    assert (
        s.strength == "none"
    ), f"approved-sub-indication evidence graded {s.strength!r}"
    assert s.direction == "none"
    _assert_self_consistent(s)


@pytest.mark.approval_aware
async def test_bupropion_bipolar_paper_under_mood_disorder_stays_drug_specific(
    test_cache_dir,
):
    """CONTROL: with the SAME approved list, a bupropion-in-bipolar paper (bipolar is NOT approved)
    is the candidate's non-approved scope → drug_specific. Proves the exclusion targets the
    approved sub-indication, not the whole 'mood disorder' candidate."""
    s = await _synthesize(
        _BUPROPION_BIPOLAR_ABSTRACTS,
        drug="bupropion",
        indication="Mood Disorder",
        cache_dir=test_cache_dir,
        approved_indications=_BUPROPION_MOOD_APPROVED,
    )
    assert (
        s.evidence_basis == "drug_specific"
    ), f"expected drug_specific, got {s.evidence_basis!r}"
    assert s.strength != "none", f"genuine bipolar evidence graded none: {s!r}"
    _assert_self_consistent(s)


# Empagliflozin-in-CKD RCTs. "Kidney disease" is the broad candidate; chronic kidney disease (CKD)
# is the APPROVED empagliflozin indication. The DECISIVE list-driven control: the SAME papers grade
# 'approved' WITH approved={CKD} and 'drug_specific' WITHOUT — proving the exclusion is driven by
# the approved list, not invented by the model.
_EMPAGLIFLOZIN_CKD_ABSTRACTS = [
    {
        "pmid": "36331190",
        "title": "Empagliflozin in Patients with Chronic Kidney Disease.",
        "abstract": (
            "BACKGROUND: The EMPA-KIDNEY trial was designed to assess the effects of empagliflozin "
            "in a broad range of patients with chronic kidney disease at risk for progression. "
            "Patients were randomly assigned to empagliflozin (10 mg once daily) or placebo. "
            "RESULTS: Progression of kidney disease or death from cardiovascular causes occurred "
            "in 13.1% of the empagliflozin group versus 16.9% of placebo (hazard ratio 0.72). "
            "CONCLUSIONS: Empagliflozin led to a lower risk of progression of kidney disease or "
            "cardiovascular death than placebo."
        ),
    },
    {
        "pmid": "39453837",
        "title": "Long-Term Effects of Empagliflozin in Patients with Chronic Kidney Disease.",
        "abstract": (
            "BACKGROUND: In EMPA-KIDNEY, empagliflozin had positive cardiorenal effects in "
            "patients with chronic kidney disease. Post-trial follow-up assessed how the effects "
            "evolved after discontinuation. During the combined active- and post-trial periods, a "
            "primary-outcome event (kidney disease progression or cardiovascular death) occurred "
            "in 26.2% of the empagliflozin group versus 30.3% of placebo (hazard ratio 0.79). "
            "CONCLUSIONS: Empagliflozin continued to have cardiorenal benefits for up to 12 months "
            "after discontinuation."
        ),
    },
]


@pytest.mark.approval_aware
async def test_empagliflozin_ckd_excluded_only_when_approved_list_supplied(
    test_cache_dir,
):
    """DECISIVE list-driven control. The SAME empagliflozin-in-CKD RCTs under the broad 'kidney
    disease' candidate flip basis purely on whether CKD is in the approved list:
      - approved=[CKD] → 'approved' (already-approved evidence), strength capped to none.
      - approved=[]    → 'drug_specific' (CKD is the candidate scope), real strength.
    Proves the approved-exclusion is driven by the list, not the model."""
    with_list = await _synthesize(
        _EMPAGLIFLOZIN_CKD_ABSTRACTS,
        drug="empagliflozin",
        indication="Kidney Disease",
        cache_dir=test_cache_dir,
        approved_indications=["chronic kidney disease", "type 2 diabetes mellitus"],
    )
    without_list = await _synthesize(
        _EMPAGLIFLOZIN_CKD_ABSTRACTS,
        drug="empagliflozin",
        indication="Kidney Disease",
        cache_dir=test_cache_dir,
        approved_indications=[],
    )
    assert (
        with_list.evidence_basis == "approved"
    ), f"with CKD approved, expected approved, got {with_list.evidence_basis!r}"
    assert with_list.strength == "none"
    assert (
        without_list.evidence_basis == "drug_specific"
    ), f"with no approved list, expected drug_specific, got {without_list.evidence_basis!r}"
    assert (
        without_list.strength != "none"
    ), f"CKD RCTs graded none without list: {without_list!r}"
    _assert_self_consistent(with_list)
    _assert_self_consistent(without_list)


# Real murine adalimumab-asthma abstract (PMID 24882395) — the sole support for the humira×asthma
# candidate. A mouse OVA model; no human data → is_animal_only must be True.
_ASTHMA_MURINE_ABSTRACT = {
    "pmid": "24882395",
    "title": (
        "A human monoclonal anti-TNF alpha antibody (adalimumab) reduces airway inflammation "
        "and ameliorates lung histology in a murine model of acute asthma."
    ),
    "abstract": (
        "BACKGROUND: A few experimental studies related to asthma have unveiled the beneficial "
        "effects of TNF alpha blocking agents on the airway histology, cytokine levels in "
        "bronchoalveolar lavage and bronchial hyper-responsiveness. In the current study, we "
        "aimed to assess the effect of adalimumab on the inflammation and histology of asthma "
        "in a murine model. METHOD: Twelve-week-old BALB/c female rats (n=18) were allocated "
        "into three groups: control, OVA-sensitized untreated, and OVA-sensitized adalimumab-"
        "treated. RESULTS: Adalimumab significantly reduced peribronchial inflammation, alveolar "
        "wall thickness and smooth muscle hypertrophy compared with untreated asthmatic mice."
    ),
}

# Real human adalimumab RCT (PMID 16200601, psoriatic arthritis) — a genuine human study for the
# same drug. Paired with the murine abstract, one human study flips is_animal_only to False.
_PSA_HUMAN_RCT = {
    "pmid": "16200601",
    "title": (
        "Adalimumab for the treatment of patients with moderately to severely active psoriatic "
        "arthritis: results of a double-blind, randomized, placebo-controlled trial."
    ),
    "abstract": (
        "OBJECTIVE: Adalimumab, a fully human anti-tumor necrosis factor monoclonal antibody, "
        "was evaluated for safety and efficacy compared with placebo in the treatment of active "
        "psoriatic arthritis (PsA). METHODS: Patients with moderately to severely active PsA were "
        "randomized to receive 40 mg adalimumab or placebo subcutaneously every other week for 24 "
        "weeks. RESULTS: Adalimumab significantly improved the signs and symptoms of arthritis and "
        "psoriasis compared with placebo."
    ),
}


@pytest.mark.approval_aware
async def test_animal_only_murine_asthma_is_true(test_cache_dir):
    """The only relevant adalimumab-asthma study is a murine OVA model → is_animal_only True."""
    s = await _synthesize(
        [_ASTHMA_MURINE_ABSTRACT],
        drug="adalimumab",
        indication="Asthma",
        cache_dir=test_cache_dir,
    )
    assert s.is_animal_only is True, f"murine-only asthma graded {s.is_animal_only!r}: {s!r}"
    _assert_self_consistent(s)


@pytest.mark.approval_aware
async def test_animal_only_false_when_relevant_human_rct_present(test_cache_dir):
    """A human adalimumab PsA RCT is relevant human evidence → is_animal_only False. The off-disease
    murine asthma abstract is excluded as contaminated, so the graded evidence is human-only."""
    s = await _synthesize(
        [_ASTHMA_MURINE_ABSTRACT, _PSA_HUMAN_RCT],
        drug="adalimumab",
        indication="Psoriatic Arthritis",
        cache_dir=test_cache_dir,
    )
    assert s.is_animal_only is False, f"human RCT present but graded {s.is_animal_only!r}: {s!r}"
    assert "16200601" in s.relevant_pmids, "human PsA RCT should be relevant"
    _assert_self_consistent(s)
