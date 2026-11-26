"""Integration test for the treats-disease gate — the LIVE LLM that decides whether a paper studies the candidate disease
as a treatment or prevention target. Hits real Anthropic; abstracts are real PubMed text.

A Mendelian randomization study gives nobody the drug: it uses genetic proxies for drug exposure, so it is not prevention
evidence. It passed the gate for metformin x Parkinson disease and was counted against metformin. The retrospective
cohort comparing metformin initiators with DPP-4 inhibitor users on incident parkinsonism is the control: a prevention
outcome in people who took the drug still passes, and the pilot trial of adjunctive metformin is direct treatment.
SUSTAIN 11 recorded severe hypoglycaemia as a safety endpoint in a type 2 diabetes trial; it passed the gate for
semaglutide x hypoglycemia and closed that signal.
"""

import logging
from pathlib import Path

import pytest

from indication_scout.services.retrieval import (
    AbstractResult,
    _judge_pmid_treats_disease,
)

from ._metformin_parkinson_abstracts import DPP4_COHORT, MENDELIAN, PILOT_RCT

logger = logging.getLogger(__name__)

# SUSTAIN 11: semaglutide vs insulin aspart in type 2 diabetes; severe hypoglycaemia is a confirmatory safety endpoint.
_SUSTAIN_11 = {
    "pmid": "35546450",
    "title": "Effect of once-weekly semaglutide versus thrice-daily insulin aspart, both as add-on to metformin and optimized insulin glargine treatment in participants with type 2 diabetes (SUSTAIN 11): A randomized, open-label, multinational, phase 3b trial.",
    "abstract": (
        "AIM: To compare the efficacy and safety of once-weekly (OW) semaglutide versus thrice-daily (TID) "
        "insulin aspart (IAsp) in participants with inadequately controlled type 2 diabetes (T2D) treated "
        "with insulin glargine (IGlar) and metformin. MATERIALS AND METHODS: SUSTAIN 11 (NCT03689374) was a "
        "randomized (1:1), parallel, open-label, multinational, phase 3b trial. After a 12-week run-in to "
        "optimize once-daily IGlar U100, 1748 adults with T2D (HbA1c >7.5% to ≤10.0%) were randomized to OW "
        "semaglutide or TID IAsp as add-on to optimized IGlar and metformin for 52\u2009weeks. The primary outcome"
        " was change in HbA1c from randomization to week 52. Confirmatory secondary endpoints included the "
        "occurrence of severe hypoglycaemic episodes and change in body weight (BW). Safety was assessed. "
        "RESULTS: HbA1c (randomization: 8.6% [70.0\u2009mmol/mol]) decreased by 1.5% points (16.6\u2009mmol/mol) and "
        "1.2% points (13.4\u2009mmol/mol) with semaglutide (n\xa0=\xa0874) and IAsp (n\xa0=\xa0874), respectively (estimated "
        "treatment difference [ETD] -0.29% points [95% confidence interval {CI} -0.38; -0.20]; P\xa0<\u2009.0001 for "
        "non-inferiority). Few severe hypoglycaemic episodes were recorded in either group, with no "
        "statistically significant difference between the groups. Change in BW from randomization (87.9\xa0kg) "
        "to week 52 was in favour of semaglutide (-4.1\xa0kg) versus IAsp (+2.8\xa0kg) (ETD -6.99\u2009kg [95% CI -7.41;"
        " -6.57]). A higher proportion of participants experienced adverse events with semaglutide (58.5%) "
        "versus IAsp (52.1%); most were mild to moderate. CONCLUSIONS: In this basal insulin-treated "
        "population, OW semaglutide improved glycaemic control to a greater extent than TID IAsp and provided"
        " numerically greater weight loss."
    ),
}

pytestmark = pytest.mark.asyncio(loop_scope="module")


@pytest.mark.parametrize(
    "drug, disease, abstracts, expected",
    [
        (
            "metformin",
            "Parkinson disease",
            [MENDELIAN, DPP4_COHORT, PILOT_RCT],
            {"42627515": False, "42216231": True, "40385486": True},
        ),
        ("semaglutide", "hypoglycemia", [_SUSTAIN_11], {"35546450": False}),
    ],
)
async def test_pmid_treats_disease(
    drug: str,
    disease: str,
    abstracts: list[dict[str, str]],
    expected: dict[str, bool],
    tmp_path: Path,
) -> None:
    """A Mendelian randomization study and a safety endpoint the drug class can cause are not treatment or prevention
    targets; a prevention outcome in people who took the drug, and direct treatment, are.
    """
    results = [
        AbstractResult(
            pmid=d["pmid"], title=d["title"], abstract=d["abstract"], similarity=0.9
        )
        for d in abstracts
    ]

    decisions = await _judge_pmid_treats_disease(
        "CHEMBL_TEST", drug, disease, results, tmp_path
    )

    assert decisions == expected
