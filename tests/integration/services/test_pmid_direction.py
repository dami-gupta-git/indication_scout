"""Integration test for the per-PMID direction sub-call — the LIVE LLM that grades each relevant
abstract as supporting / contradicting / mixed / neutral for one drug-disease pair. Hits real
Anthropic; abstracts are REAL PubMed text embedded inline so the test is self-contained.

Guards the "mixed" definition and the own-result rule in prompts/pmid_direction.txt against the
metformin x miscarriage regression. PregMet2 MISSED its primary endpoint (OR 0.50, p=0.08) while
the post-hoc pooled analysis reported in the SAME abstract was significant (OR 0.43, p=0.004). It
was graded "contradicting"; that was the only non-negative literature vote for miscarriage, so the
deterministic aggregation in retrieval.py produced a "contradicts" card and closed the candidate.

The whole batch must be graded in ONE call, as production does. The verdict is sensitive to batch
composition: graded alone, or paired with only EMPOWaR, PregMet2 returns a different verdict than
it does in the real three-abstract batch, so a per-abstract test does not reproduce the bug.

ACCELERATE is the counterweight: an adequately powered trial that showed no effect must stay
"contradicting", so neither the broadened "mixed" definition nor the narrowed own-result rule pulls
well-powered nulls out of the contradicting bucket.

Verdicts are LLM judgment, so each PMID asserts an allowed set rather than one string. PregMet2's
set excludes "contradicting", which is the regression being guarded.
"""

import logging

import pytest

from indication_scout.services.retrieval import AbstractResult, _judge_pmid_directions

logger = logging.getLogger(__name__)

_PREGMET2 = {
    "pmid": "30792154",
    "title": (
        "Use of metformin to treat pregnant women with polycystic ovary syndrome (PregMet2): a randomised, double- "
        "blind, placebo-controlled trial. "
    ),
    "abstract": (
        "BACKGROUND: Women with polycystic ovary syndrome (PCOS) have an increased risk of pregnancy complications. "
        "Epi-analysis of two previous randomised controlled trials that compared metformin with placebo during "
        "pregnancy in women with PCOS showed a significant reduction in late miscarriages and preterm births in the "
        "metformin group. The aim of this third randomised trial (PregMet2) was to test the hypothesis that "
        "metformin prevents late miscarriage and preterm birth in women with PCOS. METHODS: PregMet2 was a "
        "randomised, placebo-controlled, double-blind, multicentre trial done at 14 hospitals in Norway, Sweden, "
        "and Iceland. Singleton pregnant women with PCOS aged 18-45 years were eligible for inclusion. After "
        "receiving information about the study at their first antenatal visit or from the internet, women signed up "
        "individually to participate in the study. Participants were randomly assigned (1:1) to receive metformin "
        "or placebo by computer-generated random numbers. Randomisation was in blocks of ten for each country and "
        "centre; the first block had a random size between one and ten to assure masking. Participants were "
        "assigned to receive oral metformin 500 mg twice daily or placebo during the first week of treatment, which "
        "increased to 1000 mg twice daily or placebo from week 2 until delivery. Placebo tablets and metformin "
        "tablets were identical and participants and study personnel were masked to treatment allocation. The "
        "primary outcome was the composite incidence of late miscarriage (between week 13 and week 22 and 6 days) "
        "and preterm birth (between week 23 and week 36 and 6 days), analysed in the intention-to-treat population. "
        "Secondary endpoints included the incidence of gestational diabetes, preeclampsia, pregnancy-induced "
        "hypertension, and admission of the neonate to the neonatal intensive care unit. We also did a post-hoc "
        "individual participant data analysis of pregnancy outcomes, pooling data from the two previous trials with "
        "the present study. The study was registered with ClinicalTrials.gov, number NCT01587378, and EudraCT, "
        "number 2011-002203-15. FINDINGS: The study took place between Oct 19, 2012, and Sept 1, 2017. We randomly "
        "assigned 487 women to metformin (n=244) or placebo (n=243). In the intention-to-treat analysis, our "
        "composite primary outcome of late miscarriage and preterm birth occurred in 12 (5%) of 238 women in the "
        "metformin group and 23 (10%) of 240 women in the placebo group (odds ratio [OR] 0·50, 95% CI 0·22-1·08; "
        "p=0·08). We found no significant differences for our secondary endpoints, including incidence of "
        "gestational diabetes (60 [25%] of 238 women in the metformin group vs 57 [24%] of 240 women in the placebo "
        "group; OR 1·09, 95% CI 0·69-1·66; p=0·75). We noted no substantial between-group differences in serious "
        "adverse events in either mothers or offspring, and no serious adverse events were considered drug-related "
        "by principal investigators. In the post-hoc pooled analysis of individual participant data from the "
        "present trial and two previous trials, 18 (5%) of 397 women had late miscarriage or preterm delivery in "
        "the metformin group compared with 40 (10%) of 399 women in the placebo group (OR 0·43, 95% CI 0·23-0·79; "
        "p=0·004). INTERPRETATION: In pregnant women with PCOS, metformin treatment from the late first trimester "
        "until delivery might reduce the risk of late miscarriage and preterm birth, but does not prevent "
        "gestational diabetes. FUNDING: Research Council of Norway, Novo Nordisk Foundation, St Olav's University "
        "Hospital, and Norwegian University of Science and Technology. "
    ),
}

_EMPOWAR = {
    "pmid": "26165398",
    "title": (
        "Effect of metformin on maternal and fetal outcomes in obese pregnant women (EMPOWaR): a randomised, "
        "double-blind, placebo-controlled trial. "
    ),
    "abstract": (
        "BACKGROUND: Maternal obesity is associated with increased birthweight, and obesity and premature mortality "
        "in adult offspring. The mechanism by which maternal obesity leads to these outcomes is not well "
        "understood, but maternal hyperglycaemia and insulin resistance are both implicated. We aimed to establish "
        "whether the insulin sensitising drug metformin improves maternal and fetal outcomes in obese pregnant "
        "women without diabetes. METHODS: We did this randomised, double-blind, placebo-controlled trial in "
        "antenatal clinics at 15 National Health Service hospitals in the UK. Pregnant women (aged ≥16 years) "
        "between 12 and 16 weeks' gestation who had a BMI of 30 kg/m(2) or more and normal glucose tolerance were "
        "randomly assigned (1:1), via a web-based computer-generated block randomisation procedure (block size of "
        "two to four), to receive oral metformin 500 mg (increasing to a maximum of 2500 mg) or matched placebo "
        "daily from between 12 and 16 weeks' gestation until delivery of the baby. Randomisation was stratified by "
        "study site and BMI band (30-39 vs ≥40 kg/m(2)). Participants, caregivers, and study personnel were masked "
        "to treatment assignment. The primary outcome was Z score corresponding to the gestational age, parity, and "
        "sex-standardised birthweight percentile of liveborn babies delivered at 24 weeks or more of gestation. We "
        "did analysis by modified intention to treat. This trial is registered, ISRCTN number 51279843. FINDINGS: "
        "Between Feb 3, 2011, and Jan 16, 2014, inclusive, we randomly assigned 449 women to either placebo (n=223) "
        "or metformin (n=226), of whom 434 (97%) were included in the final modified intention-to-treat analysis. "
        "Mean birthweight at delivery was 3463 g (SD 660) in the placebo group and 3462 g (548) in the metformin "
        "group. The estimated effect size of metformin on the primary outcome was non-significant (adjusted mean "
        "difference -0·029, 95% CI -0·217 to 0·158; p=0·7597). The difference in the number of women reporting the "
        "combined adverse outcome of miscarriage, termination of pregnancy, stillbirth, or neonatal death in the "
        "metformin group (n=7) versus the placebo group (n=2) was not significant (odds ratio 3·60, 95% CI "
        "0·74-17·50; p=0·11). INTERPRETATION: Metformin has no significant effect on birthweight percentile in "
        "obese pregnant women. Further follow-up of babies born to mothers in the EMPOWaR study will identify "
        "longer-term outcomes of metformin in this population; in the meantime, metformin should not be used to "
        "improve pregnancy outcomes in obese women without diabetes. FUNDING: The Efficacy and Mechanism Evaluation "
        "(EME) Programme, a Medical Research Council and National Institute for Health Research partnership. "
    ),
}

_GLYCODELIN = {
    "pmid": "11238496",
    "title": (
        "Insulin reduction with metformin increases luteal phase serum glycodelin and insulin-like growth factor- "
        "binding protein 1 concentrations and enhances uterine vascularity and blood flow in the polycystic ovary "
        "syndrome. "
    ),
    "abstract": (
        "We hypothesized that hyperinsulinemia contributes to early pregnancy loss in the polycystic ovary syndrome "
        "by adversely affecting endometrial function and environment. Serum glycodelin, a putative biomarker of "
        "endometrial function, is decreased in women with early pregnancy loss. Insulin-like growth factor-binding "
        "protein-1 may also play an important role in pregnancy by facilitating adhesion processes at the feto- "
        "maternal interface. We studied 48 women with polycystic ovary syndrome before and after 4 weeks of "
        "administration of 500 mg metformin (n = 26) or placebo (n = 22) 3 times daily. Oral glucose tolerance "
        "tests were performed, and serum glycodelin and insulin-like growth factor-binding protein-1 were measured "
        "during the follicular and clomiphene-induced luteal phases of menses. In the metformin group, the mean "
        "(+/-SE) area under the serum insulin curve after glucose administration decreased from 62 +/- 6 to 19 +/- "
        "2 nmol/L.min (P < 0.001). Follicular phase serum glycodelin concentrations increased 20-fold from 150 +/- "
        "46 to 2813 +/- 1192 pmol/L (P < 0.001), and serum insulin-like-growth factor-binding protein-1 "
        "concentrations increased from 936 +/- 152 to 2396 +/- 300 pmol/L (P < 0.001). Similarly, luteal phase "
        "serum glycodelin concentrations increased 3-fold from 3434 +/- 1299 to 10624 +/- 1803 pmol/L (P < 0.001), "
        "and serum insulin-like growth factor-binding protein-1 concentrations increased from 1220 +/- 136 to 4916 "
        "+/- 596 pmol/L (P < 0.001). Uterine vascular penetration also increased in the metformin group, as did "
        "blood flow of spiral arteries, as demonstrated by a 20% decrease in the resistance index from 0.71 +/- "
        "0.02 to 0.57 +/- 0.03 (P < 0.001). These variables did not change in the placebo group. We conclude that "
        "insulin reduction with metformin increases follicular and luteal phase serum glycodelin and insulin-like "
        "growth factor-binding protein-1 concentrations and enhances luteal phase uterine vascularity and blood "
        "flow in the polycystic ovary syndrome. These changes may reflect an improved endometrial milieu for the "
        "establishment and maintenance of pregnancy. "
    ),
}

_ACCELERATE = {
    "pmid": "28514624",
    "title": (
        "Evacetrapib and Cardiovascular Outcomes in High-Risk Vascular Disease. "
    ),
    "abstract": (
        "BACKGROUND: The cholesteryl ester transfer protein inhibitor evacetrapib substantially raises the high- "
        "density lipoprotein (HDL) cholesterol level, reduces the low-density lipoprotein (LDL) cholesterol level, "
        "and enhances cellular cholesterol efflux capacity. We sought to determine the effect of evacetrapib on "
        "major adverse cardiovascular outcomes in patients with high-risk vascular disease. METHODS: In a "
        "multicenter, randomized, double-blind, placebo-controlled phase 3 trial, we enrolled 12,092 patients who "
        "had at least one of the following conditions: an acute coronary syndrome within the previous 30 to 365 "
        "days, cerebrovascular atherosclerotic disease, peripheral vascular arterial disease, or diabetes mellitus "
        "with coronary artery disease. Patients were randomly assigned to receive either evacetrapib at a dose of "
        "130 mg or matching placebo, administered daily, in addition to standard medical therapy. The primary "
        "efficacy end point was the first occurrence of any component of the composite of death from cardiovascular "
        "causes, myocardial infarction, stroke, coronary revascularization, or hospitalization for unstable angina. "
        "RESULTS: At 3 months, a 31.1% decrease in the mean LDL cholesterol level was observed with evacetrapib "
        "versus a 6.0% increase with placebo, and a 133.2% increase in the mean HDL cholesterol level was seen with "
        "evacetrapib versus a 1.6% increase with placebo. After 1363 of the planned 1670 primary end-point events "
        "had occurred, the data and safety monitoring board recommended that the trial be terminated early because "
        "of a lack of efficacy. After a median of 26 months of evacetrapib or placebo, a primary end-point event "
        "occurred in 12.9% of the patients in the evacetrapib group and in 12.8% of those in the placebo group "
        "(hazard ratio, 1.01; 95% confidence interval, 0.91 to 1.11; P=0.91). CONCLUSIONS: Although the cholesteryl "
        "ester transfer protein inhibitor evacetrapib had favorable effects on established lipid biomarkers, "
        "treatment with evacetrapib did not result in a lower rate of cardiovascular events than placebo among "
        "patients with high-risk vascular disease. (Funded by Eli Lilly; ACCELERATE ClinicalTrials.gov number, "
        "NCT01687998 .). "
    ),
}


_SILDENAFIL_RATS = {
    "pmid": "12411660",
    "title": "Sildenafil (Viagra) induces neurogenesis and promotes functional recovery after stroke in rats. ",
    "abstract": (
        "BACKGROUND AND PURPOSE: We tested the hypothesis that sildenafil, a phosphodiesterase type 5 (PDE5) "
        "inhibitor, promotes functional recovery and neurogenesis after stroke. METHODS: Male Wistar rats were "
        "subjected to embolic middle cerebral artery occlusion. Sildenafil (Viagra) was administered orally for 7 "
        "consecutive days starting 2 or 24 hours after stroke onset at doses of 2 or 5 mg/kg per day. Ischemic rats "
        "administered the same volume of tap water were used as a control group. Functional outcome tests "
        "(foot-fault, adhesive removal) were performed. Rats were killed 28 days after stroke for analysis of "
        "infarct volume and newly generated cells within the subventricular zone and the dentate gyrus. RESULTS: "
        "Treatment with sildenafil significantly (P<0.05) enhanced neurological recovery in all tests performed. "
        "There was no significant difference of infarct volume among the experimental groups. Treatment with "
        "sildenafil significantly (P<0.05) increased numbers of bromodeoxyuridine-immunoreactive cells in the "
        "subventricular zone and the dentate gyrus. CONCLUSIONS: Sildenafil increases brain levels of cGMP, evokes "
        "neurogenesis, and reduces neurological deficits when given to rats 2 or 24 hours after stroke. "
    ),
}

_SILDENAFIL_GERBILS = {
    "pmid": "39335590",
    "title": (
        "Effects of Sildenafil on Cognitive Function Recovery and Neuronal Cell Death Protection after Transient "
        "Global Cerebral Ischemia in Gerbils. "
    ),
    "abstract": (
        "Cerebral ischemic stroke is a major cause of death worldwide due to brain cell death resulting from "
        "ischemia-reperfusion injury. However, effective treatment approaches for patients with ischemic stroke are "
        "still lacking in clinical practice. This study investigated the potential neuroprotective effects of "
        "sildenafil, a phosphodiesterase-5 inhibitor, in a gerbil model of global brain ischemia. We investigated "
        "the effects of sildenafil on the expression of glial fibrillary acidic protein and aquaporin-4. "
        "Immunofluorescence analysis showed that the number of cells co-expressing these markers, which was "
        "elevated in the ischemia-induced group, was significantly reduced in the sildenafil-treated groups. "
        "Additionally, we performed various behavioral tests, including the open-field test, novel object "
        "recognition, Barnes maze, Y-maze, and passive avoidance tests, to evaluate sildenafil's effect on "
        "cognitive function impaired by ischemia. Overall, the results suggest that sildenafil may serve as a "
        "neuroprotective agent, potentially alleviating delayed neuronal cell death and improving cognitive "
        "function impaired by ischemia. "
    ),
}

# The abstract whose BACKGROUND sentence ("compared with placebo", about RATS) certified the pair as RCT-backed under the
# old document-wide phrase match. The study it reports is a single-arm 12-patient safety study.
_SILDENAFIL_SAFETY = {
    "pmid": "19717023",
    "title": "Sildenafil treatment of subacute ischemic stroke: a safety study at 25-mg daily for 2 weeks. ",
    "abstract": (
        "BACKGROUND: In several animal studies of young and aged rats with ischemic stroke, treatment with "
        "sildenafil improved functional outcomes compared with placebo. We conducted a safety study of sildenafil "
        "(25 mg daily for 2 weeks) shortly after ischemic stroke onset. METHODS: We recruited patients aged 18 to "
        "80 years with ischemic stroke, National Institutes of Health stroke scale (NIHSS) score 2 to 21, between "
        "days 2 and 9 after symptom onset. Patients were treated with sildenafil for 2 weeks (25 mg daily). The "
        "primary outcome measure was the adverse occurrence of any of the following during the treatment period: "
        "stroke worsening, new stroke, myocardial infarction, vision loss, hearing loss, or death from any cause. "
        "RESULTS: Twelve patients were recruited. Mean age was 57 years, 5 were female, and median NIHSS score at "
        "entry was 9.5 (range 2-20). The primary outcome measure occurred in one patient (sudden death). Among the "
        "10 survivors, at 90 days, median NIHSS score was 2 (range 0-12), median Barthel index was 95 (range "
        "15-100), and median modified Rankin score was 1.5 (range 0-5). CONCLUSIONS: Sildenafil (25 mg daily for 2 "
        "weeks) appeared to be safe in this group of patients with mild to moderately severe stroke. "
    ),
}


def _to_abstracts(dicts: list[dict[str, str]]) -> list[AbstractResult]:
    """Wrap the inline {pmid,title,abstract} dicts as AbstractResults the sub-call accepts."""
    return [
        AbstractResult(
            pmid=d["pmid"], title=d["title"], abstract=d["abstract"], similarity=0.9
        )
        for d in dicts
    ]


@pytest.mark.parametrize(
    "drug, disease, abstracts, expected",
    [
        (
            "metformin",
            "Miscarriage",
            [_PREGMET2, _EMPOWAR, _GLYCODELIN],
            {
                "30792154": {"mixed", "supporting"},
                "26165398": {"contradicting"},
                "11238496": {"neutral"},
            },
        ),
        (
            "evacetrapib",
            "Coronary Artery Disease",
            [_ACCELERATE],
            {"28514624": {"contradicting"}},
        ),
    ],
)
async def test_pmid_direction(
    drug: str,
    disease: str,
    abstracts: list[dict[str, str]],
    expected: dict[str, set[str]],
) -> None:
    """Grade one real batch for one drug-disease pair against the live sub-call."""
    judgments = await _judge_pmid_directions(drug, disease, _to_abstracts(abstracts))

    assert set(judgments) == set(expected)
    for pmid, allowed in expected.items():
        assert (
            judgments[pmid].verdict in allowed
        ), f"{pmid} graded {judgments[pmid].verdict}"


@pytest.mark.parametrize(
    "drug, disease, abstracts, expected",
    [
        (
            "sildenafil",
            "Ischemic Stroke",
            [_SILDENAFIL_RATS, _SILDENAFIL_GERBILS, _SILDENAFIL_SAFETY],
            {
                # The rodent studies ARE controlled (a tap-water control group); what disqualifies them is the species,
                # so is_controlled is left unasserted for them.
                "12411660": (False, None),
                "39335590": (False, None),
                "19717023": (True, False),
            },
        ),
        (
            "metformin",
            "Miscarriage",
            [_PREGMET2],
            {"30792154": (True, True)},
        ),
    ],
)
async def test_pmid_design(
    drug: str,
    disease: str,
    abstracts: list[dict[str, str]],
    expected: dict[str, tuple[bool, bool | None]],
) -> None:
    """The design fields must describe the study each abstract REPORTS, not one it cites. The sildenafil safety abstract
    opens by describing placebo-controlled RAT experiments; the study it reports is a single-arm 12-patient safety study,
    and grading it controlled is what rendered a rodent-graded candidate as "RCT-backed / controlled". A rodent study is
    never human, whatever the disease it models. PregMet2 is the positive control: a real placebo-controlled human trial
    must still read human and controlled.
    """

    judgments = await _judge_pmid_directions(drug, disease, _to_abstracts(abstracts))

    assert set(judgments) == set(expected)
    for pmid, (is_human, is_controlled) in expected.items():
        assert judgments[pmid].is_human is is_human, f"{pmid} is_human"
        if is_controlled is not None:
            assert (
                judgments[pmid].is_controlled is is_controlled
            ), f"{pmid} is_controlled"

    # The claim the report actually makes: a pair may render as "RCT-backed / controlled" only if some abstract is BOTH
    # human and controlled. Only the metformin batch qualifies.
    certifies_controlled = any(
        judgment.is_human and judgment.is_controlled for judgment in judgments.values()
    )
    assert certifies_controlled is (drug == "metformin")
