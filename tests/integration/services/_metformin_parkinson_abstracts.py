"""Real PubMed abstracts (title and abstract verbatim) for the per-paper grade and treats-disease gate tests on the
metformin x Parkinson disease closure (2026-11-11 report). Every paper counted against metformin there was observational,
the one trial was a 60-patient pilot, and the candidate was still closed as a signal. The imatinib glioblastoma trial is
the counterweight: a failed single-arm trial in which the drug was given, which must still close.
"""

# Randomized placebo-controlled 60-patient, 3-month pilot of adjunctive metformin in Parkinson's disease; missed UPDRS, biomarkers moved.
PILOT_RCT = {
    "pmid": "40385486",
    "title": "Randomized, double-blind, placebo-controlled pilot study of metformin as an adjunctive therapy in Parkinson's disease.",
    "abstract": (
        "BACKGROUND: Parkinson's disease (PD) is caused by the progressive loss of dopaminergic neurons in the "
        "substantia nigra. Neuroinflammation is considered a key factor contributing to the pathophysiology of PD. "
        "Current gold-standard therapies for PD provide only symptomatic relief without slowing disease progression, "
        "highlighting the need to develop new disease-modifying treatments. Metformin has been demonstrated to exert a "
        "neuroprotective role in several neurodegenerative disorders including PD. AIM: This study aimed to clarify the"
        " role of metformin as adjuvant therapy in patients with PD. METHODS: Sixty patients with PD were divided into "
        "2 groups (n = 30). Patients in group 1 received levodopa/carbidopa (250/25 mg) three times daily for 3 months "
        "plus placebo (Control group), while those in group 2 received levodopa/carbidopa (250/25 mg) three times daily"
        " and 500 mg metformin two times daily (Metformin group). Patients were assessed via Unified Parkinson's "
        "Disease Rating Scale (UPDRS). The serum concentrations of toll like receptor 4 (TLR-4), α-synuclein, brain "
        "derived neurotropic factor (BDNF), and high mobility group box 1 (HMGB-1) were measured before and after "
        "treatment. PRIMARY OUTCOME: The improvement in UPDRS from baseline to 3 months. SECONDARY OUTCOME: Change in "
        "the level of biological markers. RESULTS: The control group did not show significant difference in UPDRS when "
        "compared to their baseline value by Wilcoxon test (P > 0.05), meanwhile the metformin group showed significant"
        " difference when compared to before treatment by Wilcoxon test (P < 0.05). There were no significant "
        "differences between the two groups in UPDRS after treatment (P > 0.05) by Man Whitney test. However, the "
        "metformin group showed a significant decrease in TLR-4, HMGB-1, and α-synuclein along with a statistically "
        "significant increase in BDNF (P < 0.05) when compared to its baseline and control group. The control group did"
        " not show any significant changes in all markers when compared to their baseline. CONCLUSION: While no "
        "significant differences in UPDRS scores were observed between the metformin and control groups, trends in "
        "biomarker changes suggest a potential impact of adjunctive metformin use on the underlying pathophysiology of "
        "PD. Further studies are needed to assess its effects on motor symptoms over a longer duration. CLINICAL TRIAL "
        "REGISTRATION: identifier NCT05781711."
    ),
}

# Retrospective active-comparator cohort: metformin initiators vs DPP-4 inhibitor users, incident parkinsonism; no overall difference.
DPP4_COHORT = {
    "pmid": "42216231",
    "title": "Metformin versus DPP-4 inhibitors and risk of parkinsonism in type 2 diabetes: an active-comparator cohort study with a landmark design.",
    "abstract": (
        "BACKGROUND: Type 2 diabetes mellitus (T2DM) is a significant risk factor for Parkinson's disease (PD). While "
        "metformin shows neuroprotective potential in preclinical models, clinical evidence remains conflicting, likely"
        " due to limited follow-up, comparator heterogeneity, and insufficient consideration of differential survival. "
        "METHODS: We conducted a retrospective cohort study (observing data from 2005 to 2025) using the TriNetX global"
        " network. Using an incident user design and a 1-year landmark period to ensure sustained therapy and minimize "
        "immortal time bias, we matched 75,535 metformin initiators with 75,535 dipeptidyl peptidase-4 inhibitor "
        "(DPP-4i) users via propensity scores. The primary outcome was a composite of new-onset idiopathic PD and "
        "secondary parkinsonism. Exploratory time-stratified landmark analyses assessed whether the association "
        "differed among patients alive and outcome-free at 5 and 10 years. RESULTS: Following the landmark period, over"
        " a median follow-up of 5.2 years (IQR, 2.8-8.4; total 785,564 person-years), the overall risk of the composite"
        " neurologic outcome was similar between groups (aHR, 0.97; 95% CI, 0.87-1.09). Metformin users had lower "
        "observed all-cause mortality than DPP-4 inhibitor users (aHR, 0.75; 95% CI, 0.73-0.77). In a 5-year landmark "
        "sensitivity analysis, no association was observed (aHR, 0.98; 95% CI, 0.88-1.09). In exploratory analyses, a "
        "lower observed risk was seen among individuals surviving beyond 10 years (aHR, 0.85; 95% CI, 0.75-0.97; "
        "p = 0.0152), but this analysis was conditioned on long-term survival and should be interpreted as exploratory "
        "and hypothesis-generating. CONCLUSIONS: The primary analysis showed no overall difference in parkinsonism "
        "risk. Metformin users had lower observed all-cause mortality, which may reflect residual confounding or "
        "treatment-selection mechanisms. The lower observed parkinsonism risk among patients surviving beyond 10 years "
        "was exploratory and hypothesis-generating, and may be affected by survivor or collider bias."
    ),
}

# Mendelian randomization on GWAS summary data plus a FAERS disproportionality analysis; nobody was given metformin.
MENDELIAN = {
    "pmid": "42627515",
    "title": "Exploring the impacts of type 2 diabetes and metformin use on Parkinson's disease: Mendelian randomization analysis and a national cohort study.",
    "abstract": (
        "BACKGROUND: The contribution of type 2 diabetes (T2D) and metformin use to Parkinson's disease (PD) remains "
        "unclear. This study aimed to investigate both the overall and independent effects of T2D and metformin use on "
        "PD risk. METHODS: GWAS summary datasets were utilized for the Mendelian randomization (MR) analysis to explore"
        " the causal relationships between T2D, metformin use, and PD, complemented by sensitivity analyses to validate"
        " the robustness of the results. Multivariable MR (MVMR) was applied to assess the direct effects of T2D and "
        "metformin use on PD. Additionally, the FAERS database (2004 Q1-2024 Q2) was analyzed to elucidate the "
        "association of metformin use with Parkinsonism using the reporting odds ratio (ROR). RESULTS: Our MR analysis "
        "showed T2D was not associated with the risk of PD (OR: 1.02, 95% CI: 0.96-1.09, P = 0.47), while metformin use"
        " was associated with an increased risk of PD (OR: 1.08, 95% CI: 1.02-1.14, P = 0.011). The MVMR analysis "
        "demonstrated that T2D was robustly associated with a decreased risk of PD (OR: 0.63, 95% CI: 0.50-0.78, "
        "P = 5.09E-5), while metformin use was linked to a higher risk of PD (OR: 1.63, 95% CI: 1.31-2.03, "
        "P = 1.29E-5). Using FAERS, a total of 179 cases of Parkinsonism were identified from 228,283 reports of "
        "metformin-related adverse events, indicating a significant association of metformin use with Parkinsonism "
        "(ROR: 5.25). CONCLUSION: Our study reveals opposing causal effects of T2D and metformin use on PD, emphasizing"
        " the need to consider metformin's potential adverse effects in PD patients."
    ),
}

# Retrospective cohort testing SGLT2 inhibitors against metformin as the reference drug; reports only the SGLT2 inhibitors' effect.
SGLT2_VS_METFORMIN = {
    "pmid": "40671477",
    "title": "SGLT2 inhibitors vs. metformin for Parkinson's disease risk reduction in type 2 diabetes.",
    "abstract": (
        "SummaryThis study is the first large-scale, head-to-head comparison suggesting that sodium-glucose "
        "cotransporter-2 inhibitors (SGLT2is) may offer greater neuroprotection against Parkinson's disease (PD) "
        "compared to metformin in patients with type 2 diabetes mellitus (T2DM). Utilizing a 20-year real-world dataset"
        " and propensity score matching, we found that SGLT2i users had a 28% lower adjusted hazard ratio (aHR) for PD "
        "(0.72; 95% CI, 0.62-0.84) and reduced all-cause mortality. Unlike previous studies suggesting a potential "
        "increased PD risk with SGLT2is, our robust study design, stringent exclusion criteria, and competing risk "
        "adjustments support a protective association. The findings highlight the need for further prospective research"
        " to explore the neuroprotective benefits of SGLT2is, which may justify prioritizing their use in T2DM patients"
        " at risk for neurodegeneration.BackgroundType 2 diabetes mellitus (T2DM) is linked to an increased risk of "
        "Parkinson's disease (PD), likely mediated by insulin resistance, inflammation, and mitochondrial dysfunction. "
        "While metformin has shown neuroprotective effects, sodium-glucose cotransporter-2 inhibitors (SGLT2is) have "
        "emerging benefits in neurodegeneration. This study provides the first real-world head-to-head comparison of "
        "SGLT2is and metformin on PD risk in T2DM patients.MethodsUsing the TriNetX platform, we analyzed a 20-year "
        "dataset (2005-2025) from 142 healthcare organizations, identifying 913,428 T2DM patients (96,018 SGLT2i, "
        "817,410 metformin users). Patients with prior PD, neurodegenerative diseases, or exposure to "
        "neuroprotective/neurotoxic antidiabetic drugs were excluded. Propensity score matching (1:1) balanced cohorts "
        "across demographic, clinical, and pharmacological variables. Cox proportional hazards models estimated "
        "adjusted hazard ratios (aHRs), validated by positive and negative controls.ResultsSGLT2i use was associated "
        "with a 28% lower PD risk than metformin (aHR = 0.72; 95% CI, 0.62-0.84; p < 0.0001). Dementia, a positive "
        "control, also showed reduced risk (aHR = 0.73; 95% CI, 0.68-0.78; p < 0.0001), reinforcing the neuroprotective"
        " effect. Negative controls confirmed specificity. SGLT2i users had significantly lower all-cause mortality "
        "(aHR = 0.85; 95% CI, 0.83-0.89; p < 0.0001).ConclusionsThis first large-scale comparison suggests SGLT2is "
        "provide superior neuroprotection against PD compared to metformin in T2DM patients, warranting further "
        "investigation. Can Diabetes Medications Help Lower the Risk of Parkinson's Disease?Parkinson's disease is a "
        "brain condition that affects movement and is more likely to occur in people with type 2 diabetes. Some "
        "diabetes medications may offer protection for the brain, but it's not clear which ones work better. In this "
        "large real-world study using electronic health records from over 900,000 patients, we compared two common "
        "diabetes drugs: SGLT2 inhibitors and metformin. We found that people who took SGLT2 inhibitors had a "
        "significantly lower risk of developing Parkinson's disease compared to those who took metformin. This "
        "protective effect was especially clear in older adults, people with poor blood sugar control, and those with "
        "kidney problems. The results remained strong even after adjusting for many other health conditions, "
        "medications, and lifestyle factors such as smoking and alcohol use. Importantly, SGLT2 inhibitor users also "
        "lived longer, which could have affected how often Parkinson's disease was diagnosed. However, we accounted for"
        " this in our analysis and still found a consistent benefit. These findings suggest that SGLT2 inhibitors might"
        " not only help control blood sugar but also protect the brain in people with type 2 diabetes. More research is"
        " needed to understand how these drugs work on the brain and whether they should be considered as first-line "
        "treatment for patients at risk of neurodegenerative diseases like Parkinson's."
    ),
}

# Retrospective cohort testing GLP-1 receptor agonists against metformin as the reference drug; reports only the GLP-1 effect.
GLP1_VS_METFORMIN = {
    "pmid": "41582696",
    "title": "GLP-1RAs versus metformin and Parkinson's risk in type 2 diabetes.",
    "abstract": (
        "BACKGROUND: Parkinson's disease (PD) is an increasingly prevalent neurodegenerative condition, particularly "
        "among individuals with type 2 diabetes mellitus (T2DM). While prior studies have suggested that GLP-1 receptor"
        " agonists (GLP-1RAs) and metformin may confer neuroprotective effects, most were limited by small sample "
        "sizes, short follow-up, uncontrolled designs, or lacked direct comparisons between therapies-making their "
        "findings inconclusive for clinical decision-making. METHODS: Using the TriNetX Global Collaborative Network, "
        "we identified 92 485 patients with T2DM initiating GLP-1RA therapy and matched them 1:1 to new metformin users"
        " using propensity score matching. Patients with prior antidiabetic therapy, PD, or dementia were excluded. "
        "Incident PD was the primary outcome; all-cause mortality served as a secondary endpoint. Adjusted hazard "
        "ratios (aHRs) were estimated using Cox models. Competing risk and time-stratified (≤5, 5-10, >10 years) "
        "analyses were conducted. Positive and negative outcome/exposure controls were employed to validate internal "
        "consistency. RESULTS: The overall PD risk was comparable between GLP-1RA and metformin groups (aHR, 0.91; 95% "
        "CI, 0.79-1.05). However, between years 5 and 10, GLP-1RA use was associated with a significantly lower PD risk"
        " (aHR, 0.56; 95% CI, 0.34-0.93). Mortality risk did not differ significantly. Validation analyses confirmed "
        "the specificity of the findings. CONCLUSIONS: By addressing key limitations of earlier studies through a "
        "large-scale, active-comparator, new-user design, this study provides novel evidence of a delayed "
        "neuroprotective effect of GLP-1RAs. These findings support incorporating neurologic outcomes into long-term "
        "diabetes management and may inform therapy selection in high-risk populations."
    ),
}

# Open-label, nonrandomized phase II trial of imatinib in glioblastoma (51 patients); no measurable activity.
IMATINIB_GBM = {
    "pmid": "31514200",
    "title": "Open-Label Phase II Evaluation of Imatinib in Primary Inoperable or Incompletely Resected and Recurrent Glioblastoma.",
    "abstract": (
        "PURPOSE: Preclinical studies indicated that imatinib may have single-agent activity in glioblastoma through "
        "inhibition of tyrosine kinase activity and also that it might enhance the efficacy of radiotherapy. We "
        "therefore sought to investigate clinical efficacy in patients with newly diagnosed and recurrent glioblastoma "
        "in combination with radiotherapy. METHODS: We conducted a nonrandomized, 2-arm, open-label phase II trial "
        "including patients aged 18 years or older with an ECOG performance status of 0-2 that were either newly "
        "diagnosed (arm A) with a measurable tumor (i.e., after incomplete resection or biopsy) or that were diagnosed "
        "with progression of a glioblastoma after initial therapy (arm B). Patients in arm A received 600 mg/day "
        "imatinib in combination with hypofractionated radiotherapy (2.5 Gy per fraction, 22 fractions). Patients in "
        "arm B received 600 mg/day imatinib alone or in combination with re-irradiation at various doses. In case tumor"
        " progression occurred, CCNU was added (2 cycles, 100 mg/m2) to imatinib. The primary end point was "
        "progression-free survival (PFS). The secondary end point was safety, defined as per Common Terminology "
        "Criteria for Adverse Events (version 2.0). Overall survival (OS) was analyzed as an exploratory end point. "
        "RESULTS: Fifty-one patients were enrolled, of which 19 were included in arm A and 32 in arm B. The median "
        "follow-up was 4 (0.5-30) months in arm A and 6.5 (0.3-51.5) months in arm B. The median PFS was 2.8 months "
        "(95% CI 0-8.7) in arm A and 2.1 months (95% CI 0-11.8) in arm B. The median OS was 5.0 (0.8-30) months (95% CI"
        " 0-24.1) in arm A and 6.5 (0.3-51.5) months (95% CI 0-32.5) in arm B. The major grade 3 events were seizure "
        "(present in 17 patients), pneumonia (11 patients), and vigilance decrease (7 patients). CONCLUSIONS: Imatinib "
        "showed no measurable activity in patients with newly diagnosed or recurrent glioblastoma."
    ),
}
