# Overall report errors

These errors affect candidate stage, evidence grade, closure, or the integrity of the executive summary. The trial-linked
literature lane independently retrieves NCT-linked publications and reserves one of the fifteen shortlist positions. It
addresses the bupropion and alcohol-dependence case pending report-level verification, but it does not correct trial
relevance, non-trial-linked shortlist misses, closure, or executive-summary generation.

## Trial and literature records disagree

The same sildenafil studies are accepted as literature evidence and rejected as clinical trials.

For COVID-19, PMID 34980198 is the publication of NCT04489446. The literature section accepts the publication, while the
clinical-trial section excludes the trial and reports only NCT04304313 as relevant. For ischemic stroke, PMID 19717023 is
the publication of NCT00452582. The literature section accepts the publication as human evidence, while the clinical-trial
section excludes the trial and reports that no relevant trial programme exists.

The trial-linked literature lane does not resolve these cases because it changes literature retrieval and selection rather
than clinical-trial relevance. Both source trials are excluded by a trial gate that does not receive the registry's
primary-purpose field or secondary outcomes. No validation checks whether an accepted PMID identifies an excluded NCT.

### Fix

Add the ClinicalTrials.gov `primaryPurpose` field and secondary outcomes to the `Trial` data contract, parse them with the
existing trial fields, and pass them to the therapeutic-target gate. Revise the gate so a safety or dose-finding study counts
as therapeutic development when its registered primary purpose is treatment and it includes a prespecified disease-efficacy
or recovery outcome. Pure pharmacokinetic, tolerability, and safety-characterization studies without a disease-efficacy
outcome remain excluded.

Increment the trial-target logic version so cached decisions made without these fields are not reused. Verify NCT04489446
and NCT00452582 as relevant, and retain representative pure pharmacokinetic and complication-target trials as exclusions.

After the literature and clinical-trial analyses complete, cross-check relevant PMIDs against the references attached to
excluded NCT records. Re-adjudicate any conflict using the publication abstract together with the trial's registered purpose
and outcomes. If the conflict remains unresolved, report the trial-programme status as unavailable rather than stating that
no relevant programme exists. A linked publication must not automatically make its registry trial relevant.

Affected report: `snapshots/sildenafil_2026-09-09_21-54-36.md`.

## Multi-drug trials can create a false active programme

Bupropion is reported as having an active Phase 3 bipolar-disorder programme based on NCT05973786.

The trial compares an intensified multi-drug strategy with treatment as usual. Bupropion is one of several antidepressants
that may be selected within the experimental strategy, and the trial cannot produce a bupropion-specific treatment effect.
Counting it as a bupropion Phase 3 programme materially overstates the candidate's development stage and supports its
second-place ranking.

The trial relevance step accepts the presence of the drug in a heterogeneous treatment arm without requiring an arm or
comparison that isolates the drug's effect. The trial-linked literature lane does not change trial relevance.

### Fix

Add a deterministic arm-comparison check to clinical-trial finalization. A trial may support an active programme only when
its design isolates the target drug's effect. This includes a target-drug-only arm and a matched add-on comparison in which
the target drug is the only treatment difference between otherwise equivalent background-therapy arms. Placebo and sham
interventions do not count as additional active drugs.

If the target drug appears only within a heterogeneous multi-drug strategy and no matched comparison isolates it, classify
the trial as contaminated and exclude it from programme stage and ranking. If arm composition is unavailable, the trial must
not support an active-programme claim. Verify that NCT05973786 is excluded while representative monotherapy, head-to-head,
factorial, and matched add-on trials remain relevant.

Affected report: `snapshots/bupropion_2026-09-09_20-59-55.md`.

## Important human evidence can remain outside the shortlist

The literature pipeline can omit controlled human evidence even when it is available locally.

For bupropion and alcohol dependence, NCT04167306 references PMID 40487775, a completed randomized trial with a bupropion
monotherapy arm. The report grades the evidence as weak and animal-only. PMID 40487775 is absent from the rerank candidate
set because the PubMed queries use the historical MeSH descriptor "Alcoholism", while the unindexed paper uses "alcohol use
disorder".

The trial-linked literature lane addresses this case by retrieving publications independently through their NCT identifiers
and reserving one shortlist position for the highest-ranked eligible publication. Its implementation is complete, but the
bupropion report-level verification remains pending.

For duloxetine and irritable bowel syndrome, PMIDs 34476222 and 38192887 are present in the retrieved PMID pool but absent
from the semantic shortlist. Both are placebo-controlled human trials. The report consequently states that all available
studies are uncontrolled and that placebo-controlled trials are still needed. These papers are not publications of the
relevant ClinicalTrials.gov records, so reserving a trial-linked position does not ensure their inclusion.

### Fix

Add a separate one-paper reserve for controlled human evidence already present in the ordinary rerank candidate set. After
semantic scoring, apply the existing exact-drug and disease-treatment gates to candidates outside the normal top fifteen.
Use the existing per-paper study-design judge to identify studies that are both human and controlled. Reserve one shortlist
position for the highest-ranked eligible paper, then fill the remaining positions in the existing ranking order. If no paper
qualifies, retain the normal fifteen-paper shortlist.

Do not infer controlled design from PubMed publication types alone. Both missing duloxetine studies are labelled only as
"Journal Article". Do not use text-pattern matching because design wording in an abstract can describe cited background
work rather than the reported study.

Before implementation, validate this selector across a broad sample of drug-disease pairs. Measure controlled-study recall,
false inclusion, shortlist churn, and changes to evidence strength and direction. Confirm that PMIDs 34476222 and 38192887
enter the duloxetine and irritable-bowel-syndrome shortlist without admitting papers about another drug, disease, species, or
non-efficacy outcome.

Affected reports: `snapshots/bupropion_2026-09-09_20-59-55.md` and
`snapshots/duloxetine_2026-09-09_02-11-49.md`.

## Closure conflates prevention with treatment

Sildenafil is closed for acute kidney injury using negative perioperative prevention studies.

The cited human trials evaluated prevention of renal injury around partial nephrectomy or cardiac surgery. They do not
establish failure for treatment of established acute kidney injury. The report nevertheless states that human evidence has
disproved the indication generally.

The closure input records the drug, disease, aggregate evidence direction, strength, and study design, but not treatment
versus prevention. Negative prevention evidence can therefore close treatment of established disease. The trial-linked
literature lane does not change this distinction.

### Fix

Add a study-intent field to the existing per-PMID judgment, with treatment, prevention, mixed, and other as its explicit
values. An absent judgment remains unavailable and cannot support closure. Build a closure-specific evidence body from PMIDs
that are human, controlled, and evaluate treatment of established disease. Apply the existing rules for evidence strength
and overall direction to that body independently of prevention and observational evidence.

Literature may close a candidate only when this closure-specific body is itself moderate or strong and its direction is
`contradicts`. Prevention-only failures must leave the broader candidate live, while the literature summary continues to
report failure in the prevention setting. Verify that the perioperative sildenafil studies do not close treatment of
established acute kidney injury, while a moderate or strong controlled negative treatment body remains eligible to close its
matching indication.

Affected report: `snapshots/sildenafil_2026-09-09_21-54-36.md`.

## Executive summaries can disagree with structured findings

The executive summary is not consistently derived from the structured candidate findings.

In the latest metformin report, Cardiovascular Disorder appears twice and the summary lists fifteen candidates while the
structured ranking contains fourteen. Breast Cancer and metabolic dysfunction-associated steatotic liver disease are marked
closed, but no closed-signals footer is produced. In the bupropion report, four candidate cards are marked closed while the
footer names only Cocaine Use Disorder. In the latest sildenafil report, Chronic Obstructive Pulmonary Disease and Acute
Kidney Injury are marked closed while the closed-signals footer is absent.

The ranked prose and footer are generated separately from the structured findings. Finalization rebuilds some ranked content
but preserves the model-written closed-signals footer instead of deriving it from the authoritative closure values.

Affected reports: `snapshots/metformin_2026-09-09_21-42-35.md`,
`snapshots/bupropion_2026-09-09_20-59-55.md`, and `snapshots/sildenafil_2026-09-09_21-54-36.md`.
