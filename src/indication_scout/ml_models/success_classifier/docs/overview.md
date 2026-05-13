# Success Classifier — Overview

## Status (2026-05-03): paused after probes

This was an attempt to predict whether a (target, disease) pair has reached
a clinical milestone (Phase II+) using only **non-clinical** evidence from
Open Targets, framed as a reproduction of Nelson et al. 2015.

**Result of the investigation: not pursued.** Probes against both
`_cache/target_evidences/` and `_cache/target/` showed the cache cannot
honestly answer this question. Details below; full session log in
[../../SESSION_FINDINGS.md](../../SESSION_FINDINGS.md).

The original design (kept further down for context) assumed Nelson 2015
would reproduce on this cache. The probes ruled that out.

## What was actually built

- `success_classifier/labels.py` — extracts (target, disease) labels from
  `target_evidences`, masks clinical records.
- `success_classifier/features.py` — per-pair feature vector
  (count/max/mean per datatype + cross-cutting indicators).

No trainer was written. No model was fit. No metrics generated.

## Why it was paused

Three probes, three dead ends:

1. **`target_evidences/` (1,305 pairs):** All genetic-feature lifts ran
   *opposite* to Nelson 2015. Drilling in: 532/921 (57.8%) of positives
   have **zero** non-clinical evidence records. A direct probe against the
   OT `associatedDiseases` API confirmed the data really isn't there
   upstream — the cache wasn't lossy. Re-pulling won't fix it.

2. **`target/` with full features (25,139 pairs after filtering):**
   Univariate lifts looked correct (genetic features positive,
   matching Nelson). Trained LR on 9 features, leave-one-target-out CV
   over 87 folds:

   ```
   ROC-AUC: 0.748   PR-AUC: 0.074  (baseline 0.020, 3.7× lift)
   ```

   Looked like a working model. Coefficients told a different story:
   six of seven datatype scores went *negative* once `literature` and
   `n_dt` were partialled out. The model was learning **fame**:
   pairs-with-drugs generate post-hoc papers → `literature` score
   rises → "predicts" clinical advancement. Pure co-evolution leakage.

3. **`target/` with genetic-only features (honest probe):** Dropped
   `literature`, `n_dt`, `animal_model`, `rna_expression`,
   `affected_pathway` — kept only datatypes that shouldn't grow
   post-hoc when a drug exists.

   ```
   ROC-AUC: 0.527   (baseline 0.500)
   ```

   Barely above random. The 0.748 was almost entirely fame leakage.

## Why this dataset can't reproduce Nelson 2015

- OT genetic-association scores are **current**, not time-locked to
  trial start. They've grown alongside clinical activity.
- `literature` and `genetic_literature` scores are fed by post-trial
  publications.
- The 87 cached targets were surfaced by IndicationScout's mechanism
  agent for 15 already-marketed drugs. Even with OT's full disease
  graph, the target selection is biased toward "already-clinical"
  pairs.

Don't revisit `target_evidences/` or `target/` for this label without
publication-date metadata that lets features be time-locked to before
the clinical record exists. OT doesn't expose that.

## Two angles still worth probing (different label, same probe pattern)

- **Drug-repurposing approval classifier** — labels from
  `fda_drug_disease_approval` (60 approved / 484 not), features from
  `competitors_merged`, `ct_completed/terminated`, ATC classes,
  `expand_search_terms`. Closer to IndicationScout's actual question.
  Probe feature distributions first using the lift +
  partial-coefficient pattern from the session log. If literature
  volume dominates again, kill it.
- **Phase 2 → Phase 3 advancement** — derive labels from the cached
  trial set itself. Sidesteps the OT feature-leakage problem because
  features and labels come from different sources. The `trial_risk`
  pipeline already handles per-trial date-bounded literature.

---

## Original design (for context — assumed Nelson 2015 would reproduce)

The text below is the pre-probe design doc. It is preserved for
reference; do not implement against it without re-reading the probe
results above.

### What this does

For each (target, disease) pair, predicts whether the pair has reached a
clinical milestone (Phase II+ trial activity), using **only non-clinical
evidence** as features. The label and the features come from the same
cached Open Targets dataset; the label is masked out at feature time so
the model can't see its own answer.

This is the "genetics-to-clinic" question that the IndicationScout
mechanism agent asks every time it scores a target — but answered with a
calibrated classifier instead of hand-tuned heuristics.

### Why this is a sensible ML target

- **Data scale.** 1,305 (target, disease) pairs already cached in
  `_cache/target_evidences/`. ~15× more examples than the trial-risk
  model.
- **Deterministic labels.** Pulled from the same cache (`clinical`
  evidence records carry the score). No LLM labeling, no manual review,
  no schema design.
- **Class balance is workable.** ~70% positive / 30% negative.
- **Reproduces a known published result.** Nelson et al. 2015 found
  genetic support roughly doubles clinical success odds. If the model
  recovers that direction and rough magnitude, it's behaving sanely. If
  not, it's a data problem — not a modeling problem.
- **Plugs into the existing system.** Output can feed the mechanism
  agent's target-scoring step directly.

### Data sources

- `_cache/target_evidences/` — 87 files, one per target. Each entry
  groups records by disease ID and datatype. 8 datatypes total:
  `clinical`, `genetic_association`, `somatic_mutation`, `animal_model`,
  `rna_expression`, `literature`, `genetic_literature`,
  `affected_pathway`.

Nothing else is needed for v1.

### Label

For each (target, disease) pair:

```
clinical_records = [r for r in pair.records if r.datatype_id == "clinical"]
max_clinical_score = max(r.score for r in clinical_records, default=0)
label = 1 if max_clinical_score >= 0.7 else 0
```

`>= 0.7` corresponds to Phase II+ activity in Open Targets' clinical
scoring scale. We may also train a stricter variant at `== 1.0`
(approved indications only).

`clinical` records are removed before feature extraction so the model
can't trivially read its own answer.

### Features

All derived from the non-clinical evidence on the same pair. Per
datatype (`genetic_association`, `somatic_mutation`, `animal_model`,
`rna_expression`, `affected_pathway`, `literature`,
`genetic_literature`):

- `n_<datatype>` — record count
- `max_score_<datatype>` — max score across records
- `mean_score_<datatype>` — mean score across records

Plus a few cross-datatype features:

- `n_distinct_datatypes` — how many evidence types are present
- `has_any_genetic` — indicator: any of `genetic_association`,
  `genetic_literature`, `somatic_mutation` present
- `log_total_evidence` — log of total record count
- (Optional) `direction_consistency` — fraction of genetic records
  whose `direction_on_trait == "protect"` (a proxy for "this target's
  loss of function helps")

~20 features total.

### Model

- `LogisticRegression(class_weight="balanced", max_iter=1000)` wrapped
  in `StandardScaler` + `CalibratedClassifierCV(method="isotonic")`.
- `GradientBoostingClassifier` for comparison — non-linear baseline,
  doesn't need scaling.
- Cross-validation: **leave-one-target-out**. Each target's diseases
  go into one fold. Prevents the model from learning "target X always
  ends up in clinic" via leakage across diseases.

### Evaluation

- **ROC-AUC** (against baseline 0.5)
- **PR-AUC** (against class balance ~0.70)
- **Brier score** (calibration)
- **Calibration curve** plot
- **Ablation table** — drop each datatype's feature group, retrain,
  record ΔAUC. This is the actual deliverable: it answers "*which
  kinds of evidence predict clinical progress on this data slice?*"

### Date-leakage handling

Open Targets evidence and clinical status co-evolve — once a drug
reaches Phase II, more papers get written about its target, which can
leak back into `literature` records. Mitigation:

- If `literature` records carry publication dates, filter to records
  predating the target's first `clinical` record. v1 will check whether
  dates are present and structured; if not, this masking is dropped and
  flagged as a known limitation.

### Out of scope (v1)

- No drug-level features (this is a target-disease question, not a
  drug-target-disease one).
- No PubMed enrichment beyond what's already in `target_evidences`.
- No wiring into the mechanism agent.
- No new external data sources.

### Honest limitations

- **Selection bias.** The 87 targets are the ones IndicationScout's
  mechanism agent surfaced for the 15 cached drugs — not a random
  sample of druggable targets. The model learns on a biased slice. We
  state this explicitly; we don't claim drug-discovery generality.
- **Asymmetric negatives.** Some pairs lack clinical evidence because
  the target was tested and failed; others because nobody looked. Open
  Targets doesn't distinguish these. The model conflates them.
- **Co-evolution leakage.** Even with date-masking on literature, other
  datatypes (animal_model, rna_expression) can also reflect post-hoc
  research interest. We accept this and note it.

### Pipeline

```
target_evidences/*.json
  → for each (target_id, disease_id) pair:
      label = max(clinical.score) >= 0.7        # before masking
      mask out clinical records
      features = aggregate non-clinical records
  → leave-one-target-out CV
      train LR(balanced) + GBM
      record AUC, PR-AUC, Brier per fold
  → ablation: for each datatype group:
      retrain without those columns
      record ΔAUC
  → save model + metrics to models/success_classifier_v1.{pkl,metrics.json}
```

### Suggested file layout

```
src/indication_scout/success_classifier/
  __init__.py
  features.py     # extract per-pair feature vector
  labels.py       # extract label, with date masking
  train.py        # CV + ablation runner + CLI
  inspect.py      # per-pair audit tool
  docs/
    overview.md   # this file
    progress.md   # iteration log, populated as runs happen
```

### Expected outcome (pre-probe)

ROC-AUC 0.70–0.78. PR-AUC 0.80–0.85 (high baseline ~0.70). The
ablation should show genetic evidence (`genetic_association`,
`genetic_literature`) carrying the largest weight, animal/RNA evidence
moderate, pathway evidence smallest — matching published literature.

If results diverge from that pattern, treat it as a finding to
investigate, not a failure to fix.

**Probe outcome (2026-05-03): results diverged. See top of file.**
