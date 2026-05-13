# Trial-Termination Risk Classifier — Overview

## What this does

For each clinical trial, computes 3 numbers describing what the medical
literature was saying about the (drug, disease) pair *before* the trial ended:

- How much does the literature talk about **failure**?
- How much about **safety problems**?
- How much about **efficacy**?

Those 3 numbers, plus basic trial metadata (phase, sponsor, enrollment, etc.),
feed a calibrated logistic-regression model that predicts whether the trial
will be terminated.

## How we get the 3 numbers

1. **Find papers** about the drug + disease via a PubMed search (broad, no
   date filter — same query reused across trials).
2. **Filter to papers published before a cutoff date** (`completion_date - 6
   months`). The filter is enforced at the pgvector layer on `pub_date`, so
   the model can't "cheat" by seeing post-mortem papers.
3. **Run 3 semantic searches** against the BioLORD-2023 embeddings of those
   abstracts: failure / safety / efficacy intent. Take the mean cosine
   similarity of the top-5 hits per query.
4. **Average across the trial's MeSH conditions** to get one final triple
   per trial.

## Why this might work

If a trial gets terminated, there's often prior literature hinting at trouble
— early safety signals, mechanism doubts, similar failed trials. The model
learns whether that prior signal correlates with termination.

## Pipeline (per trial, run in parallel up to N at a time)

```
trial → cutoff_date = completion_date - lookback_months
      → for each mesh_condition:
            drug_aliases = first 5 names from ChEMBL
            pmids = ⋃ PubMed("{alias} AND {mesh}")            # broad, cached
            failure  = mean cosine top-5 in pgvector
                       restricted to pmids AND pub_date < cutoff
            safety   = same, with safety query
            efficacy = same, with efficacy query
      → average (failure, safety, efficacy) across mesh_conditions
      → cache result by (nct_id, lookback_months, top_k)
```

## Data sources

- `_cache/ct_completed/`, `_cache/ct_terminated/` — labels and trial metadata
- `_cache/pubmed_search/` — broad PMID lists per (drug-alias, mesh) pair
- pgvector `pubmed_abstracts` — abstract text + BioLORD embeddings + pub_date
- `_cache/trial_risk_lit_signals/` — persisted per-trial signals (skip on rerun)

## Model

- `LogisticRegression(penalty="l2", class_weight="balanced")` wrapped in
  `CalibratedClassifierCV(method="isotonic", cv=5)`.
- Cross-validation: leave-one-drug-out (group = drug name from cache).
- Won't ship the artifact unless PR-AUC beats the class-balance baseline
  (~21% terminated).

## Features

| Feature                                  | Source                        |
|------------------------------------------|-------------------------------|
| `phase__*` (one-hot)                     | `Trial.phase`                 |
| `sponsor__*` (one-hot: industry/academic/nih/other/unknown) | regex on `Trial.sponsor` |
| `log_enrollment`, `has_enrollment`       | `Trial.enrollment`            |
| `n_mesh_ancestors`, `n_interventions`, `n_primary_outcomes` | counts |
| `start_year`, `has_start_date`           | `Trial.start_date`            |
| `lit_failure_signal`                     | mean cosine, failure query    |
| `lit_safety_signal`                      | mean cosine, safety query     |
| `lit_efficacy_signal`                    | mean cosine, efficacy query   |
| `lit_signal_available`                   | indicator, false if no PMIDs  |

## Running

```bash
# Stage 1 — warm the per-trial signal cache (slow first time)
python -m indication_scout.ml_models.trial_risk.train \
    --drugs metformin,rituximab,thalidomide --dry-run

# Stage 2 — actually train (fast: signals are cached)
python -m indication_scout.ml_models.trial_risk.train \
    --drugs metformin,rituximab,thalidomide

# Score known trials by NCT ID
python -m indication_scout.ml_models.trial_risk.score NCT00064337
```

## Tunables (in `literature.py`)

- `DEFAULT_LOOKBACK_MONTHS = 6`
- `DEFAULT_TOP_K = 5` (top-N cosine hits averaged per signal)
- `MAX_DRUG_ALIASES = 5` (cap on PubMed query expansion per drug)
- `PUBMED_SLEEP_SECONDS = 1.0` (NCBI rate-limit cushion)
- `CONCURRENCY = 4` (parallel trials in `train.py`)

## Out of scope (v1)

- No wiring into the supervisor or the IndicationScout report.
- No KMeans / `why_stopped` clustering.
- No retraining pipeline or scheduling.
- No new external data sources beyond what the existing cache uses.
