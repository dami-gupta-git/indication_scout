# Drug-Repurposing Approval Classifier — Probe Findings (2026-05-04)

This is the follow-up to the "If picking this up again" recommendation in
[`SESSION_FINDINGS.md`](../SESSION_FINDINGS.md): **Drug-repurposing approval
classifier — labels from `fda_drug_disease_approval` (60 / 484), features from
`competitors_merged`, `ct_completed/terminated`, ATC classes,
`expand_search_terms`. Probe feature distributions first using the same lift +
partial-coefficient pattern as probes 4 and 6 here. If literature volume
dominates again, kill it.**

**Net result: kill it. Not because literature dominates — because the labels
and the features come from two different cache populations with disjoint
selection processes. 53% of negatives and 66% of positives have no usable
features at all.**

---

## Setup

- **Labels.** `cache/fda_drug_disease_approval/` over 19 drugs.
  650 (drug, disease) verdicts: **76 positive, 574 negative** (11.7% pos rate).
  Slightly higher than the SESSION_FINDINGS estimate (60 / 484) because the
  cache grew between sessions.
- **Features per (drug, disease):**
  - `n_competitors`, `has_competitor` from `competitors_merged` (chembl_id → disease → drug list)
  - `n_ct_completed`, `n_ct_terminated`, `n_ct_total`, `term_to_total_ratio` from `ct_completed`/`ct_terminated` (keyed by drug + mesh_term)
  - `n_expand_terms` from `expand_search_terms` (chembl_id + disease)
  - `n_pubmed`, `log_n_pubmed`, `has_pubmed` — counted directly against the
    175,887-row `pubmed_abstracts` pgvector table: count of abstracts whose
    title or abstract mentions the drug name AND whose `mesh_terms` array
    contains the disease's MeSH descriptor.
  - `atc_level1`, `atc_level2` (per drug) from `atc_description`
- Disease → MeSH descriptor mapping from `mesh_resolver/`.
- Drug → ChEMBL via `chembl_id_to_names/`.

Validation: leave-one-drug-out CV (19 folds) with
`LogisticRegression(class_weight="balanced")`.

Code: [`probe.py`](probe.py). Outputs: [`out/`](out/).

---

## What the probes showed

### Coverage was the headline finding

| Feature | Rows with feature populated | Coverage |
|---|---|---|
| ATC code | 635 / 650 | 97.7% |
| `n_competitors > 0` | 270 / 650 | 41.5% |
| MeSH resolved (gates trial + pubmed features) | 247 / 650 | 38.0% |
| `n_pubmed > 0` | 229 / 650 | 35.2% |
| `n_ct_total > 0` | 95 / 650 | 14.6% |
| `n_expand_terms > 0` | **0 / 650** | **0%** |

`expand_search_terms` is keyed by `(chembl_id, disease_name)` from the
mechanism agent's expansion step. **None of those (chembl, disease) tuples
overlap the (drug, disease) label tuples** — the agent expands queries for
diseases it's investigating, never for the indications already in the FDA
verdict cache. This is the same disjoint-population pattern that broke the
success_classifier project.

### MeSH coverage gates everything

Of the 650 labeled pairs, 403 (62%) have no `mesh_resolver` entry for the
disease. That instantly zeros out `n_ct_completed`, `n_ct_terminated`, and
`n_pubmed` for two-thirds of the dataset. The unresolved set includes obvious
diseases that are trivially MeSH-able:

- baricitinib + rheumatoid arthritis (positive)
- rituximab + lymphoma, follicular lymphoma, DLBCL, CLL, NHL, RA (all positive)
- methotrexate + RA, psoriasis, lymphoma, osteosarcoma, ALL (all positive)
- imatinib + ALL, dermatofibrosarcoma, blast-phase CML (all positive)
- bevacizumab + cervical / colorectal / ovarian neoplasm (all positive)

The mesh_resolver cache is populated by the trial-search pipeline, which
is invoked for *candidate* diseases the agent decided to investigate — not
for the diseases an FDA-verdict cache happens to contain. **66% of the
positive class lives in the gap.**

### Per-class means look promising at first

| Feature | Neg mean | Pos mean | Diff |
|---|---|---|---|
| `n_pubmed` | 19.10 | 112.01 | +92.91 |
| `log_n_pubmed` | 1.10 | 1.66 | +0.55 |
| `n_ct_completed` | 1.37 | 1.78 | +0.41 |
| `n_ct_total` | 1.60 | 1.99 | +0.38 |
| `n_competitors` | 2.52 | 2.83 | +0.31 |
| `term_to_total_ratio` | 0.021 | 0.007 | −0.014 |
| `has_competitor` | 0.42 | 0.39 | −0.02 |
| `n_ct_terminated` | 0.24 | 0.21 | −0.03 |
| `has_pubmed` | 0.36 | 0.33 | −0.03 |

Continuous counts go the right way. But the binary indicators are flat or
slightly *negative* — a positive pair is no more likely than a negative pair to
have any pubmed hits at all. The continuous-count signal comes from a small
tail of well-documented positives (rituximab + DLBCL has 1000s of papers;
metformin + diabetes has 10000s) inflating the pos mean, not from genuine
class separation across the bulk of the data.

### Indicator-style lifts confirmed: no signal

| Indicator | n_present | P(pos\|present) | Lift over baseline (0.117) |
|---|---|---|---|
| `has_competitor` | 270 | 0.111 | **0.95×** |
| `has_pubmed` | 229 | 0.109 | **0.93×** |

Both lifts are *below 1*. The baseline pos rate is 11.7%; conditioning on
"this pair has at least one competitor / pubmed hit" *lowers* it slightly. This
is the opposite of probe 4 in the success_classifier work, where indicator
lifts were ~1.3–2.9× above baseline.

Quartile binning of `log_n_pubmed` only produced two non-empty buckets
(because 65% of rows have `n_pubmed = 0`):

| `log_n_pubmed` quartile | n | pos rate |
|---|---|---|
| 0 (zero or near-zero) | 487 | 0.105 |
| 1 (high) | 163 | 0.153 |

A 1.5× lift in the high-pubmed bucket — encouraging in isolation, but the
bucket is dominated by ~10 well-known approved indications.

### The LR model is below random on held-out drugs

Leave-one-drug-out CV, six features
(`n_competitors`, `n_ct_completed`, `n_ct_terminated`, `term_to_total_ratio`,
`n_expand_terms`, `log_n_pubmed`):

```
n=650  p=6  pos=76 (11.69%)  unique_drugs=19
ROC-AUC (LOGO): 0.405
PR-AUC  (LOGO): 0.159   (baseline 0.117)
```

**ROC-AUC = 0.405 is *worse than random*.** Standardized coefficients on the
full-data fit:

```
+ 0.321  log_n_pubmed
+ 0.104  n_ct_terminated
  0.000  n_expand_terms
- 0.035  n_competitors
- 0.037  n_ct_completed
- 0.557  term_to_total_ratio
```

`n_competitors` flips negative once you condition on the others — same
collinearity-flip pattern the success_classifier showed, but here without
even an honest in-fold AUC to defend it.

Stripping the literature feature does not rescue the model:

```
n=650  p=5  pos=76  groups=19
ROC-AUC (LOGO): 0.380
PR-AUC  (LOGO): 0.096   (baseline 0.117)
```

A literature-only model performs about the same:

```
ROC-AUC (LOGO): 0.394
PR-AUC  (LOGO): 0.192   (baseline 0.117)
```

None of these meaningfully separate held-out drugs.

### Why ROC < 0.5 — the drugs are the problem, not the features

Per-drug positive rate ranges from 0% (dasatinib: 0/15) to 36%
(methotrexate, rituximab: 10/28 each). LOGO averages predictions across
held-out drugs: when the model trains on 18 drugs with widely varying
prevalence and applies to a 19th, it systematically under- or over-predicts
the held-out drug's base rate. With features that don't carry a real
class-conditional signal, that drug-prevalence mismatch becomes the dominant
driver of the score, in the wrong direction.

A model that learned "high pubmed → approved" is locally true for famous
drug+indication pairs but actively misleading for, say, **metformin** (which
has high pubmed counts on dozens of conditions but is approved for one).

---

## Why this fails despite the label cache being real

The same root cause as the success_classifier work, but with a sharper
illustration:

1. **Labels come from `fda_drug_disease_approval`** — a verdict cache the
   agent populates by querying the FDA label parser for a fixed list of
   curated (drug, disease) pairs. These are diseases the agent *considered*
   for approval-checking, not all diseases.

2. **Features come from caches populated by *different* agent decisions.**
   `mesh_resolver` is populated for diseases the trial search needs.
   `competitors_merged` is populated for diseases the competitor agent
   surfaced. `expand_search_terms` is populated for diseases the literature
   agent decided to probe. None of these populations overlap the verdict
   cache cleanly.

3. **Result:** `expand_search_terms` overlap is exactly **zero**. MeSH
   coverage is 38%. Even with pgvector pubmed counts (which I added on top
   of the pre-existing caches to side-step this for one feature), the model
   has no real signal beyond "well-known drug+indication pair has lots of
   papers" — which is co-evolution leakage of the same kind that wrecked
   the success_classifier in probe 6.

4. **The 11.7% baseline pos rate is partly an artifact of the verdict cache
   being skewed toward exploratory diseases the agent is unsure about.**
   The bulk of the negatives are speculative (drug, disease) pairs the
   agent flagged for approval-checking, found "no", and stored. That is not
   the same negative class as "diseases this drug was tried on and failed."

---

## What was actually built this probe

- [`probe.py`](probe.py) — single self-contained script: loads caches,
  builds feature matrix, joins to pgvector pubmed counts, runs probes 1–5.
- [`out/features.csv`](out/features.csv) — 650 rows × 19 columns. Persisted
  so any follow-up work can skip the 35-minute pgvector scan.
- [`out/coverage.csv`](out/coverage.csv), `probe1_distributions.csv`,
  `probe2_lifts.csv`, `probe3_coefs.csv`, `probe4_coefs_nolit.csv`.

No model trained. No retraining recommended on this cache configuration.

---

## Recommendations

1. **Don't pursue the drug-repurposing classifier on the current cache.**
   The labels and features are from disjoint selection processes.

2. **If you want to revive this:** populate `mesh_resolver`,
   `competitors_merged`, and `expand_search_terms` for *every* disease in
   `fda_drug_disease_approval/`, not just the ones the agent decided to
   investigate. That means a one-shot batch job that re-runs those caches
   over the full label set. The cost: you'd still face the leakage problem
   from probe 6 of the success_classifier work — `n_pubmed` will dominate
   any LR fit, and that signal is co-evolved with approval status (more
   papers get written about indications a drug works for).

3. **The Phase 2 → Phase 3 advancement classifier** (the other unpursued
   idea in SESSION_FINDINGS.md) doesn't have this disjoint-cache problem,
   because its labels and features both come from `ct_completed/`. That is
   a stronger next probe than this one was.

4. **The pgvector pubmed table is genuinely useful infrastructure** —
   175,887 indexed abstracts with MeSH terms means any future probe can
   compute clean (drug, disease) literature counts in ~10 ms per pair
   without depending on the agent's query-string cache. Consider keeping
   that lookup as a reusable helper in `services/`.

---

## Lessons added on top of the prior session

9. **A label cache and a feature cache populated by different agent
   decisions cannot be naively joined for ML.** Even when both are about
   "(drug, disease) pairs," the selection process behind each cache shapes
   what counts as "present" in a way that leaks into any indicator
   feature. The fda_drug_disease_approval cache and the trial/competitor
   caches share fewer than half their pairs.

10. **Honest LOGO-CV will catch a feature-class mismatch the in-sample fit
    misses.** Per-class means looked promising (probe 1) and the LR
    converged with sensible-looking coefficients, but ROC-AUC fell below
    0.5 once the model had to predict held-out drugs. Always evaluate
    drug-grouped CV, not random splits, for any (drug, X) classifier on
    this kind of cache.

11. **Adding 175k indexed abstracts as a feature did not save the project.**
    The pubmed feature gave the highest univariate diff (pos mean 6× neg
    mean) but the highest LR coefficient (+0.32 standardized) couldn't
    overcome the disjoint-cache structural problem. More features don't fix
    a label-feature population mismatch.
