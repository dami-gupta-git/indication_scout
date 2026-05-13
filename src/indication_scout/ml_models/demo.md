# ML Models — Working Notes & Interview Prep

Captures what was discussed in a planning session about how to talk about
the existing ML work in an interview, what additional small project to
take on with limited time, and where fine-tuning / contrastive learning
fits in.

This is a discussion log, not a technical plan. For implementation
details see `trial_risk/docs/`, `success_classifier/docs/`, and
`SESSION_FINDINGS.md`.

---

## Starting context

The portfolio has two ML pieces:

- **trial_risk** — calibrated logistic regression predicting clinical
  trial termination. v1 ROC-AUC 0.609 on 303 trials / 3 drugs. v2
  (keyword fractions) and v3 (BioLORD fingerprints) underperformed v1.
- **success_classifier** — target-disease success classifier attempting
  to reproduce Nelson 2015. Killed at the probe stage after multiple
  data-quality probes showed the cache couldn't honestly answer the
  question.

User had limited time and wanted to know what small project to take on,
and how to frame the existing work for an interview.

---

## Drug-repurposing approval classifier (considered, not pursued)

`_cache/fda_drug_disease_approval/` has 544 (drug, disease) labels —
60 approved (11%) / 484 not — across 15 drugs. Tempting as a third ML
target.

### Coverage audit findings

Walked through the cache namespaces to see what features are actually
populated for the 544 pairs:

| Namespace | Coverage | Note |
|---|---|---|
| `_cache/drug/` (OT records) | 13/15 drugs | rituximab and sildenafil missing |
| `disease→mesh` resolution | 32% of pairs | gates every downstream feature |
| `ct_completed`/`ct_terminated` | 17% of pairs | sparse |
| FDA label + indications | 15/15 drugs | populated |

Per-drug positive rate ranges from 0.9% (metformin: 1/106) to 35.7%
(methotrexate: 10/28). Leave-one-drug-out CV would swing wildly.

### Why the project was rejected

Three structural problems surfaced before any modeling:

1. **OT `max_clinical_stage == APPROVAL`** is a near-definitional encoding
   of FDA approval. P(approved | stage=APPROVAL) = 64.6%; P(approved |
   stage=PHASE_2) = 1.6%. Strong leakage.
2. **FDA label exact-indication match** → P(approved) = 100%; fuzzy
   match → 59%; no match → 2.3%. The `fda_label_indications` extraction
   is essentially a lookup of the label.
3. **Trial counts in the cache are not what they seem.** The trial cache
   isn't a clinical history — it's the agent's repurposing-mode
   probes. Trial count features had no clean lift in the partial
   sample.

### What "more data" would actually mean

User clarified two things:
- The cache is in **repurposing mode** — agent surfaces non-approved
  candidate diseases. The 549 trials aren't full drug histories.
- Holdout mode = pull data **as of a date** (e.g., semaglutide @
  2017-12-01 = pre-Ozempic). This is real time-locking, the lever
  that probe 7 of `success_classifier` said was missing.

Recommendation: don't pursue the approval classifier. The leakage
features dominate, the non-leaky features are too sparse, and the
project is multi-session research, not a one-week ship.

---

## What "0.609 AUC" actually means

User asked whether 0.6 is "pretty good." Honest answer:

- ROC-AUC 0.5 = random; 1.0 = perfect; **0.609 = 11 points above
  random**. A small but real signal.
- Comparable medical-ML benchmarks: HINT (Lo et al. 2022) at ~0.70 with
  17,000 trials; DiMasi industry models at ~0.65 with proprietary data.
- 0.609 with 303 trials, 3 drugs, free public data is not embarrassing.
  It's roughly where a small model on cached data should land.

But: not good enough to make individual-trial decisions on. The strongest
v1 feature was `has_enrollment` (+2.5 weight) — a metadata-completeness
proxy, not a real content signal. v1's headline AUC was partly a leak.

### What matters more than the AUC

**Per-fold consistency.** A drug-out CV with stable AUC across folds is
deployable; a drug-out CV with AUC 0.45–0.85 across folds is a per-drug
lookup pretending to be a model.

The interview deliverable is "X drugs, AUC Y, per-fold variance Z" —
not just "AUC Y."

---

## Explaining v1 / v2 / v3 in plain terms

User asked for clearer explanations of the three trial_risk variants.

### v1 — cosine similarity over BioLORD embeddings

For each (drug, disease) pair, three semantic signals: failure, safety,
efficacy. Computed as mean cosine of top-5 hits from BioLORD-2023
embeddings of cached PubMed abstracts. AUC 0.609.

**The leak:** v1's `lit_signal_available` feature (binary "is there
literature for this pair?") had coefficient +0.67 — second-largest in
the model. That's a popularity proxy, not a content feature. Pairs with
literature are over-represented because (a) companies trial well-studied
combinations, (b) trial failures generate post-hoc papers. The "do we
have data" defensive null-handler became the load-bearing feature.

### v2 — keyword fractions

Replaced cosine with `count(abstracts matching keyword set) / count(all
abstracts)`. Three sets: failure (`terminat`, `discontinu`, `futility`,
…), safety (`adverse event`, `toxicit`, …), efficacy (`efficac`,
`remission`, …). AUC 0.570.

**Why the score dropped:** v2 stripped v1's leak. Only `lit_failure_signal`
had the expected positive sign; safety and efficacy were inverted because
their keywords double-count topic and intent (oncology mechanism papers
mention "toxicity" regardless of trial outcome).

**Honest read:** v2 is a weaker model but a more truthful one. The drop
from 0.61 → 0.57 is roughly the leak's contribution. v2's number is
closer to the model's real skill at the stated task.

### v3 — dense BioLORD fingerprint

768-dim averaged embedding per pair, concat with metadata + keyword
signals = ~792 features. Three classifier variants, all underperformed
baseline (0.41–0.47).

**Why it failed:** 792 features against 303 examples is the wrong end of
the bias-variance tradeoff. Curse of dimensionality. Inspection of one
trial (NCT01996696, metformin/prostate, terminated for futility)
revealed the literature *did* contain the warning — papers said
"controversial" and "conflicting" — but the signal lives at sentence
level, and document-mean pooling blurred it away.

---

## success_classifier — why it was killed

For each (target, disease) pair in OT cache, predict whether the pair
reached Phase II+ using only non-clinical evidence. Trying to reproduce
Nelson 2015 (genetic support ~2× clinical success odds).

### Probe sequence (chronological)

1. Feature distributions: genetic features were *inverted* — pairs that
   reached the clinic had **less** genetic evidence.
2. Drilled in: 532 of 921 positives (58%) had zero non-clinical
   evidence in cache.
3. Direct API probe: confirmed Open Targets genuinely has no
   mechanistic data for those pairs. Cache wasn't lossy.
4. Pivoted to `_cache/target/` (36k pairs, no record budget).
   Univariate lifts now matched Nelson's direction.
5. Trained quick LR. **AUC 0.748.** Looked great.
6. Coefficients told a different story — almost all weight on
   `literature` and `n_dt` (count of datatypes). Six of seven datatype
   features had inverted signs once those were partialled out.
7. Stripped leakage features (`literature`, `animal_model`,
   `rna_expression`, `affected_pathway`, `n_dt`). **AUC fell to 0.527.**
   Barely above random.

### The lesson

A working AUC doesn't mean a working model. With co-evolved features
and labels, the only honest answer was "this dataset can't reproduce
the published result, here's why." The deliverable was the probe
writeup; no model trained.

### What time-locking would unlock

User mentioned having a time-lock capability. If OT snapshots are
available at per-pair `T = first clinical record date`, the fame leak
narrows — `literature` and `genetic_literature` scores at trial-start
time are much smaller than current values. **This is the methodology
Nelson 2015 was careful about that the original probe didn't have.**

Caveat: time-locking helps the temporal axis. It doesn't fix the
**selection axis** — the 87 cached targets came from agent-surfacing on
15 already-marketed drugs, which biases the sample.

### Probe to run before retraining

Same 5 genetic-only features as probe 7, but computed at per-pair `T`.
Same LR, same leave-one-target-out CV. Compare AUC to 0.527.

- AUC moves to ~0.60+ → time-locking fixed it; expand to all drugs.
- AUC stays at ~0.527 → selection bias is the killer, not temporal
  leakage; project still dead.

---

## The interview story

### Opening line

> "I built a small ML pipeline on top of a biomedical drug-repurposing
> system. The interesting part wasn't the model — it was the iteration."

### Three sentences they should walk away with

1. "I noticed v1's strongest feature was a popularity proxy, not content."
2. "I killed a project because AUC 0.75 turned out to be feature leakage."
3. "The deliverable was honest probes, not models."

### What to emphasize

- **Methodology over headline number.** AUC 0.61 isn't impressive on
  its own. The probes are the skill.
- **Killing your own results.** Reporting v2 (0.57) as more honest than
  v1 (0.61) is senior behavior. Junior candidates only report the
  highest number.
- **Designed for honest evaluation.** Leave-one-drug-out CV. pgvector
  pub_date filtering. Time-locked holdout snapshots.
- **Knew when to stop.** success_classifier wasn't shipped because it
  shouldn't have been.

### What not to say

- Don't lead with the AUC.
- Don't claim "I built a clinical trial classifier." Frame it as "small
  honest signal, here's the per-fold breakdown."
- Don't apologize for the killed project — it's the strongest material.

---

## Fine-tuning — what to say

User has not actually fine-tuned anything in this project. The honest
answer in an interview:

> "I considered it for both projects. Decided against it for different
> reasons. For trial_risk, n=303 is too small to meaningfully shift a
> 768-dim encoder, and the signal lives at sentence level which
> document-level fine-tuning blurs. The right next step was an LLM call
> per abstract for stance classification, not fine-tuning. For
> success_classifier, the killer was leakage, not capacity — a more
> powerful model would just exploit the leak harder. Fine-tuning escapes
> ceilings; it doesn't escape leakage."

### Defensible follow-ups

- **"When would you fine-tune?"** When there's a working baseline and
  10k+ labels in a domain-specific task where pretrained encoders are
  close-but-not-quite right.
- **"What's the risk?"** Catastrophic forgetting at high lr, overfitting
  at small n, label noise amplification, loss of base-model generality.
- **"PEFT / LoRA?"** Right tool for small-n domain adaptation. Freeze
  most params, train low-rank adapter, get most benefit at fraction of
  parameter budget.

### Defaults worth memorizing

- Batch 16–32 for contrastive on a single GPU
- LR 1e-5 for MLM, 2e-5 for contrastive (LoRA can go to ~1e-4)
- 3–5 epochs typical
- AdamW, weight decay 0.01

These are the difference between sounding like you've done it and
sounding like you've read about it.

---

## Recommended week-long project

Three options ranked by realism in a one-week budget with a GPU.

### Option A — Contrastive projection head on BioLORD (recommended)

Frozen BioLORD + small (~5M param) MLP projection head. Triplet loss
with semi-hard negative mining. Triplets are constructed within MeSH
area to force outcome-related learning, not topic.

- ~5,000 triplets achievable from current cache
- CPU-trainable; GPU just speeds iteration
- Plugs into v1's pipeline as drop-in replacement for the 3 cosine
  signals

**Expected outcome:** AUC 0.61 → 0.65-ish, or no movement. Either is a
publishable interview story.

### Option B — LoRA fine-tune for stance classification

If Option A plateaus, pivot to fine-tuning. Generate weak stance labels
with Claude API on ~2000 abstracts, manually verify 100-200 for held-out
test. LoRA adapters (r=8, ~0.5% of base parameters), classification head
on `[supports, doubts, neutral]`. Fine-tuned encoder feeds trial_risk.

**Risk:** label noise amplification, overfitting at small n.

### Option C — Time-locked success_classifier reproduction

Use time-lock to re-run probe 7. Highest payoff (Nelson 2015 reproduction
is recognized) but highest risk (selection bias persists; might still
die at probe stage).

### Day-by-day plan for Option A

| Day | Task |
|---|---|
| 1 | Build triplet dataset, eyeball 50 triplets — confirm they encode outcome differences not just topic |
| 2 | Train projection head, save checkpoints, eval on held-out drug fold |
| 3 | Integrate into trial_risk pipeline, first real AUC number |
| 4 | If it worked: ablations + writeup. If plateaued: pivot to Option B |
| 5 | Buffer / writeup |

**Day 1 is the unblocking step.** Until 50 triplets have been eyeballed,
all model-config discussion is premature. If triplets look like topic
separation rather than outcome separation, hard-negative mining strategy
needs rethinking before any training.

### Three interview outcomes for the week

- **A worked**: "Trained a contrastive projection head with semi-hard
  negative mining stratified within MeSH area. AUC moved from 0.61 to
  0.66 with halved per-fold variance."
- **A plateaued, B worked**: "Started frozen-base contrastive, didn't
  move. Pivoted to LoRA fine-tune on stance classification. Used the
  fine-tuned encoder as a feature extractor."
- **Both plateaued**: "Tried two approaches, neither moved AUC. The
  diagnostic told me the data, not the representation, was the
  bottleneck — the signal exists but lives in too few sentences per
  pair to extract reliably at this scale." This is the most senior
  answer, even though it sounds disappointing.

---

## Open items

- Verify time-lock capability includes `fda_drug_disease_approval`
  verdicts (not just trial / literature data) before using holdout-mode
  pulls as a leakage probe.
- Backfill the cache for rituximab and sildenafil OT drug records.
- If pursuing Option A: build the triplet construction script.
- If pursuing trial_risk scaling first: warm `trial_risk_lit_signals`
  for the 246 trials beyond the v1 set (Stage 1 of the existing
  trainer).
