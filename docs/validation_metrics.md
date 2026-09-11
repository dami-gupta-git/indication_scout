# Plan: validation metrics

Five measurements, each with its own unit and its own answer key. Only the first counts errors; the rest measure stability, judge accuracy, card accuracy, and overall correctness. Build in order; 1 and 2 need no
labeling and run over reports already on disk.

All scripts live under `scripts/validation/`. Each writes one CSV per run to `results/validation_metrics/`
and prints a one-line summary. Inputs are the saved `test_reports/<drug>_<ts>.json` payloads and the
per-judge caches; nothing calls a live API or LLM except measurement 2's reruns.

## 1. Consistency lint (per report)

Unit: one report. Metric: violations per rule, per report, and total per report. Answer key: none;
each rule encodes a defect already confirmed in `for_me/errors/`.

Rules, each a pure function over the payload returning a list of (disease, detail):

- [ ] paper-vs-trial disagreement: a PMID in `relevant_pmids` appears in `references` of a trial
      in `contaminated_nct_ids` for the same disease (error 1).
- [ ] pharmacovigilance indication confounding: an adverse-event term in
      `drug_pharmacovigilance_summary` equals (case-insensitive) an approved indication or a
      disease in `candidate_diseases` flagged `contaminated`/`combination_only`. ASSUMPTION: the
      approved-indication list is not in the payload; parse the FAERS terms from the summary text
      and take approved indications from the `fda_label_indications` cache. Flag if either is
      unavailable rather than skipping silently.
- [ ] approved-scope leak: a ranked disease whose `approval_relationship` is `contaminated`. Not
      an error by itself (contaminated is kept by design); report as a separate count so the
      Eisenmenger class is visible without failing the rule.
- [ ] prevention-only closure: `closure == "closed"` where the clinical-trials `closure_reason`
      or literature summary names only prevention/perioperative settings. ASSUMPTION: no
      study-intent field exists yet (error 5); until it does this rule is a keyword screen over
      `closure_reason` and `summary` and is reported as "suspected", not counted in the total.
- [ ] empty-signal verdict: blurb `verdict` contains "bottlenecked" or "maturing" while
      `strength == "none"` and `relevant_nct_ids` is empty or only terminated.
- [ ] status-unknown over readout: blurb `verdict` contains "status unconfirmed" while
      `signals.has_completed_phase3` is true and literature direction is `contradicts` or `mixed`.
- [ ] footer accounting: every disease in `disease_findings` is in `top_diseases` or named in a
      footer line of `summary`; no disease appears in two footer lines; ranked block has no
      duplicate disease.
- [ ] harm quote polarity: `indication_harm_summary` quote starts with a negation of the named
      outcome ("without", "no ", "did not"). Keyword screen, reported as "suspected".
- [ ] Runner: `scripts/validation/report_lint.py <payload.json>...` → CSV with columns
      drug, run_ts, rule, disease, detail, severity(error|suspected). Summary line: errors per
      report, suspected per report.
- [ ] Unit tests: one fixture payload per rule, positive and negative case each.
- [ ] Baseline: run over every payload in `test_reports/` and record the per-rule totals in
      `results/validation_metrics/baseline_lint.md`.

## 2. Run-to-run stability (per drug)

Unit: one drug, N runs on a warm cache (N = 3 to start). Metric: per pair, the fraction of trials
whose relevant/contaminated verdict differs across runs; per card, whether stage, strength,
direction, design word, harm flag or closure differ; rank agreement between runs (Kendall tau over
diseases ranked in all runs). Answer key: none.

- [ ] Runner: `scripts/validation/stability.py <drug> --runs N` invokes the existing pipeline N
      times (same code, same cache), keeps every payload, and compares them pairwise.
- [ ] Output CSV: drug, disease, field, values_seen, n_distinct. Summary: trial verdict flip rate,
      card field flip rate, mean rank tau.
- [ ] Separate the uncached batch relevance call from the cached per-item judges in the report,
      so the flip rate is attributed to the call that produced it.
- [ ] Baseline on the four demo drugs (metformin, semaglutide, sildenafil, bupropion). Cost is
      one full warm run per repetition; LLM spend only, no new API fetches.
- [ ] Record which pairs flip in `for_me/errors/` if the rate is above zero on any pair; that
      list is the input to the evidence-pool redesign.

## 3. Per-judge accuracy (per item)

Unit: one judged item. Metric: precision and recall per judge against hand labels. Answer key:
a labeled sample per judge, stored as CSV under `tests/regression/labels/<judge>.csv` with columns
item_key, label, note. Labels are entered by hand once and never regenerated.

Judges and their cache namespaces:

| judge | cache | question | sample |
|---|---|---|---|
| trial treats disease | `trial_treats_disease` | does this trial treat the disease | 150 |
| trial relevance (batch, uncached) | none; take from payload `relevant_nct_ids` / `contaminated_nct_ids` | is the drug the studied agent for this disease | 150 |
| paper drug identity | `pmid_drug_identity` | does the abstract study the exact drug | 100 |
| paper treats disease | `pmid_treats_disease` | is the disease the treatment target | 100 |
| paper direction + design | synthesize per-PMID judgments (not separately cached; extract from payload buckets and `is_human`/`is_controlled` via a rerun) | verdict, human, controlled | 100 |
| indication harm | `indication_harm_verdict` | confirmed harm, and quote polarity | 100 |

- [ ] Sampler: `scripts/validation/sample_judgments.py <judge> --n N` draws stratified by drug
      and by cached verdict (half positive, half negative) so recall on the minority class is
      measurable. Writes an unlabeled CSV with the item's input text alongside, for labeling.
- [ ] Label the trial relevance and indication harm sets first; these two judges are behind most
      of the confirmed errors. Others follow as time allows.
- [ ] Scorer: `scripts/validation/score_judge.py <judge>` joins labels to current cached (or
      re-run) verdicts and prints precision, recall, and the disagreements with item keys.
      Re-scoring after a prompt change means clearing that judge's cache namespace and re-running
      only the labeled items.
- [ ] Baseline numbers into `results/validation_metrics/baseline_judges.md`.

## 4. Card-level accuracy (per drug)

Unit: one candidate card. Metric: field mismatch rate against a reviewed key. Answer key: the
existing `specs/<drug>.yaml` extended with, per pinned disease, the expected stage tier, strength,
direction, design word, harm flag, and closure. Limit to the four demo drugs.

- [ ] Extend the spec schema with those six fields; leave a field absent where the reviewer is not
      confident (an absent field is not scored).
- [ ] Scorer reuses the Layer 2 regression loader and reports mismatches per field instead of
      pass/fail. Existing pass/fail behaviour is untouched.
- [ ] Author keys for the four drugs from the 2026-09-10 runs, corrected by hand where the run
      is known wrong (Eisenmenger, heart-failure verdict, AKI closure). The key records what is
      right, not what the system printed.

## 5. Overall correctness (per drug set)

Two halves, because the report makes two kinds of claim: which candidates it surfaces, and what it
says about each one. Both need an external answer key; neither can be read off the report alone.

Candidate recall (did it find the right ones). Answer key: history, via the runbook rows
(drug, later-validated indication, cutoff). Reported at two levels:

- Seed recall: the indication is present in the merged candidate list at the cutoff, and its rank.
  `scripts/validation/gen_seed_candidate_recall.py` already measures this over the runbook
  (`results/holdout_validation/validation_results_*.md`); reuse it, do not rebuild.
- Ranked recall: the indication appears among the final ranked cards of a full holdout run at the
  cutoff. This is seed recall after the investigation cap; the gap between the two is the cap's
  cost, already recorded in findings. Needs one full holdout run per runbook row, so limit to the
  rows for the four demo drugs.

Precision is not scored on this side: a candidate that never panned out is not thereby wrong.

Card correctness (is what it says true). Answer key: the reviewed per-card keys from measurement 4.
A card is correct only when every scored field matches. Cards with a missing key field are scored on
the fields present.

- [ ] Combined figure: over the fixed drug set, correct cards / ranked cards, where a card is
      correct only if the disease is a legitimate candidate (not inside the approved label scope)
      AND every scored field matches the key. Report alongside holdout recall so dropping
      candidates cannot raise the figure without lowering recall.
- [ ] Drug set: the four demo drugs for card correctness; the existing runbook rows and cutoffs
      for recall. Do not add drugs until the keys for these four are reviewed.
- [ ] Scorer: `scripts/validation/overall_correctness.py` reads the latest payload per drug, the
      measurement 4 keys, and the latest recall results file, and prints: drugs, ranked cards,
      correct cards, correctness, recall rows, recall hits, recall.
- [ ] Record the first figure in `results/validation_metrics/baseline_overall.md` with the payload
      timestamps and key version it was computed from. Expect it to be low; it is strict by design.

Not measured here: omissions the key does not know about, and prose quality. Measurements 1 and 2
remain the leading indicators; this is the lagging outcome.

## Reporting

- [ ] `scripts/validation/validation_metrics.py` runs 1, 4 and 5 over the latest payload per drug and 3
      over the labeled sets, and prints one table: measurement, unit, count, rate. Measurement 2
      is run separately because it is expensive.
- [ ] Add the table to the session file after each run so the trend is visible across sessions.

## Not in scope

- LLM-judged claim checking of prose against cited abstracts. Needs its own labeled calibration
  set before its numbers mean anything; revisit after 3 exists.
- Fixing any error the measurements surface. This plan only measures.
