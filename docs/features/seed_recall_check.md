# Seed-phase candidate recall check

A CI gate that asks one question per row: under a cutoff several months before a real approval,
does the approved disease reach the drug's candidate list?

It runs only the seed phase — mechanism analysis, competitor surfacing, and the merge that
combines them. It does not run literature, trials, or synthesis, so it costs a fraction of a full
pipeline run.

## Why a cutoff

Every row's target is a disease the drug is now approved for, and in live mode the pipeline drops
diseases the drug is already approved for. Measured live, every row would read as absent. Passing
a cutoff suppresses the approval signal: the mechanism score is recomputed without clinical
precedence, and the approval filter consults a date-aware table rather than today's FDA labels.

The dates in the runbook are already set six months before the approval they correspond to.
Changing that window means editing the date column; the check needs no code change to follow it.

## Scoring

A row passes when one of its accepted names appears in the merged candidate list, compared by
exact string equality after lowercasing and trimming. The accepted names are the row's indication
plus the semicolon-separated alternatives in its `accepted` column. Nothing is inferred — no LLM
matching, no fuzzy or substring matching.

The `accepted` column exists because Open Targets names diseases its own way. A target that
surfaces under a name the row does not list reads as a miss until the name is added by hand. Judge
a candidate name against what exists upstream rather than against the label wording: an entry that
looks like a broad parent can be the most specific name the ontology carries.

## Two failure modes

A row expected to pass that does not is a failure. A row listed under `known_missing` that starts
passing is also a failure. The second rule means a fix to the underlying pipeline forces an update
to the expectations file instead of passing unnoticed.

## Files

- `scripts/check_seed_recall.py` — the check. Reads the spec, selects the matching runbook rows,
  runs one seed phase per distinct drug and cutoff, and exits non-zero on any mismatch.
- `tests/regression/labels/seed_recall.yaml` — which drugs to cover and which rows are expected to
  be absent, each with the reason it is absent.
- `scripts/validation/runbook.txt` — the ground truth: one row per approval, holding drug,
  indication, cutoff date, and accepted names.
- `scripts/validation/gen_seed_candidate_recall.py` — the exploratory harness over the whole
  runbook. The check imports its seed-phase runner and its matcher, so both score identically.
- `results/ci/seed_recall.json` — written on every run, not committed. Carries a summary with the
  recall figure and the dial values it was measured at, plus per-row position, matched name,
  source and verdict. The precision check writes the same shape into the same directory.

## Running

```
make seed-recall                          # the REGRESSION_DRUGS subset of seed_recall.yaml
python scripts/check_seed_recall.py --out /tmp/recall.json          # every drug in the spec
python scripts/check_seed_recall.py --drugs semaglutide bupropion
```

`REGRESSION_DRUGS` defaults to semaglutide and bupropion in the Makefile and CI; a manual CI run or
`make seed-recall REGRESSION_DRUGS="..."` can name more.

Needs an Anthropic key and a reachable database. In CI it is a step in the regression job, after
the structural specs, reusing that job's key, weekly data cache and Postgres service.

## Adding a drug

Add the drug to the spec's `drugs` list. Every runbook row for it is then checked. If a row is a
known miss, add it under `known_missing` with the reason, and record the underlying defect in
`for_me/errors/errors.md` when there is one.

## Related

The same seed phase over the whole runbook is what `gen_seed_candidate_recall.py` measures; use it
when tuning, and this check to hold the result. Known defects that cause rows here to miss are
recorded in `for_me/errors/errors.md`.
