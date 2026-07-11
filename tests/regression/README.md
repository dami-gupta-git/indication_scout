# Regression tests

These tests pin **known-good IndicationScout reports** so that a change to the
supervisor, a sub-agent, the ranking logic, or an upstream data source can't
silently drop a signal we care about. The source of truth is the frozen
`gold_standard/` snapshots — hand-vetted `SupervisorOutput` reports for a small
set of drugs.

## What's here

```
gold_standard/          # frozen SupervisorOutput snapshots (JSON) + rendered reports (MD)
specs/                  # per-drug YAML: the invariants extracted from each snapshot
layer1_deterministic/   # pure-unit tests of the evidence gate (no fixtures)
layer2_structural/      # spec-driven assertions run against the gold_standard snapshots
pipeline_replay/        # full-pipeline replay test + its cassettes + compare_reports unit tests
common/                 # shared helpers: constants, failure-mode taxonomy, cassette wiring
```

## The layers

### Layer 1 — deterministic (`layer1_deterministic/`)

Pure unit tests of the evidence-gate logic (what survives when a candidate has
no trials, weak literature, etc.). No fixtures, no snapshots — just the gate
function and hand-built inputs. Runs in the default `pytest` suite.

### Layer 2 — structural specs (`layer2_structural/` + `specs/`)

The main regression layer. For each `specs/<drug>.yaml`, the test loads the
matching `gold_standard/<drug>_*.json` snapshot, deserializes it into a
`SupervisorOutput`, and runs every assertion in the spec against it. The test
harness and assertion functions are shared across all drugs — only the spec
data (which invariants to pin) differs per drug.

## What we test — the assertion types

Each spec is a list of assertions. Every assertion type checks one specific
property of the `SupervisorOutput`. These are the only things Layer 2 tests:

### `ranked_order`
The named indications appear in `top_diseases` in exactly the given relative
order. Only the listed indications are compared, in order; any others in
`top_diseases` are ignored. Every listed indication must be present — a missing
one fails. Use to pin a rank order that must not change. Bucket: `ranking`.

### `required_in_ranked`
An indication is present anywhere in `top_diseases` (order-agnostic). Use when
membership in the top set matters but the exact rank is not stable enough to
pin. Bucket: `ranking`.

### `forbidden_in_ranked`
An indication is **absent** from `top_diseases` — i.e. a demotion or gate fired
and kept it out of the ranked signals. Bucket: `demotion_logic`.

### `candidate_set_contains`
All named indications appear in `candidate_diseases` (the full considered set,
before ranking). Guards that a candidate isn't silently dropped upstream.
Bucket: `structural_integrity`.

### `required_ncts_surfaced`
The named NCT ids appear in a clinical-trials pool for that indication. The
`section` selects the pool:
- `relevant` (default) — `clinical_trials.relevant_nct_ids`, the curated set
  the report actually surfaces to the user. **This is the meaningful check.**
- `completed` / `terminated` / `search` — the corresponding raw trial list.
- `any` — the union of completed + terminated + search.

Bucket: `literature_coverage`.

### `required_pmids_cited`
The named PMIDs appear in a literature pool for that indication. The `mode`
selects the pool:
- `cited` (default) — `evidence_summary.supporting_pmids` +
  `contradicting_pmids`, the citations the report actually shows. **This is the
  meaningful check.**
- `pool` — the raw `literature.pmids` retrieval pool (~100+ PMIDs). A weak
  check: presence here says nothing about whether the report used the PMID.

Bucket: `literature_coverage`.

### `forbidden_phrases`
A phrase does **not** appear in the rendered report (case-insensitive). `scope`
selects where to look: `summary` (the summary text), `blurb` (the per-disease
blurb prose), or `anywhere` (the whole rendered markdown). Use for factual
guards — a statement that would be wrong if it appeared. Bucket:
`factual_accuracy`.

---

Failures roll up by **bucket** (`failure_buckets.Bucket`) so failure
distributions can be tracked over time.

The default `relevant` / `cited` pools pin the **curated** facts the report
surfaces to the user — not the raw retrieval pools, where a trial or PMID being
present says nothing about whether the report actually used it.

No LLM, no network — this layer reads only the committed snapshots.

### Full-pipeline replay (`test_pipeline_regression.py`)

Replays a recorded cassette through the **entire** supervisor pipeline and
diffs the resulting `SupervisorOutput` against the same
`gold_standard/<drug>_*.json` snapshot via `compare_reports`. This is the
coarse, whole-report check; Layer 2 is the targeted, fact-by-fact check.
Requires a live Postgres + pgvector connection (the cassette only stubs
external HTTP/LLM traffic).

## Running

```bash
# Layer 1 runs in the default suite:
pytest tests/regression/layer1_deterministic/

# Layer 2 (marked, excluded from the default run — opt in with -m):
pytest -m regression_layer2
pytest -m regression_layer2 -k bupropion        # one drug

# Full-pipeline replay (marked, needs a live DB):
pytest -m regression                             # replay (default cassette mode)
SCOUT_CASSETTE_MODE=live   pytest -m regression  # bypass cassette, hit real APIs
```

The `regression` and `regression_layer2` markers are excluded from the default
`pytest` run (see `addopts` in `pytest.ini`) — you must opt in with `-m`.

## Adding a new drug

**1. Capture the snapshot.** Get a known-good `SupervisorOutput` JSON (the same
payload `scout find` produces) and save it as
`gold_standard/<drug>_<datetime>.json` — e.g.
`aspirin_2026-07-10_14-30-00.json`. Drop the rendered `.md` report next to it
too (used by `forbidden_phrases` with `scope: anywhere`). Vet it: this file
**defines** what "correct" means for the drug, so read it and confirm the
ranking, trials, and citations are actually right before pinning them.

**2. Scaffold the spec.** Pre-fill a starter `specs/<drug>.yaml` from the
snapshot:

```bash
python scripts/scaffold_regression_spec.py <drug>           # writes specs/<drug>.yaml
python scripts/scaffold_regression_spec.py <drug> --stdout  # preview first
python scripts/scaffold_regression_spec.py <drug> --ncts 2 --force
```

It seeds `ranked_order`, `candidate_set_contains`, `required_ncts_surfaced`
(from `relevant_nct_ids`), and `required_pmids_cited` (from `evidence_summary`).
`forbidden_in_ranked` and `forbidden_phrases` are commented stubs — the
scaffolder can't infer them.

**3. Prune to the high-signal invariants.** The scaffold seeds everything; a
good spec pins only what matters:

- Trim `required_ncts_surfaced` / `required_pmids_cited` to 2–3 anchor
  trials/citations per disease (`relevant_nct_ids` can run to dozens).
- Trim `candidate_set_contains` to the diseases that must not silently drop.
- Fill in `forbidden_in_ranked` if the drug has a known demotion (e.g. a
  combination-product approval that must stay out of the ranking).
- Fill in `forbidden_phrases` for any factual guard (e.g. a monotherapy that
  must not be described as "approved for" a combination-only indication).

Rule of thumb: **error by omission is fine, inaccuracy is not.** Pin facts you
are confident are correct; don't pin marginal signals that a legitimate
pipeline change might reasonably move.

**4. Verify it passes against the snapshot:**

```bash
pytest tests/regression/layer2_structural/test_per_drug.py -m regression_layer2 -k <drug>
```

Layer 2 auto-discovers the new `specs/<drug>.yaml` — no test code to touch.

**5. (Optional) add the full-pipeline replay.** To also cover the drug in
`test_pipeline_regression.py`, add it to `PINNED_DRUGS` there and record a
cassette:

```bash
SCOUT_CASSETTE_MODE=record pytest -m regression -k <drug>
```

Record mode overwrites the existing `gold_standard/<drug>_*.json`, so the
datestamped snapshot file must already exist (from step 1).
