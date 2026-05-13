# IndicationScout

An agentic system for discovering drug repurposing opportunities.

## Overview

IndicationScout is an agentic drug repurposing system. A drug name goes in; coordinated AI agents query multiple biomedical data sources and produce a repurposing report.

A **Supervisor** agent orchestrates three specialist sub-agents:

- **Literature agent** — Queries PubMed via EUtils, then runs a RAG pipeline: fetch abstracts, embed with BioLORD-2023, semantic search, and LLM-based synthesis of evidence for each candidate disease.
- **Clinical Trials agent** — Queries ClinicalTrials.gov v2 (REST) per drug × indication pair: all-status search (whitespace verdict + per-status counts), completed-trial query (with Phase 3 count), terminated-trial query (with `why_stopped` text classified into safety / efficacy / business / enrollment categories at the tool layer), competitive landscape for the indication, and an FDA-label approval check used to short-circuit when the drug is already approved for the candidate. Indications are resolved to a MeSH descriptor via NCBI E-utilities and the resolved preferred term is fed to CT.gov's server-side `AREA[ConditionMeshTerm]"<term>"` filter — so e.g. "hypertension" trials aren't mixed in with glaucoma/portal/pulmonary hypertension.
- **Mechanism agent** — Queries Open Targets target-level data (GraphQL) to retrieve disease associations with evidence scores and Reactome pathway annotations.

The Supervisor first calls `find_candidates` (Open Targets competitor analysis) and `analyze_mechanism` in parallel, then delegates to the Literature and Clinical Trials agents per candidate disease.

### Top-5 candidate blurbs

After investigating candidates, the Supervisor produces a ranked Summary. For each of the **top 5 ranked candidates**, the Supervisor also writes a **2-sentence interpretive blurb** characterizing the *state of the hypothesis* — live, stalled, niche, untested, post-readout-and-stuck, etc. — grounded in the literature and clinical-trials sub-agent summaries seen during that run. Mechanism content is intentionally excluded from the blurb to keep it focused on clinical evidence. Blurbs are produced by the same `finalize_supervisor` tool call that emits the ranked summary (no extra LLM call) and rendered inline under each disease in the Summary section of both the Markdown report and the Streamlit Overview tab. Holdout runs (`--date-before`) skip blurbs.

Data sources:

- Open Targets (GraphQL) — drug targets, disease associations, competitor drugs
- ClinicalTrials.gov (REST v2) — trial search, whitespace detection, competitive landscape
- PubMed (NCBI EUtils) — literature retrieval and abstract indexing
- NCBI MeSH (E-utilities) — indication → MeSH D-number resolution for clinical-trials post-filtering
- ChEMBL — molecule metadata and ATC classifications
- openFDA — drug labels for FDA-approval extraction

## Installation

### Clone the Repository

```bash
git clone https://github.com/dgupta/IndicationScout.git
cd IndicationScout/indication_scout
```

### Install Dependencies

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

### Environment Setup

The app loads two env files in order: `.env` (secrets, DB credentials, API keys,
model names) and `.env.constants` (tunable numeric limits — top-k, timeouts,
batch sizes, etc.). `.env.constants` has **no defaults**; if a field is missing
the app fails to start. Both files must exist before running the CLI or API.

Copy `.env.example` to `.env` and fill in your API keys. A checked-in
`.env.constants` provides the tunable numeric limits — leave it as-is unless
you have a reason to change a limit.

```bash
cp .env.example .env
```

To swap the constants file at runtime (e.g. for tests or experiments):

```bash
CONSTANTS_FILE=.env.constants.test pytest
CONSTANTS_FILE=.env.constants.experiment scout find -d "metformin"
```

Required environment variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | Yes | PostgreSQL connection string (e.g. `postgresql+psycopg2://scout:scout@localhost:5438/scout`) |
| `DB_PASSWORD` | Yes | Database password |
| `TEST_DATABASE_URL` | No | Separate PostgreSQL URL for integration tests (e.g. `postgresql+psycopg2://scout:scout@localhost:5438/scout_test`) |
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key for Claude LLM calls |
| `NCBI_API_KEY` | No | NCBI API key for PubMed (increases rate limits) |
| `PUBMED_API_KEY` | No | PubMed API key (separate from NCBI key in config) |
| `OPENAI_API_KEY` | No | OpenAI API key |
| `OPENFDA_API_KEY` | No | OpenFDA API key |
| `LLM_MODEL` | No | Primary LLM model (default: `claude-sonnet-4-6`) |
| `SMALL_LLM_MODEL` | No | Lightweight LLM model (default: `claude-haiku-4-5-20251001`) |
| `EMBEDDING_MODEL` | No | Embedding model (default: `FremyCompany/BioLORD-2023`) |

The project requires a PostgreSQL database with the `pgvector` extension for storing PubMed abstract embeddings. See `docs/rag_details.md` for the Docker setup.

### Database Setup

Start the database container:

```bash
docker compose up -d
```

Then apply migrations to both the main and test databases:

```bash
# Main database
alembic upgrade head

# Test database (used by integration tests)
DATABASE_URL=postgresql+psycopg2://scout:<password>@localhost:5438/scout_test alembic upgrade head
```

Replace `<password>` with the value of `DB_PASSWORD` from your `.env` file.

> **Note:** After deleting Docker volumes (`docker compose down -v`), you must re-run both migration commands — the schema is not persisted.

## Usage

### CLI

```bash
scout find -d "metformin"                          # writes <drug>_<timestamp>.md to ./snapshots and <drug>_<timestamp>.json to ./test_reports
scout find -d "metformin" --out-dir reports/       # custom markdown output directory
scout find -d "metformin" --no-write               # print the markdown report to stdout (JSON payload is still saved to ./test_reports)
scout --help
```

#### Re-rendering a report without re-running the pipeline

Every `scout find` run also writes the `SupervisorOutput` payload to
`test_reports/<drug>_<timestamp>.json`. `scout render` reloads that payload and
re-runs the markdown formatter — no agents, no LLM calls, no network:

```bash
scout render -i test_reports/metformin_2026-05-11_14-30-22.json    # writes <stem>.md next to the JSON
scout render -i <payload.json> --out-dir reports/                  # custom output directory
scout render -i <payload.json> --no-write                          # print the markdown report to stdout
```

This is the fast path for iterating on `format_report` or producing fresh
markdown after a renderer change. Holdout runs (`--date-before`) do not save a
JSON payload.

#### Temporal holdout (`--date-before`)

For evaluation only, the pipeline can be run as a holdout study — restricting
every evidence query to what was knowable on or before a cutoff date:

```bash
scout find -d "semaglutide" --date-before 2022-01-01
# → snapshots/holdouts/semaglutide_holdout_2022-01-01_<timestamp>.md
```

What it does:

- **PubMed**: returns only abstracts published before the cutoff.
- **ClinicalTrials.gov**: returns only trials whose `start_date` is before the
  cutoff; for trials that started pre-cutoff but completed/terminated after,
  outcome fields (`overall_status`, `why_stopped`, `completion_date`) are
  scrubbed and the trial appears with status `UNKNOWN`. The competitive
  landscape tool short-circuits empty under a cutoff.
- **FDA approvals**: looked up against a hardcoded
  [`drug_approvals.json`](drug_approvals.json) table gated on the
  cutoff. Without `--date-before`, the pipeline falls back to today's
  openFDA labels as usual. Drugs not in the table get no approval reasoning
  during a holdout (a warning is logged).

> **Note:** To run a prospective holdout study, [`drug_approvals.json`](drug_approvals.json) must be updated with the drug's known approvals (and their approval dates) before running with `--date-before`. Drugs missing from this table will not get approval reasoning during the holdout.

Holdout reports are written to `snapshots/holdouts/{drug}_holdout_{cutoff}_{timestamp}.md`
to keep them visually distinct from current-state runs. No JSON payload is
saved for holdout runs, so `scout render` is not available for them.

Known limitations are documented in [`future.md`](future.md) — most notably,
the OpenTargets candidate list and mechanism scores remain current-state
because OT has no temporal API.

### API

```bash
uvicorn indication_scout.api.main:app --reload
```

## Development

### Running Tests

```bash
# All tests (regression + live tests are excluded by default)
pytest

# Integration tests only
pytest tests/integration/

# Unit tests only
pytest tests/unit/
```

### Regression Testing

Catch unintentional changes to the repurposing report as the pipeline evolves.
A regression test runs the full supervisor pipeline against committed
**cassettes** (recorded HTTP + LLM traffic) and compares the resulting
`SupervisorOutput` against a committed **golden** snapshot.

Comparison is structural + semantic — not exact text. The harness checks set
overlap of candidate diseases (Jaccard), presence of required nested fields,
and bounded drift on numeric counts (PMID totals, trial totals). Free-text
fields (summary, blurb prose) are checked for length bounds only, never exact
match. Thresholds live in `src/indication_scout/regression/constants.py`.

#### Prerequisites

Two things to set up once, in the venv you'll run pytest from:

1. **Install dev deps into the venv (not the system Python):**
   ```bash
   .venv/bin/python -m pip install -e ".[dev]"
   ```
   The `.venv/bin/python -m pip` form bypasses any `pip` Homebrew put on your
   `PATH` and guarantees the install lands in the active venv. If you see
   `ModuleNotFoundError: No module named 'vcr'` during a regression run, it
   means `pip` resolved to a different Python — re-run the line above.
   Verify with:
   ```bash
   .venv/bin/python -c "import vcr; print(vcr.__file__)"
   ```
   The path should be under `.venv/lib/.../site-packages/vcr/...`.

2. **Choose which constants file to record against.** By default,
   `tests/conftest.py` pins `CONSTANTS_FILE=.env.constants.test` (test
   limits — small top-k, short timeouts). Regression goldens are more useful
   when recorded against `.env.constants` (production limits) since that's
   what users actually see. Export the production value in your shell
   **before** invoking pytest:
   ```bash
   export CONSTANTS_FILE=.env.constants
   ```
   Setting it in the shell wins over the root `conftest.py`'s `setdefault`,
   and it pre-dates any conftest loading — important because `config.py`
   reads `CONSTANTS_FILE` at import time, which a `tests/regression/
   conftest.py` cannot reliably beat. Keep this consistent across all your
   pinned drugs so the cassettes and goldens are comparable.

#### Three-layer architecture

The regression suite is split into three layers, each catching a different
kind of failure and each runnable independently. Don't conflate them — they
trade off coverage, cost, and confidence differently.

| Layer | Marker | What it catches | Cost | Determinism |
|-------|--------|-----------------|------|-------------|
| **1: deterministic** | none (always runs) | Logic regressions in pure functions (evidence gate, threshold filters, parsers). No LLM, no network. | ms | byte-exact |
| **2: structural** | `regression_layer2` | Domain-meaningful invariants from a per-drug YAML spec (required NCTs/PMIDs appear, demoted indications stay demoted, forbidden phrases stay out). Runs against the latest `scout find` payload — no cassette replay. | seconds | structural |
| **3: triage** *(planned)* | `regression_layer3` | LLM-as-judge over the rendered report. Never fails CI; writes a triage report for human review. | minutes + API cost | stochastic |

Plus the existing **whole-pipeline replay** test (marker `regression`) — runs
the full pipeline against a vcrpy cassette and compares the result to a
committed `SupervisorOutput` golden. Heavier than Layer 2, narrower than
Layer 3. Keep it as a backstop.

#### Layout

```
src/indication_scout/regression/
├── constants.py        # tunable thresholds for the whole-pipeline diff
├── diff.py             # Diff dataclass + Jaccard
└── harness.py          # compare_reports(golden, current) -> list[Diff]
tests/regression/
├── failure_buckets.py  # Bucket enum + BucketedDiff (failure-mode taxonomy)
├── cassette.py         # vcrpy wiring for the whole-pipeline test
├── specs/              # per-drug YAML: the actual "test data"
│   └── bupropion.yaml
├── fixtures/           # (reserved) dated upstream snapshots per drug
├── cassettes/          # committed HTTP cassettes for the whole-pipeline test
├── golden/             # committed SupervisorOutput JSON per pinned drug
├── layer1_deterministic/
│   └── test_evidence_gate.py   # locks the top-N evidence-gate predicate
├── layer2_structural/
│   ├── spec.py         # Pydantic schema for the YAML
│   ├── loader.py       # YAML → DrugSpec
│   ├── assertions.py   # run_spec(spec, report) -> list[BucketedDiff]
│   ├── test_per_drug.py        # parametrized over specs/*.yaml
│   ├── test_assertions.py      # unit tests for the assertion functions
│   └── test_loader.py
├── layer3_triage/      # (reserved) LLM-as-judge triage reports
├── test_harness.py     # unit tests for compare_reports (whole-pipeline diff)
└── test_pipeline_regression.py # marker-gated full-pipeline replay test
```

Pinned drugs for the whole-pipeline replay test live in `PINNED_DRUGS` at the
top of `tests/regression/test_pipeline_regression.py`. Layer 2 picks up every
`*.yaml` in `specs/` automatically.

#### Layer 1: deterministic checks

Pure-function tests on the parts of the pipeline that have no LLM in them.
Run on every commit; no marker needed. The current proof-of-pattern is the
top-N evidence gate — see
[layer1_deterministic/test_evidence_gate.py](tests/regression/layer1_deterministic/test_evidence_gate.py).

When the production logic legitimately changes, update the predicate in the
test file in the same PR. The test is the contract; if it can't be edited
to match the new behaviour, the new behaviour is the regression.

```bash
pytest tests/regression/layer1_deterministic/
```

#### Layer 2: spec-driven structural assertions

A per-drug YAML in `tests/regression/specs/<drug>.yaml` encodes
domain-meaningful invariants. Each entry is bucket-tagged (see
[failure_buckets.py](tests/regression/failure_buckets.py)) so failures roll
up into the same taxonomy used in the bioRxiv failure-mode analysis.

Supported assertion types:

- `candidate_set_contains` — these indications must appear in `candidate_diseases`.
- `required_in_ranked` — this indication must appear in `top_diseases`.
- `forbidden_in_ranked` — this indication must NOT appear in `top_diseases`
  (demotion / gate worked).
- `required_ncts_surfaced` — these NCTs must appear under
  `disease_findings[indication].clinical_trials.{completed,terminated,search,any}`.
- `required_pmids_cited` — these PMIDs must appear in
  `disease_findings[indication].literature.pmids`.
- `forbidden_phrases` — this phrase must NOT appear (scoped to `summary`,
  `blurb`, or `anywhere`).

To run Layer 2:

```bash
# 1. Produce a fresh payload via the CLI
CONSTANTS_FILE=.env.constants scout find -d bupropion

# 2. Run the spec against the latest test_reports/<drug>_*.json
pytest -m regression_layer2 -k bupropion
```

A failure prints both the per-diff detail and a bucket rollup, so you can
tell at a glance whether you've broken (e.g.) demotion logic vs. literature
coverage vs. ranking. To add a new drug, drop a new YAML in `specs/` — no
test code changes needed.

The current [bupropion spec](tests/regression/specs/bupropion.yaml) is the
worked example and a good template.

#### Layer 3: LLM-as-judge triage (planned)

Layer 3 is reserved. The intent is to run a separate LLM judge over the
rendered report with a structured rubric, log the judgment to a triage
report, and **never fail CI on the judgment alone**. A judge that gates CI
quietly normalizes "approximately correct," which is the failure mode the
project rule "error by omission is acceptable, inaccuracy is not" exists to
prevent. Use Layer 3 as a flag for human review, not as a gate.

#### Whole-pipeline replay (marker `regression`)

The heaviest test: runs the full supervisor pipeline against a committed
vcrpy cassette and compares the resulting `SupervisorOutput` to a committed
golden via the `compare_reports` harness in
`src/indication_scout/regression/`. Catches structural drift across the
entire system in one shot. Slower than Layer 2 and harder to debug, but
catches things spec-driven assertions can't (e.g. "the mechanism block
disappeared entirely").

##### Modes

The cassette mode is selected by the `SCOUT_CASSETTE_MODE` env var:

| Mode | Behaviour | When to use |
|------|-----------|-------------|
| `replay` (default) | Play back the committed cassette. No network. Fails if a request isn't recorded. | Day-to-day regression checks. |
| `record` | Hit real APIs, overwrite cassette **and** golden together. | First-time setup, or when a pipeline change legitimately moves the report. |
| `live`   | Hit real APIs without recording. | Sanity-check against real services without touching cassettes. |

##### First-time setup: pinning a golden

```bash
# 0. Add the drug to PINNED_DRUGS in tests/regression/test_pipeline_regression.py
#    (e.g. PINNED_DRUGS = ["metformin"]). The `-k <drug>` filter only matches
#    parametrize ids, so the drug must be pinned before pytest will run it.

# 1. Set the constants file once per shell, before invoking pytest.
#    (See Prerequisites above for why this must happen in the shell, not a
#    conftest.)
export CONSTANTS_FILE=.env.constants

# 2. Record a fresh cassette and golden together. The test deliberately does
#    not skip when the golden is missing AND record mode is on — this is the
#    bootstrap path.
SCOUT_CASSETTE_MODE=record pytest -m regression -k metformin

# 3. Eyeball the golden — confirm candidate diseases, top diseases, and
#    per-disease findings look correct. If they don't, fix the pipeline and
#    re-record. Don't commit a bad golden.
$EDITOR tests/regression/golden/metformin.json

# 4. Replay it once with no env var to confirm the cassette is sound.
pytest -m regression -k metformin   # CONSTANTS_FILE still exported from step 1

# 5. Commit both files together
git add tests/regression/golden/metformin.json tests/regression/cassettes/metformin/
git commit
```

Shortcut: `scout find` already saves a `SupervisorOutput` payload to
`test_reports/<drug>_<timestamp>.json`. That's the same shape the harness
compares, so a good run can be promoted directly without re-recording:

```bash
cp test_reports/metformin_2026-05-11_14-30-22.json tests/regression/golden/metformin.json
# (You still need a cassette — capture one with SCOUT_CASSETTE_MODE=record
#  once the golden is in place.)
```

##### Day-to-day: running the test

```bash
pytest -m regression                    # replay all pinned drugs
pytest -m regression -k metformin       # replay just metformin
```

If the test fails, the assertion message lists the `error`-severity Diffs.
For a richer diff between two saved snapshots (no test runner needed):

```bash
scout diff-report tests/regression/golden/metformin.json test_reports/metformin_<timestamp>.json
```

`scout diff-report` exits non-zero on any `error`-severity Diff, so it's
scriptable from a shell.

##### Re-recording after an intentional pipeline change

When you legitimately change the pipeline (new agent, prompt change, new
data field) and the regression test fails for the right reason:

```bash
SCOUT_CASSETTE_MODE=record pytest -m regression -k <drug>
# Inspect the new golden — confirm the changes are the ones you expected
git diff tests/regression/golden/<drug>.json
# If the diff looks right, commit cassette + golden together
```

##### Why HTTP-layer recording

vcrpy records at the HTTP layer, so one cassette per drug covers **all**
external traffic: the `aiohttp`-based data source clients (PubMed, Open
Targets, ClinicalTrials.gov, ChEMBL, openFDA) and the `httpx`-based
Anthropic SDK + LangChain LLM calls. A future third LLM entry point cannot
silently bypass it.

#### Database is still live for every layer

Layer 2 and the whole-pipeline replay both need a live Postgres + pgvector
connection — Layer 2 because `scout find` writes embeddings to the DB
before the spec runs, the replay test because the cassette only stubs
**external** HTTP / LLM traffic. Same setup as the integration suite — see
[Database Setup](#database-setup) above. Layer 1 has no DB dependency.

### Code Formatting & Linting

```bash
black src/ tests/
ruff check src/ tests/              # lint
ruff check --fix src/ tests/        # lint + autofix
mypy src/                           # type checking
```

## Project Structure

```
src/indication_scout/
├── agents/          # AI agents
│   ├── base.py             # BaseAgent abstract class (currently unused by ReAct agents)
│   ├── _trial_formatting.py # Shared trial formatting helpers
│   ├── supervisor/         # Supervisor agent (orchestrates sub-agents) — supervisor_agent.py, supervisor_tools.py, supervisor_output.py
│   ├── literature/         # Literature agent (PubMed RAG) — literature_agent.py, literature_tools.py, literature_output.py
│   ├── clinical_trials/    # Clinical Trials agent (ClinicalTrials.gov + MeSH post-filter) — clinical_trials_agent.py, clinical_trials_tools.py, clinical_trials_output.py
│   └── mechanism/          # Mechanism agent (Open Targets targets) — mechanism_agent.py, mechanism_tools.py, mechanism_output.py, mechanism_candidates.py, mechanism_row_builder.py
├── api/             # FastAPI application (main.py, routes/, schemas/) -- /health endpoint only
├── cli/             # Click-based CLI (cli.py) — exposes the `scout` command
├── data_sources/    # Async API clients (OpenTargets, ClinicalTrials.gov, PubMed, ChEMBL, FDA, DrugBank stub)
├── db/              # SQLAlchemy session factory and declarative base
├── helpers/         # Utility functions (drug name normalization)
├── markers.py       # Code review exclusion markers (@no_review decorator)
├── ml_models/       # ML modeling code
│   ├── trial_risk/         # Trial-risk model (data.py, features.py, literature.py, score.py, train.py, inspect.py)
│   └── success_classifier/ # Trial-success classifier (features.py, labels.py)
├── models/          # Pydantic data contracts (model_open_targets, model_clinical_trials, model_pubmed_abstract, model_chembl, model_drug_profile, model_evidence_summary)
├── prompts/         # LLM prompt templates (supervisor, supervisor_holdout, synthesize, synthesize_holdout, expand_search_terms, extract_fda_approvals, extract_fda_approval_single, list_label_indications, extract_organ_term, merge_diseases, normalize_disease, normalize_disease_batch)
├── report/          # Report formatting (format_report.py) — turns SupervisorOutput into the final markdown report
├── runners/         # Pipeline runners (rag_runner.py) and exploration scripts (pubmed_runner.py); wandb/ logs
├── services/        # Business logic -- LLM calls (llm.py, including parse_llm_response), embeddings (embeddings.py), disease normalization + MeSH resolver (disease_helper.py: llm_normalize_disease, normalize_for_pubmed, resolve_mesh_id), PubMed query building (pubmed_query.py), FDA approval extraction (approval_check.py), RAG pipeline (retrieval.py -- fetch_and_cache, semantic_search, synthesize)
├── sqlalchemy/      # SQLAlchemy ORM models (pubmed_abstracts with pgvector embedding)
├── utils/           # Shared file-based cache utility (cache_key, cache_get, cache_set)
├── config.py        # Settings via pydantic-settings, loaded from .env
└── constants.py     # URLs, timeouts, lookup maps
```

## Known Limitations

- **Abstract-only indexing**: PubMed articles without an abstract (letters, editorials, conference summaries) are excluded from the vector store and will not appear in semantic search results. Only articles with a non-empty abstract are embedded and cached.
- **Incomplete Open Targets approval data**: Open Targets does not record all approved indications for every drug. Approved indications missing from Open Targets will not be filtered from repurposing candidates and may appear as false positives. For example, tofacitinib's ulcerative colitis and ankylosing spondylitis approvals are absent from Open Targets, causing them to appear as repurposing candidates.

### Citations
Open Targets: Ochoa, D. et al. (2023). The next-generation Open Targets Platform: reimagined, redesigned, rebuilt. 
Nucleic Acids Research, 51(D1), D1353–D1359. DOI: 10.1093/nar/gkac1037. 

## Acknowledgments

This codebase was created with the help of [Claude Code](https://claude.com/claude-code).

## License

MIT
