# IndicationScout

An agentic system for discovering drug repurposing opportunities.

**Live demo:** <https://indicationscout-production.up.railway.app/>

A drug name goes in; coordinated AI agents query live biomedical databases and produce a
structured repurposing report — candidate indications, evidence strength, trial activity, safety
signals, and a per-disease read on the state of the hypothesis.

> For what every report field, verdict, and label means — and the exact rule behind each — see
> [GLOSSARY.md](GLOSSARY.md). For the system design, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Overview

IndicationScout pulls **live evidence at query time** — no static knowledge base, no precomputed
answers. For each candidate indication it gathers current trial activity, recent literature,
mechanistic associations, and regulatory status, then synthesizes them into one report that
characterizes the *state of the hypothesis* per disease (live, stalled, niche, untested,
post-readout-and-stuck).

The system is built for **accuracy over coverage**: it would rather miss a legitimate candidate
than surface one that isn't grounded in the upstream data. Candidates are seeded only from Open
Targets target–disease associations, and reject paths are never loosened to rescue a missing
result.

A **Supervisor** agent orchestrates three specialist sub-agents:

- **Literature** — PubMed retrieval via NCBI E-utilities, then a RAG pipeline (BioLORD-2023
  embeddings, semantic search, LLM synthesis) plus a dedicated adverse-event safety pass.
- **Clinical Trials** — ClinicalTrials.gov v2 queries per drug × indication, MeSH-resolved,
  with termination-reason classification and an FDA-label approval check.
- **Mechanism** — Open Targets target-level disease associations, evidence scores, and Reactome
  pathway annotations.

---

## Key characteristics

- **Multi-agent, live-data-first.** Real-time queries to Open Targets, ClinicalTrials.gov, PubMed,
  Europe PMC, ChEMBL, and openFDA — parsed into Pydantic contracts at every boundary; agents never
  see raw API responses.

- **Explicit safety reasoning.** A drug-wide safety summary (deterministic severity — withdrawn /
  black-box / serious) from Open Targets warnings and FAERS signals, plus a per-indication harm
  flag from disease-scoped adverse-event literature ranked by Europe PMC citation count.

- **Contamination handling.** A four-way FDA approval label per candidate — `approved` (dropped),
  `combination_only` (demoted), `contaminated` (kept, trial tables suppressed), `none` — decided
  once from the FDA label and threaded down to the trial and literature relevance gates. Filters
  approved indications, wrong-drug trials, and comorbid conditions out of a candidate's evidence.

- **Temporal holdouts.** Run the system "as of" a past cutoff date to test whether it would have
  surfaced an opportunity before it was validated. PubMed and trial queries are restricted to
  pre-cutoff records; undateable signals are suppressed.

- **Anti-hallucination by construction.** Exact identifiers (NCT IDs, PMIDs, OT scores) are pulled
  straight off typed tool artifacts, never re-typed by the LLM (`content_and_artifact` pattern).
  Floor/cap patterns let the model propose while deterministic code clamps to the underlying data.
  No fabricated or default values for scientific/clinical fields.

- **Prompt caching.** The ReAct agent loops use Anthropic ephemeral prompt caching (system-prompt
  and growing-history breakpoints), cutting warm-run cost ~10–20% at a ~44–49% cache hit rate. See
  [docs/anthropic_caching.md](docs/anthropic_caching.md).

- **Full stack.** CLI, FastAPI backend, and React web UI; shared disk cache; snapshot regression
  harness; async job API.

Data sources: Open Targets (GraphQL), ClinicalTrials.gov (REST v2), PubMed (NCBI E-utilities),
Europe PMC (citation counts), NCBI MeSH (descriptor resolution), ChEMBL (molecule metadata),
openFDA (drug labels).

---

## Sample output

A trimmed top-candidate block from `scout find -d "semaglutide"` (full report in `snapshots/`):

```
1. Cardiovascular Disorder
   Stage        Active Phase 3 development on record (NCT05669755)
   Literature   strong, supports, RCT-backed / controlled
   Assessment   Maturing, awaiting readout

2. Kidney Disorder
   Stage        Early-phase only, no completed pivotal readout
   Literature   strong, supports, RCT-backed, ⚠️ safety signal reported for this indication
   Assessment   Live but bottlenecked
```

Each candidate carries a stage, graded literature strength/direction, a safety flag, and a
one-line assessment. See [GLOSSARY.md](GLOSSARY.md) for how each field is derived.

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/dami-gupta-git/indication_scout.git
cd indication_scout
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
| `ANTHROPIC_API_KEY` | Yes¹ | Anthropic API key for Claude LLM calls |
| `NCBI_API_KEY` | No | NCBI API key for PubMed (increases rate limits) |
| `PUBMED_API_KEY` | No | PubMed API key (separate from NCBI key in config) |
| `OPENFDA_API_KEY` | No | OpenFDA API key |
| `LLM_MODEL` | No | Primary LLM model (default: `claude-sonnet-4-6`) |
| `SMALL_LLM_MODEL` | No | Lightweight LLM model (default: `claude-haiku-4-5-20251001`) |
| `EMBEDDING_MODEL` | No | Embedding model (default: `FremyCompany/BioLORD-2023`) |

¹ `ANTHROPIC_API_KEY` is required at runtime, not at startup: the app boots with an empty key
and fails on the first Claude call. Set it before running any analysis.

The project requires a PostgreSQL database with the `pgvector` extension for storing PubMed abstract embeddings. See `docs/rag.md` for the Docker setup.

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
scout find -d "metformin" --date-before 2020-01-01 # temporal holdout: only evidence dated before the cutoff (see Temporal Holdout below)
scout investigate -d "metformin" -i "alzheimer disease"  # run the pipeline on a fixed drug+disease pair (skips candidate discovery)
scout --help
```

> **Use the generic (INN) name, not a brand name.** Only the FDA-approval check resolves
> brand → generic; the ClinicalTrials.gov and PubMed queries use the name you type verbatim. A
> brand name silently misses evidence — e.g. `wegovy` misses ~84% of the trials `semaglutide`
> finds, `vioxx` ~38% vs `rofecoxib`.

#### Fixed pair (`investigate`)

`scout investigate -d <drug> -i <indication>` runs the pipeline on a drug+disease pair you
specify, skipping candidate discovery: no competitor or mechanism-surfaced diseases and no
ranking. It still runs the mechanism (MoA), literature, and clinical-trials sub-agents for the
pair and renders the same report as `find`. Useful for inspecting the literature and
clinical-trial evidence for a hypothesis that Open Targets would not surface as a candidate.
CLI-only; report is written to `./snapshots` (or `--out-dir`), and `--no-write` prints to stdout.
No JSON payload is saved, so `scout render` is not available for these runs. Supports
`--date-before YYYY-MM-DD` for a temporal holdout (same semantics as `find`; see below).

#### Temporal holdout (`--date-before`)

`--date-before YYYY-MM-DD` runs IndicationScout "as of" a past cutoff: PubMed and
ClinicalTrials.gov queries are restricted to records dated strictly before the cutoff, and
trials that complete on/after the cutoff have their outcome fields scrubbed so they appear
still-in-progress. This simulates whether the system would have surfaced a repurposing
opportunity that was only validated later. Mechanism (Open Targets) data has no date filter and
is always current. Holdout reports are written to `snapshots/holdouts/` as
`<drug>_holdout_<cutoff>_<timestamp>.md` with a `> **HOLDOUT**` banner. See
[holdout.md](holdout.md) for the full per-layer methodology.

### Web UI (React + FastAPI)

The web app is two processes: a **FastAPI backend** (the API + agent runner) and a
**Vite/React frontend** (the UI). In development they run side by side — Vite serves the
UI on port 5173 and proxies all `/api` requests to uvicorn on port 8000. You open the
**frontend** URL in your browser; the proxy handles the rest.

> For the full dev workflow (native vs Dockerized, rebuild rules), see
> [DEV_SETUP.md](DEV_SETUP.md).

**First time only** — install the frontend dependencies:

```bash
cd frontend
npm install
```

**Run both servers** (two terminals):

```bash
# Terminal 1 — backend (port 8000)
uvicorn indication_scout.api.main:app --reload

# Terminal 2 — frontend (port 5173)
cd frontend
npm run dev
```

Then open **http://localhost:5173**, enter a drug name, and click **Analyze**. The UI
submits the run, polls for status, and renders the result across four tabs (Overview,
Mechanism, Clinical Trials, Literature). A real run takes several minutes — it queries
the live data sources and calls the LLM, so `ANTHROPIC_API_KEY` and both env files
(`.env`, `.env.constants`) must be set up first (see Environment Setup above).

> Single-worker only: jobs are held in an in-memory store keyed by job id, lost on
> restart. Run uvicorn with a single worker (the default with `--reload`).

The async job API the frontend uses:

| Method & path | Purpose |
|---------------|---------|
| `POST /api/analyses` | Start a run for a drug; returns `202` with a `job_id`. |
| `GET /api/analyses/{job_id}` | Poll status; returns the `SupervisorOutput` payload when done. |
| `GET /api/analyses/{job_id}/report.md` | The formatted Markdown report for a finished run. |
| `DELETE /api/analyses/{job_id}` | Cancel a running job. |

**Production build** — `npm run build` (in `frontend/`) emits a static bundle that
FastAPI serves directly; uvicorn alone then hosts both the API and the UI on port 8000.

## Development

> **[DEV_SETUP.md](DEV_SETUP.md)** is the authoritative dev-workflow doc — native vs Dockerized
> dev, command quick-reference tables, test/rebuild rules. The commands below are a quick summary.

### Running Tests

```bash
# All tests (regression + live tests are excluded by default)
pytest

# Integration tests only
pytest tests/integration/

# Unit tests only
pytest tests/unit/
```

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
├── agents/          # Supervisor + literature, clinical_trials, mechanism sub-agents
├── api/             # FastAPI app (async analyses API; serves the React frontend in prod)
├── cli/             # Click-based `scout` CLI
├── data_sources/    # Async API clients (Open Targets, CT.gov, PubMed, Europe PMC, ChEMBL, FDA)
├── models/          # Pydantic data contracts
├── prompts/         # LLM prompt templates
├── regression/      # Snapshot regression harness (backs `scout diff-report`)
├── report/          # SupervisorOutput → markdown
├── services/        # Business logic (LLM, embeddings, RAG, disease/FDA resolution, job store)
├── ml_models/       # Trial-risk + trial-success modeling
├── config.py        # Settings (pydantic-settings, .env + .env.constants)
└── constants.py     # URLs, timeouts, lookup maps
```

For a fully annotated tree and the data contracts, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Known Limitations

- **Generic name required**: Only the FDA-approval check resolves brand → generic; ClinicalTrials.gov and PubMed queries use the typed drug string verbatim. Entering a brand name silently loses evidence and shifts downstream ranking (e.g. `wegovy` misses ~84% of trials vs `semaglutide`, `vioxx` ~38% vs `rofecoxib`). Type the INN.
- **Best for focused-target drugs**: Drugs with specific molecular targets (semaglutide, metformin, imatinib) yield clean signal. Pleiotropic drugs (aspirin, corticosteroids, broad-spectrum antibiotics) surface noisy candidates via their many weak associations — treat those runs as exploratory.
- **Abstract-only indexing**: PubMed articles without an abstract (letters, editorials, conference summaries) are excluded from the vector store and will not appear in semantic search results. Only articles with a non-empty abstract are embedded and cached.
- **Incomplete Open Targets approval data**: Open Targets does not record all approved indications for every drug. Approved indications missing from Open Targets will not be filtered from repurposing candidates and may appear as false positives. For example, tofacitinib's ulcerative colitis and ankylosing spondylitis approvals are absent from Open Targets, causing them to appear as repurposing candidates.

## Citations

This project relies on the following public data sources and models:

- **Open Targets** — Ochoa, D. et al. (2023). The next-generation Open Targets Platform:
  reimagined, redesigned, rebuilt. *Nucleic Acids Research*, 51(D1), D1353–D1359.
  DOI: [10.1093/nar/gkac1046](https://doi.org/10.1093/nar/gkac1046).
- **ClinicalTrials.gov** — National Library of Medicine (NLM). ClinicalTrials.gov.
  <https://clinicaltrials.gov> (API v2).
- **PubMed / NCBI E-utilities** — Sayers, E.W. et al. (2024). Database resources of the
  National Center for Biotechnology Information. *Nucleic Acids Research*, 52(D1), D33–D43.
  DOI: [10.1093/nar/gkad1044](https://doi.org/10.1093/nar/gkad1044).
- **ChEMBL** — Zdrazil, B. et al. (2024). The ChEMBL Database in 2023: a drug discovery
  platform spanning multiple bioactivity data types and time periods. *Nucleic Acids Research*,
  52(D1), D1180–D1192. DOI: [10.1093/nar/gkad1004](https://doi.org/10.1093/nar/gkad1004).
- **openFDA** — Kass-Hout, T.A. et al. (2016). OpenFDA: an innovative platform providing
  access to a wealth of FDA's publicly available data. *Journal of the American Medical
  Informatics Association*, 23(3), 596–600. DOI: [10.1093/jamia/ocv153](https://doi.org/10.1093/jamia/ocv153).
- **BioLORD-2023** — Remy, F. et al. (2024). BioLORD-2023: semantic textual representations
  fusing large language models and clinical knowledge graph insights. *Journal of the American
  Medical Informatics Association*, 31(9), 1844–1855. DOI: [10.1093/jamia/ocae029](https://doi.org/10.1093/jamia/ocae029).

## Acknowledgments

This codebase was created with the help of [Claude Code](https://claude.com/claude-code).

## License

MIT