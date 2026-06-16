# IndicationScout

An agentic system for discovering drug repurposing opportunities.

**Live demo:** <https://indicationscout-production.up.railway.app/>

## Overview

IndicationScout is an agentic drug repurposing system. A drug name goes in; coordinated AI agents query multiple biomedical data sources and produce a repurposing report.

IndicationScout pulls **live evidence** from multiple biomedical databases at query time — no static knowledge base, no precomputed answers. For each candidate indication it gathers current trial activity, recent literature, mechanistic associations, and regulatory status, then synthesizes them into a single repurposing report that characterizes the *state of the hypothesis* (live, stalled, niche, untested, post-readout-and-stuck) per disease.

A **Supervisor** agent orchestrates three specialist sub-agents:

- **Literature agent** — Queries PubMed via EUtils, then runs a RAG pipeline: fetch abstracts, embed with BioLORD-2023, semantic search, and LLM-based synthesis of evidence for each candidate disease.
- **Clinical Trials agent** — Queries ClinicalTrials.gov v2 (REST) per drug × indication pair: all-status search (whitespace verdict + per-status counts), completed-trial query (with Phase 3 count), terminated-trial query (with `why_stopped` text classified into safety / efficacy / business / enrollment categories at the tool layer), competitive landscape for the indication, and an FDA-label approval check (the candidate is still investigated fully even when the drug is already approved — the approval status is recorded for the supervisor rather than short-circuiting the analysis). Indications are resolved to a MeSH descriptor via NCBI E-utilities and the resolved preferred term is fed to CT.gov's server-side `AREA[ConditionMeshTerm]"<term>"` filter — so e.g. "hypertension" trials aren't mixed in with unrelated free-text matches like glaucoma. (Known bug: `AREA[ConditionMeshTerm]` also matches on MeSH ancestors, so a parent term like "hypertension" can still pull in pulmonary-hypertension trials; the post-filter fix is not yet implemented.)
- **Mechanism agent** — Queries Open Targets target-level data (GraphQL) to retrieve disease associations with evidence scores and Reactome pathway annotations.

The Supervisor first calls `find_candidates` (Open Targets competitor analysis) and `analyze_mechanism` in parallel, then delegates to the Literature and Clinical Trials agents per candidate disease.

### Top-5 candidate blurbs

After investigating candidates, the Supervisor produces a ranked Summary. For each of the **top 5 ranked candidates**, the Supervisor also writes a **2-sentence interpretive blurb** characterizing the *state of the hypothesis* — live, stalled, niche, untested, post-readout-and-stuck, etc. — grounded in the literature and clinical-trials sub-agent summaries seen during that run. Mechanism content is intentionally excluded from the blurb to keep it focused on clinical evidence. Before finalizing, the Supervisor must call `critique_ranking` once — a separate reviewer LLM audits the ranking order (closed/adverse-signal candidates must not outrank live ones) and repairs specific factual contradictions in the blurb fields; `finalize_supervisor` is rejected until it has run. The blurbs are then emitted by `finalize_supervisor` and rendered inline under each disease in the Summary section of both the Markdown report and the web UI Overview tab. 

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
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key for Claude LLM calls |
| `NCBI_API_KEY` | No | NCBI API key for PubMed (increases rate limits) |
| `PUBMED_API_KEY` | No | PubMed API key (separate from NCBI key in config) |
| `OPENFDA_API_KEY` | No | OpenFDA API key |
| `LLM_MODEL` | No | Primary LLM model (default: `claude-sonnet-4-6`) |
| `SMALL_LLM_MODEL` | No | Lightweight LLM model (default: `claude-haiku-4-5-20251001`) |
| `EMBEDDING_MODEL` | No | Embedding model (default: `FremyCompany/BioLORD-2023`) |

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
scout --help
```

### Web UI (React + FastAPI)

The web app is two processes: a **FastAPI backend** (the API + agent runner) and a
**Vite/React frontend** (the UI). In development they run side by side — Vite serves the
UI on port 5173 and proxies all `/api` requests to uvicorn on port 8000. You open the
**frontend** URL in your browser; the proxy handles the rest.

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
├── agents/          # AI agents
│   ├── base.py             # BaseAgent abstract class (currently unused by ReAct agents)
│   ├── _trial_formatting.py # Shared trial formatting helpers
│   ├── supervisor/         # Supervisor agent (orchestrates sub-agents) — supervisor_agent.py, supervisor_tools.py, supervisor_output.py
│   ├── literature/         # Literature agent (PubMed RAG) — literature_agent.py, literature_tools.py, literature_output.py
│   ├── clinical_trials/    # Clinical Trials agent (ClinicalTrials.gov + MeSH post-filter) — clinical_trials_agent.py, clinical_trials_tools.py, clinical_trials_output.py
│   └── mechanism/          # Mechanism agent (Open Targets targets) — mechanism_agent.py, mechanism_tools.py, mechanism_output.py, mechanism_candidates.py, mechanism_row_builder.py
├── api/             # FastAPI application (main.py, routes/, schemas/) — /health + async analyses routes (POST/GET/report.md/DELETE), drill-down routes, and example-cache routes; serves the built React frontend in prod
├── cli/             # Click-based CLI (cli.py) — exposes the `scout` command
├── data_sources/    # Async API clients (OpenTargets, ClinicalTrials.gov, PubMed, ChEMBL, FDA, DrugBank stub)
├── db/              # SQLAlchemy session factory and declarative base
├── helpers/         # Utility functions (drug name normalization)
├── markers.py       # Code review exclusion markers (@no_review decorator)
├── ml_models/       # ML modeling code
│   ├── trial_risk/         # Trial-risk model (data.py, features.py, literature.py, score.py, train.py, inspect.py)
│   └── success_classifier/ # Trial-success classifier (features.py, labels.py)
├── models/          # Pydantic data contracts (model_open_targets, model_clinical_trials, model_pubmed_abstract, model_chembl, model_drug_profile, model_evidence_summary)
├── prompts/         # LLM prompt templates (supervisor, literature, clinical_trials, synthesize, expand_search_terms, extract_fda_approvals, extract_fda_approval_single, list_label_indications, extract_organ_term, merge_diseases, normalize_disease, normalize_disease_batch)
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