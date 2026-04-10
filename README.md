# IndicationScout

An agentic system for discovering drug repurposing opportunities.

## Overview

IndicationScout is an agentic drug repurposing system. A drug name goes in; coordinated AI agents query multiple biomedical data sources and produce a repurposing report.

A **Supervisor** agent orchestrates three specialist sub-agents:

- **Literature agent** — Queries PubMed via EUtils, then runs a RAG pipeline: fetch abstracts, embed with BioLORD-2023, semantic search, and LLM-based synthesis of evidence for each candidate disease.
- **Clinical Trials agent** — Queries ClinicalTrials.gov v2 (REST) to detect whitespace (indications with no active trials), search for relevant trials, map the competitive landscape, and surface terminated trials.
- **Mechanism agent** — Queries Open Targets target-level data (GraphQL) to retrieve disease associations with evidence scores and Reactome pathway annotations.

The Supervisor first calls `find_candidates` (Open Targets competitor analysis) and `analyze_mechanism` in parallel, then delegates to the Literature and Clinical Trials agents per candidate disease.

Data sources:

- Open Targets (GraphQL) — drug targets, disease associations, competitor drugs
- ClinicalTrials.gov (REST v2) — trial search, whitespace detection, competitive landscape
- PubMed (NCBI EUtils) — literature retrieval and abstract indexing
- ChEMBL — molecule metadata and ATC classifications

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

Copy the example environment file and fill in your API keys:

```bash
cp .env.example .env
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

> **Note:** The CLI entry point (`scout find -d "metformin"`) is defined in `pyproject.toml` but the CLI module (`indication_scout.cli.cli`) has not been implemented yet. The API can be started with:

```bash
uvicorn indication_scout.api.main:app --reload
```

## Development

### Running Tests

```bash
# All tests
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
│   ├── supervisor/      # Supervisor agent (orchestrates sub-agents) — supervisor_agent.py, supervisor_tools.py, supervisor_output.py
│   ├── literature/      # Literature agent (PubMed RAG) — literature_agent.py, literature_tools.py, literature_output.py
│   ├── clinical_trials/ # Clinical Trials agent (ClinicalTrials.gov) — clinical_trials_agent.py, clinical_trials_tools.py, clinical_trials_output.py
│   ├── mechanism/       # Mechanism agent (Open Targets targets) — mechanism_agent.py, mechanism_tools.py, mechanism_output.py
│   └── orchestrator.py  # Stub (not yet implemented)
├── api/             # FastAPI application (main.py, routes/, schemas/) -- /health endpoint only
├── data_sources/    # Async API clients (OpenTargets, ClinicalTrials.gov, PubMed, ChEMBL, DrugBank stub)
├── db/              # SQLAlchemy session factory and declarative base
├── helpers/         # Utility functions (drug name normalization)
├── markers.py       # Code review exclusion markers (@no_review decorator)
├── models/          # Pydantic data contracts (model_open_targets, model_clinical_trials, model_pubmed_abstract, model_chembl, model_drug_profile, model_evidence_summary)
├── prompts/         # LLM prompt templates (extract_organ_term, expand_search_terms, disease_synonyms, synthesize)
├── runners/         # Pipeline runners (rag_runner.py) and exploration scripts (pubmed_runner.py)
├── services/        # Business logic -- LLM calls (llm.py, parse_llm_response), embeddings (embeddings.py), disease normalization (disease_helper.py), PubMed query building (pubmed_query.py), RAG pipeline (retrieval.py -- fetch_and_cache, semantic_search, synthesize)
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

## License

MIT
