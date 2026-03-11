# IndicationScout

An agentic system for discovering drug repurposing opportunities.

## Overview

IndicationScout uses multiple AI agents to analyze drugs and identify potential new therapeutic indications by aggregating evidence from:

- Scientific literature (PubMed)
- Clinical trials (ClinicalTrials.gov)
- Drug databases (DrugBank, ChEMBL, Open Targets)
- Mechanism of action analysis

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

### Database Migrations

```bash
alembic upgrade head
```

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
├── agents/          # AI agents (orchestrator, literature, clinical_trials, mechanism, safety) -- all stubs
├── api/             # FastAPI application (main.py, routes/, schemas/) -- /health endpoint only
├── data_sources/    # Async API clients (OpenTargets, ClinicalTrials.gov, PubMed, ChEMBL, DrugBank)
├── db/              # SQLAlchemy session factory and declarative base
├── helpers/         # Utility functions (drug name normalization)
├── markers.py       # Code review exclusion markers (@no_review decorator)
├── models/          # Pydantic data contracts (model_open_targets, model_clinical_trials, model_pubmed_abstract, model_chembl, model_drug_profile, model_evidence_summary)
├── prompts/         # LLM prompt templates (extract_organ_term, expand_search_terms, disease_synonyms, synthesize)
├── runners/         # Pipeline runners (rag_runner.py) and exploration scripts (pubmed_runner.py)
├── services/        # Business logic -- LLM calls (llm.py, parse_llm_response), embeddings (embeddings.py), disease normalization (disease_normalizer.py), PubMed query building (pubmed_query.py), RAG pipeline (retrieval.py -- fetch_and_cache, semantic_search, synthesize)
├── sqlalchemy/      # SQLAlchemy ORM models (pubmed_abstracts with pgvector embedding)
├── utils/           # Shared file-based cache utility (cache_key, cache_get, cache_set)
├── config.py        # Settings via pydantic-settings, loaded from .env
└── constants.py     # URLs, timeouts, lookup maps
```

## Known Limitations

- **Abstract-only indexing**: PubMed articles without an abstract (letters, editorials, conference summaries) are excluded from the vector store and will not appear in semantic search results. Only articles with a non-empty abstract are embedded and cached.

### Citations
Open Targets: Ochoa, D. et al. (2023). The next-generation Open Targets Platform: reimagined, redesigned, rebuilt. 
Nucleic Acids Research, 51(D1), D1353–D1359. DOI: 10.1093/nar/gkac1037. 

## License

MIT
