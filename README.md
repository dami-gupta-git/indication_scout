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

## Usage

```bash
scout find -d "metformin"
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

### Code Formatting

```bash
black src/
```

## Project Structure

```
src/indication_scout/
├── agents/          # AI agents (orchestrator, literature, clinical_trials, mechanism, safety) -- all stubs
├── api/             # FastAPI application (main.py, routes/, schemas/) -- /health endpoint only
├── data_sources/    # Async API clients (OpenTargets, ClinicalTrials.gov, PubMed, ChEMBL, DrugBank)
├── db/              # SQLAlchemy session factory
├── helpers/         # Utility functions (drug name normalization)
├── models/          # Pydantic data contracts (model_open_targets, model_clinical_trials, model_pubmed_abstract, model_chembl, model_drug_profile)
├── prompts/         # LLM prompt templates (extract_organ_term, expand_search_terms, disease_synonyms)
├── runners/         # Standalone runner scripts
├── scripts/         # Session management, exploratory pipeline scripts
├── services/        # Business logic -- LLM calls, embeddings, disease normalization, PubMed query building, retrieval (partial)
├── sqlalchemy/      # SQLAlchemy ORM models (pubmed_abstracts with pgvector embedding)
├── utils/           # Shared file-based cache utility (cache_key, cache_get, cache_set)
├── config.py        # Settings via pydantic-settings, loaded from .env
└── constants.py     # URLs, timeouts, lookup maps
```

### Citations
Open Targets: Ochoa, D. et al. (2023). The next-generation Open Targets Platform: reimagined, redesigned, rebuilt. 
Nucleic Acids Research, 51(D1), D1353–D1359. DOI: 10.1093/nar/gkac1037. 

## License

MIT
