# IndicationScout — Project State

> Auto-updated. Do not edit manually. Run /memory to refresh.

## What This Is

IndicationScout is an agentic drug repurposing system. A drug name goes in; coordinated AI agents query multiple biomedical data sources and produce a repurposing report identifying candidate indications not yet approved or explored.

## Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| `data_sources/base_client.py` | Complete | Async HTTP client with retry/backoff, REST GET, GraphQL POST, XML GET |
| `data_sources/open_targets.py` | Complete | Full GraphQL client; drug, target, disease, disease synonyms; file-based cache |
| `data_sources/clinical_trials.py` | Complete | ClinicalTrials.gov v2 REST; 5 public methods; whitespace detection, landscape, terminated |
| `data_sources/pubmed.py` | Complete | PubMed ESearch/EFetch; search (cached), count, fetch abstracts from XML; file-based cache on `search()` |
| `data_sources/chembl.py` | Complete | `get_molecule` + `get_atc_description` both complete; `__init__(cache_dir)` consistent with other clients; both methods use `self.cache_dir` |
| `data_sources/drugbank.py` | Stub | `get_drug` and `get_interactions` both raise `NotImplementedError` |
| `models/model_chembl.py` | Complete | `MoleculeData` + `ATCDescription` (10 fields: level1–5, all _description fields, who_name) |
| `models/model_open_targets.py` | Complete | Full Pydantic contract for all OT data types |
| `models/model_clinical_trials.py` | Complete | Full Pydantic contract for trial, whitespace, landscape, terminated |
| `models/model_pubmed_abstract.py` | Complete | `PubmedAbstract` model |
| `models/model_drug_profile.py` | Complete | `DrugProfile` + `from_rich_drug_data(rich, atc_descriptions=None)` factory; `atc_descriptions` is optional (None → `[]`); 7 fields; dedup via `dict.fromkeys`; comments explain rich.targets vs rich.drug.targets distinction and why only ATC level3/4 descriptions are used |
| `agents/orchestrator.py` | Stub | `run()` raises `NotImplementedError` |
| `agents/literature.py` | Stub | `run()` raises `NotImplementedError` |
| `agents/clinical_trials.py` | Stub | `run()` raises `NotImplementedError` |
| `agents/mechanism.py` | Stub | `run()` raises `NotImplementedError` |
| `agents/safety.py` | Stub | `run()` raises `NotImplementedError` |
| `services/llm.py` | Complete | `query_llm` and `query_small_llm` via Anthropic SDK |
| `services/disease_normalizer.py` | Complete | LLM normalization; blocklist guard; PubMed count verification; file-based cache for both LLM results and PubMed counts |
| `services/pubmed_query.py` | Complete | Builds PubMed queries by normalizing disease name and combining with drug name |
| `services/retrieval.py` | Partial | `build_drug_profile`, `get_disease_synonyms`, `extract_organ_term`, `expand_search_terms`, `get_stored_pmids`, `fetch_new_abstracts` all implemented; `fetch_and_cache`, `semantic_search`, `synthesize` still raise `NotImplementedError` |
| `sqlalchemy/pubmed_abstracts.py` | Complete | SQLAlchemy ORM model with pgvector embedding column (768 dims) |
| `db/session.py` | Complete | SQLAlchemy session factory; `get_db()` dependency |
| `api/main.py` | Partial | FastAPI app with `/health` endpoint only; `api/routes/` and `api/schemas/` subdirs contain only `__init__.py` |
| `utils/cache.py` | Complete | Shared file-based cache utility: `cache_key`, `cache_get`, `cache_set`; used by all data source clients and services |
| `helpers/drug_helpers.py` | Complete | `normalize_drug_name` strips common salt suffixes |
| `prompts/extract_organ_term.txt` | Complete | Haiku prompt; Input/Output few-shot format; returns single organ term |
| `prompts/expand_search_terms.txt` | Complete | Full prompt with 9 template variables; 5-axis structure with per-axis caps; JSON array output |
| `prompts/disease_synonyms.txt` | Complete | Input/Output few-shot format (updated to match other prompts) |
| `scripts/session.py` | Complete | Session file manager: `startup` (create/rotate/print) and `append` subcommands; rotation to `session_bak/` with 5-file cap |
| `scripts/open_target_pipeline.py` | Complete | Exploratory pipeline: fetches bupropion competitors, builds PubMed queries, fetches abstracts; proper async with client context managers, `asyncio.run` under `__main__`, uses logging |
| `runners/pubmed_runner.py` | Partial | Development/exploration script; uses `print()` which violates project rules |
| `config.py` | Complete | Pydantic settings from `.env`; LLM model, DB URL, API keys |
| `constants.py` | Complete | Timeout, cache TTL, OT URL, interaction type map, stop-reason keywords |

## Architecture & Data Flow

The system is organized into five layers. Agents sit above data sources and communicate only through Pydantic models; they never receive raw API responses.

```
CLI / API
    |
Orchestrator (agents/orchestrator.py)  — stub
    |
Specialist agents: literature, clinical_trials, mechanism, safety  — all stubs
    |
Services layer: llm.py, disease_normalizer.py, pubmed_query.py, retrieval.py
    |
Data source clients: OpenTargetsClient, ClinicalTrialsClient, PubMedClient, ChEMBLClient
    |
Pydantic models (models/)  — contract boundary; agents receive these, never raw dicts
    |
External APIs: Open Targets GraphQL, ClinicalTrials.gov v2 REST, PubMed EUtils, ChEMBL REST
```

The database layer (PostgreSQL + pgvector) is used for caching PubMed abstracts with their embeddings. The embedding model is `FremyCompany/BioLORD-2023` (768 dims), loaded via `sentence-transformers` (added to runtime deps in `pyproject.toml`). The Open Targets client and the disease normalizer service both use a separate file-based cache (`_cache/` dir, SHA-256-keyed JSON files, 5-day TTL) that does not require the database.

## Key Paths

| Path | Purpose |
|------|---------|
| `src/indication_scout/data_sources/open_targets.py` | Most complex client; GraphQL queries are defined as module-level strings at the bottom of the file |
| `src/indication_scout/data_sources/clinical_trials.py` | `detect_whitespace()` runs 3 concurrent API calls; `get_landscape()` aggregates trials into a competitive map |
| `src/indication_scout/services/disease_normalizer.py` | LLM-driven disease term normalization; two-step strategy (normalize then verify/broaden); uses `cache_get`/`cache_set` from `utils/cache.py` |
| `src/indication_scout/services/retrieval.py` | RAG pipeline; `get_disease_synonyms`, `extract_organ_term`, `expand_search_terms` all implemented; `fetch_and_cache`, `semantic_search`, `synthesize` still stubbed |
| `src/indication_scout/models/model_drug_profile.py` | `DrugProfile` — flat LLM-facing projection of `RichDrugData`; built via `from_rich_drug_data(rich, atc_descriptions)` |
| `src/indication_scout/prompts/` | All LLM prompt templates; `extract_organ_term.txt`, `expand_search_terms.txt`, `disease_synonyms.txt` |
| `src/indication_scout/sqlalchemy/pubmed_abstracts.py` | ORM model for the `pubmed_abstracts` pgvector table |
| `scripts/open_target_pipeline.py` | Exploratory async pipeline script; not part of production path |
| `runners/pubmed_runner.py` | Development/exploration script; uses `print()` in violation of project rules |
| `tests/integration/data_sources/test_open_targets.py` | Extensive integration suite with exact field assertions; doubles as API contract verification |
| `tests/integration/services/test_pubmed_query.py` | Parametrized integration tests for `get_pubmed_query`; 5 drug-disease pairs; asserts query structure and disease keyword presence |
| `tests/integration/services/test_retrieval.py` | Integration tests for retrieval service; `test_get_disease_synonyms`, `test_extract_organ_term_returns_string`, `test_expand_search_terms_returns_queries`, `test_build_drug_profile` (parametrized), `test_get_stored_pmids_returns_only_inserted_pmids`, `test_fetch_new_abstracts_skips_stored_pmid`, `test_fetch_new_abstracts_all_stored_skips_network` |
| `tests/unit/services/test_retrieval.py` | Unit tests for retrieval service (no network); covers `DrugProfile.from_rich_drug_data`, `extract_organ_term`, `expand_search_terms`, `build_drug_profile`, `get_stored_pmids` (5 cases), `fetch_new_abstracts` (3 parametrized cases) |
| `tests/integration/data_sources/test_pubmed.py` | Integration tests for PubMedClient; includes `test_search_returns_known_pmids` parametrized with verified PMID sets per query (e.g. sildenafil + diabetic nephropathy → 4 known PMIDs) |
| `tests/unit/data_sources/test_chembl.py` | Unit tests for ChEMBLClient; covers `get_molecule` (parametrized, 5 cases) and `get_atc_description` (parametrized, 2 cases, all 10 fields) |
| `docs/open_targets.md` | Full data contract documentation for Open Targets client |
| `docs/chembl.md` | Full data contract documentation for ChEMBL client (method, field mapping, agent usage) |
| `docs/api_clients.md` | Quick-reference guide for all data source clients; includes ChEMBLClient section |
| `docs/rag_details.md` | RAG pipeline implementation details; disease normalizer strategy documented in section 4 |
| `src/indication_scout/utils/cache.py` | Shared file-based cache utility; `cache_key`, `cache_get`, `cache_set` |
| `planning_docs/` | Sprint plans and technical implementation docs |

## Non-Obvious Patterns & Decisions

- The Open Targets client resolves drug names to ChEMBL IDs via a GraphQL search before fetching drug data. Names are case-insensitive and fuzzy; the first match wins.
- `get_rich_drug_data()` fetches all target data in parallel using `asyncio.gather`, benefiting from the file cache for repeated calls.
- `RichDrugData` (not `DrugData`) is the canonical input type for any agent or service that needs a complete drug profile. `DrugData` contains only drug-level metadata (name, synonyms, ATC codes, mechanisms). `RichDrugData` bundles `DrugData` with full `TargetData` for every target (pathways, disease associations, interactions, expression, safety liabilities). Always use `RichDrugData` when downstream code needs target-level context.
- `DrugProfile` (`models/model_drug_profile.py`) is a flat LLM-facing projection of `RichDrugData`: name, synonyms, target gene symbols, mechanisms of action, ATC codes, ATC descriptions (human-readable from `/atc_class/{code}.json`), drug type. Built via `DrugProfile.from_rich_drug_data(rich, atc_descriptions=None)` — `atc_descriptions` is optional; omitting it (or passing `[]`) yields `atc_descriptions=[]` on the profile. The preferred way to assemble a profile is via `build_drug_profile(drug_name)` in `services/retrieval.py`, which orchestrates the Open Targets + ChEMBL calls. Used as the typed input to `expand_search_terms`. Note: `target_gene_symbols` comes from `rich.targets` (list[TargetData], has `.symbol`); `mechanisms_of_action` comes from `rich.drug.targets` (list[DrugTarget], has `.mechanism_of_action`) — different collections, both needed.
- `DrugProfile.atc_descriptions` uses only ATC level3 and level4 descriptions (e.g. "Blood glucose lowering drugs" + "Biguanides"). Level1/2 are too broad for useful PubMed queries and are intentionally excluded.
- `expand_search_terms` generates diverse PubMed queries across 5 axes (drug name, drug class+organ, mechanism+organ, target gene, synonym) with per-axis caps (total 5–10). Organ term is pre-extracted via a separate Haiku call (`extract_organ_term`). Both functions cache results under `"organ_term"` and `"expand_search_terms"` namespaces. LLM output varies in disease synonym wording — integration tests use axis-level keyword assertions (not exact-match) for `expand_search_terms`; `extract_organ_term` uses exact-match.
- All three prompt files use `Input: "{var}"` / `Output:` few-shot format to anchor the LLM completion.
- `detect_whitespace()` runs three concurrent API calls (exact match, drug-only count, condition-only count) then only fetches full condition trials if whitespace exists. When whitespace exists, `condition_drugs` is populated with Phase 2+ competitors ranked by phase then active status.
- The ClinicalTrials v2 API uses `AREA[Phase]` syntax for phase filtering embedded in `query.term`, not a dedicated parameter.
- The `DiseaseSynonyms.all_synonyms` property deliberately excludes `broad` and `narrow` synonyms; it only returns `exact + related + parent_names`.
- The disease normalizer has a two-step LLM strategy: first normalize, then if PubMed hit count is below `MIN_RESULTS` (3), ask the LLM to generalize further. Both steps are blocked if the result is an over-generic term in `BROADENING_BLOCKLIST`.
- The disease normalizer caches two types of results in `_cache/`: LLM-normalized terms under namespace `"disease_norm"` (key: `raw_term`) and PubMed result counts under namespace `"pubmed_count"` (key: full query string). Both use `cache_get`/`cache_set` from `utils/cache.py` with `DEFAULT_CACHE_DIR` passed explicitly.
- `PubMedClient.search()` now caches PMID lists under namespace `"pubmed_search"` (key: `query + max_results + date_before`) using the same file-based SHA-256 pattern. `get_count()` and `fetch_abstracts()` are not cached.
- `pubmed_query.get_pubmed_query()` returns a list of queries (one per disease term when the normalized result contains `OR`), not a single string.
- The `stop_category` field on `TerminatedTrial` is a keyword-based pre-classification; the docstring notes that LLM refinement is intended to happen at the agent layer (not yet implemented).
- LLM model: `claude-sonnet-4-6` for main calls, `claude-haiku-4-5-20251001` for lightweight calls (e.g. disease normalization, organ term extraction).
- The `get_drug_competitors()` method on `OpenTargetsClient` has a `# TODO needs rework` comment; it is implemented but may produce inconsistent results.
- `api/routes/` and `api/schemas/` subdirectories exist but contain only `__init__.py` — no routes or schemas are defined yet.
- The CLI entry point `scout find -d "metformin"` is referenced in pyproject.toml scripts but the CLI module `indication_scout.cli.cli` does not exist yet.
- File-based cache utility (`cache_get`, `cache_set`, `cache_key`) lives in `utils/cache.py` and is shared by all callers. All data source clients and services import from there — no per-module duplication. Callers pass `cache_dir` explicitly; the utility never silently no-ops. Active namespaces: `"drug"`, `"target"`, `"disease_drugs"`, `"disease_synonyms"` (OpenTargets), `"pubmed_search"` (PubMed), `"disease_norm"`, `"pubmed_count"` (disease_normalizer), `"atc_description"` (ChEMBL), `"organ_term"`, `"expand_search_terms"` (retrieval.py).
- Unit tests that touch the file-based cache must pass `tmp_path` as `cache_dir` (or patch `DEFAULT_CACHE_DIR`) to avoid contamination from real cached results left by integration test runs.

## Delta — 2026-03-01 (Pydantic defensive defaults rollout)

- Applied `coerce_nones` `model_validator` to all 5 model files: `model_chembl.py`, `model_clinical_trials.py`, `model_open_targets.py`, `model_pubmed_abstract.py`, `model_drug_profile.py`. Every model that ingests external API data now follows the CLAUDE.md defensive defaults pattern.
- Previously-required fields that are structural identifiers (e.g. `nct_id`, `title`, `phase`) now default to `""` so API `null` is coerced cleanly.
- Fields with scientific meaning that can be absent (e.g. `overall_score`, `count`, `log_likelihood_ratio`) changed to `T | None = None`.
- `Trial.results_posted` changed from `bool = False` to `bool | None = None` — `False` is a real value, not a safe missing-data default.
- Unit tests in `tests/unit/models/test_clinical_trials_models.py` updated: tests that previously asserted `ValidationError` on missing fields now assert empty-string defaults instead.

## Known Issues / Caveats

- All five agents (`Orchestrator`, `LiteratureAgent`, `ClinicalTrialsAgent`, `MechanismAgent`, `SafetyAgent`) raise `NotImplementedError` in their `run()` methods — the agent layer is completely unimplemented.
- The RAG pipeline in `services/retrieval.py` is partially implemented: `fetch_and_cache`, `semantic_search`, `synthesize` all raise `NotImplementedError`. Stage 0 (`expand_search_terms` + `extract_organ_term`) is complete. Stage 1 steps 1–3 are complete (`embed`, `get_stored_pmids`, `fetch_new_abstracts`).
- `DrugBankClient` is a stub (`get_drug` and `get_interactions` raise `NotImplementedError`).
- The CLI module referenced in `pyproject.toml` (`indication_scout.cli.cli`) does not exist.
- `tests/integration/data_sources/test_open_targets.py` contains two tests marked `# TODO rework` (`test_surfacing_pipeline`, `test_get_drug_target_competitors_semaglutide`) — they call the partially-implemented `get_drug_competitors()` method and may be fragile.
- `runners/pubmed_runner.py` uses `print()` instead of the `logging` module, which violates project rules.
- `tests/integration/services/test_pubmed_query.py` contains two tests marked `# TODO delete` (`test_get_single_pubmed_query_returns_drug_and_term`, `test_get_single_disease_synonym`) — these are superseded by the parametrized suite but have not been removed yet.
- `db/session.py` creates a new engine and session factory on every call to `get_db()` — there is no connection pooling singleton.
