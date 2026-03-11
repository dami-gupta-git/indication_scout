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
| `models/model_evidence_summary.py` | Complete | `EvidenceSummary` with 7 fields: `summary`, `study_count`, `study_types`, `strength` (Literal), `has_adverse_effects`, `key_findings`, `supporting_pmids`; `coerce_nones` validator applied |
| `agents/orchestrator.py` | Stub | `run()` raises `NotImplementedError` |
| `agents/literature.py` | Stub | `run()` raises `NotImplementedError` |
| `agents/clinical_trials.py` | Stub | `run()` raises `NotImplementedError` |
| `agents/mechanism.py` | Stub | `run()` raises `NotImplementedError` |
| `agents/safety.py` | Stub | `run()` raises `NotImplementedError` |
| `services/llm.py` | Complete | `query_llm`, `query_small_llm`, and `parse_llm_response` via Anthropic SDK |
| `services/disease_normalizer.py` | Complete | LLM normalization; blocklist guard; PubMed count verification; file-based cache for both LLM results and PubMed counts |
| `services/pubmed_query.py` | Complete | Builds PubMed queries by normalizing disease name and combining with drug name |
| `services/retrieval.py` | Complete | `build_drug_profile`, `get_disease_synonyms`, `extract_organ_term`, `expand_search_terms`, `get_stored_pmids`, `fetch_new_abstracts`, `embed_abstracts`, `insert_abstracts`, `fetch_and_cache`, `semantic_search`, `synthesize` all implemented |
| `sqlalchemy/pubmed_abstracts.py` | Complete | SQLAlchemy ORM model with pgvector embedding column (768 dims) |
| `db/session.py` | Complete | SQLAlchemy session factory; `get_db()` dependency |
| `api/main.py` | Partial | FastAPI app with `/health` endpoint only; `api/routes/` and `api/schemas/` subdirs contain only `__init__.py` |
| `utils/cache.py` | Complete | Shared file-based cache utility: `cache_key`, `cache_get`, `cache_set`; used by all data source clients and services |
| `helpers/drug_helpers.py` | Complete | `normalize_drug_name` strips common salt suffixes |
| `prompts/extract_organ_term.txt` | Complete | Haiku prompt; Input/Output few-shot format; returns single organ term |
| `prompts/expand_search_terms.txt` | Complete | Full prompt with 9 template variables; 5-axis structure with per-axis caps; JSON array output |
| `prompts/disease_synonyms.txt` | Complete | Input/Output few-shot format (updated to match other prompts) |
| `prompts/synthesize.txt` | Complete | Evidence synthesis prompt; takes `drug_name`, `disease_name`, `abstracts`; outputs JSON matching `EvidenceSummary` fields |
| `scripts/session.py` | Complete | Session file manager: `startup` (create/rotate/print) and `append` subcommands; rotation to `session_bak/` with 5-file cap |
| `scripts/open_target_pipeline.py` | Complete | Exploratory pipeline: fetches bupropion competitors, builds PubMed queries, fetches abstracts; proper async with client context managers, `asyncio.run` under `__main__`, uses logging |
| `runners/rag_runner.py` | Complete | End-to-end RAG pipeline runner; `run_rag(drug_name, db)` iterates over top 10 disease indications from `get_drug_competitors`, runs full pipeline per disease, returns `dict[str, EvidenceSummary]` |
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
| `src/indication_scout/services/retrieval.py` | RAG pipeline; Stage 0 (`expand_search_terms`, `extract_organ_term`), Stage 1 (`fetch_and_cache`, `embed_abstracts`, `insert_abstracts`, `get_stored_pmids`, `fetch_new_abstracts`), Stage 2 (`semantic_search`), and Stage 3 (`synthesize`) all implemented |
| `src/indication_scout/models/model_drug_profile.py` | `DrugProfile` — flat LLM-facing projection of `RichDrugData`; built via `from_rich_drug_data(rich, atc_descriptions)` |
| `src/indication_scout/prompts/` | All LLM prompt templates; `extract_organ_term.txt`, `expand_search_terms.txt`, `disease_synonyms.txt`, `synthesize.txt` |
| `src/indication_scout/sqlalchemy/pubmed_abstracts.py` | ORM model for the `pubmed_abstracts` pgvector table |
| `scripts/open_target_pipeline.py` | Exploratory async pipeline script; not part of production path |
| `runners/pubmed_runner.py` | Development/exploration script; uses `print()` in violation of project rules |
| `src/indication_scout/runners/rag_runner.py` | End-to-end RAG pipeline runner; `run_rag(drug_name, db)` orchestrates `get_drug_competitors` then per-disease: `build_drug_profile` -> `expand_search_terms` -> `fetch_and_cache` -> `semantic_search` -> `synthesize` |
| `tests/unit/runners/test_rag_runner.py` | Unit tests for `run_rag`; 4 tests covering pipeline ordering, argument passing between stages |
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
- The RAG pipeline in `services/retrieval.py` is fully implemented: Stage 0 (`expand_search_terms` + `extract_organ_term`), Stage 1 (`fetch_and_cache`, `embed_abstracts`, `insert_abstracts`, `get_stored_pmids`, `fetch_new_abstracts`), Stage 2 (`semantic_search`), and Stage 3 (`synthesize`) are all complete.
- `DrugBankClient` is a stub (`get_drug` and `get_interactions` raise `NotImplementedError`).
- The CLI module referenced in `pyproject.toml` (`indication_scout.cli.cli`) does not exist.
- `tests/integration/data_sources/test_open_targets.py` contains two tests marked `# TODO rework` (`test_surfacing_pipeline`, `test_get_drug_target_competitors_semaglutide`) — they call the partially-implemented `get_drug_competitors()` method and may be fragile.
- `runners/pubmed_runner.py` uses `print()` instead of the `logging` module, which violates project rules.
- `tests/integration/services/test_pubmed_query.py` contains two tests marked `# TODO delete` (`test_get_single_pubmed_query_returns_drug_and_term`, `test_get_single_disease_synonym`) — these are superseded by the parametrized suite but have not been removed yet.
- `db/session.py` creates a new engine and session factory on every call to `get_db()` — there is no connection pooling singleton.

---

## Update (2026-03-07)

### Implementation Status Changes
- `docs/claude_code_tooling.md` — Complete. New file documenting the full Claude Code setup for this project.
- `.claude/skills/session.md` — Complete. New skill file defining session continuation format, rules, and when-to-write guidance.
- `.claude/commands/remember.md` — Complete. Updated to write session entry per `skills/session.md` and invoke the `project-state-updater` agent.
- `project-state-updater` agent — Complete. Updated model to haiku; description clarified as invoked by `/remember`.
- `CLAUDE.md` — Partial. Session File Workflow section updated to reference `skills/session.md` for format and timing rules.
- `.claude/rules/testing.md` (formerly `skills/testing.md`) — Partial. Removed `globs: tests/**` frontmatter; file is now a plain reference file with no slash-command role.

### New Patterns / Decisions
- `/remember` is the single end-of-session command: writes a session entry then auto-triggers the `project-state-updater` agent.
- Session writing rules live exclusively in `skills/session.md`; `CLAUDE.md` references it rather than duplicating content.
- `project-state-updater` agent uses haiku — the append-only curator task requires no deep reasoning.
- Agents are auto-invoked via their `description` field and can also be called explicitly from slash commands.
- `skills/testing.md` is a plain reference file, not a slash-command skill — no frontmatter needed.

### Known Issues / Caveats Added
- None discovered this session.

## Update (2026-03-07 continued)

### Implementation Status Changes
- `open_targets.py` `get_drug_competitors` — Complete. Fixed ADHD filtering bug by tracking max phase per drug per disease; fixed BROADENING_BLOCKLIST filter logic from OR to AND (all words must be blocklisted, not any).
- `services/embeddings.py` — Complete. Added `local_files_only=True` to SentenceTransformer instantiation to skip HuggingFace HTTP checks on load.
- `runners/rag_runner.py` — Complete. Added per-step timing logs using `time.perf_counter()` inside disease loop.
- `constants.py` — Complete. Added `PUBMED_MAX_RESULTS: int = 200` constant.
- `services/retrieval.py` — Complete. Replaced hardcoded PubMed max_results with `PUBMED_MAX_RESULTS` constant.
- `tests/unit/services/test_retrieval.py` — Complete. Updated `test_fetch_and_cache_calls_search_per_query` to use `PUBMED_MAX_RESULTS` constant.

### New Patterns / Decisions
- Only delete a disease from `siblings` if the input drug appears there at phase >= 4 (not just any phase) — stricter criterion to preserve valid competitors.
- BROADENING_BLOCKLIST filter now requires ALL words to match the blocklist (`words <= BROADENING_BLOCKLIST`) instead of ANY word — prevents false filtering of multi-word disease names containing generic terms like 'disorder'.
- `PUBMED_MAX_RESULTS = 200` (down from 500) — reduces embedding time on cold cache as acceptable tradeoff; BioLORD-2023 on MPS is CPU-bound and benefits from smaller batch size.
- Parallelization plan approved (Option A: shared session + asyncio.Lock, concurrency limit = 3); plan written to PLAN.md but not yet implemented.

### Known Issues / Caveats Added
- `fetch_and_cache` takes 11-50s per disease on cold cache due to BioLORD-2023 embedding volume (CPU-bound on MPS).
- `synthesize` takes ~10s per disease (Sonnet API call) — 15 diseases serially = ~5 min; parallelization needed for acceptable runtime.
- Debug logging was at DEBUG level and not visible; temporarily changed to ERROR level to verify bug fixes during session.

---

## Update (2026-03-07 session continuation)

### Implementation Status Changes
- `tests/integration/runners/test_rag_runner.py` — Complete. New integration test file with two tests: `test_run_rag_empagliflozin` and `test_run_rag_semaglutide`. Both verify all 15 disease indications and full `EvidenceSummary` field values from live pipeline runs using data-driven assertions.
- `tests/integration/runners/__init__.py` — Complete. New directory marker for runner integration tests.

### New Patterns / Decisions
- Runner integration tests use `db_session_truncating` fixture (not `db_session`) because `fetch_and_cache` calls `db.commit()` internally and cannot be rolled back via savepoint.
- Set-based assertions (`study_types`, `supporting_pmids`) are order-independent; LLM outputs (`summary`, `key_findings`) checked for presence only due to inter-run variability.
- Test data sourced from live results files (`data/empag_results.md`, `data/semaglutide_results.md`) — assertions use file values as written.

### Known Issues / Caveats Added
- `eating disorder` (semaglutide): 5 key findings cite PMIDs but only 4 appear in `supporting_pmids` — data inconsistency accepted from source.
- `major depressive disorder` (semaglutide): `supporting_pmids` is empty despite 5 findings with cited PMIDs — data inconsistency accepted from source.
- LLM output variability may require loosening assertions after first real test run.

---

## Update (2026-03-08)

### Implementation Status Changes
- `services/retrieval.py` — Complete. Refactored from module-level functions into a `RetrievalService` class with `cache_dir: Path` parameter in `__init__`, exposing `get_drug_competitors()` method.
- `runners/rag_runner.py` — Complete. Updated to instantiate `RetrievalService(cache_dir)` and route all pipeline calls through it; replaced all `print()` calls with `logger.info()`.
- `tests/integration/conftest.py` — Complete. Added `test_cache_dir` fixture returning `TEST_CACHE_DIR` to decouple tests from direct constants imports.
- `tests/integration/runners/test_rag_runner.py` — Complete. Updated both tests to accept `test_cache_dir` fixture and pass it to `run_rag`; removed direct `TEST_CACHE_DIR` imports.

### New Patterns / Decisions
- Cache dir definition centralized in `constants.py` (`DEFAULT_CACHE_DIR`, `TEST_CACHE_DIR`); `RetrievalService` is the single point where cache dir is wired into the pipeline.
- `run_rag` defaults to `DEFAULT_CACHE_DIR` so production callers (CLI, API) require no changes.
- Test cache dir flows from conftest fixture, not individual test imports — eliminates test-to-constants coupling.

### Known Issues / Caveats Added
- Old `rag_runner.py` was calling removed module-level functions (`build_drug_profile`, `expand_search_terms`, etc.); refactor to service class eliminated the runtime `NameError`.
