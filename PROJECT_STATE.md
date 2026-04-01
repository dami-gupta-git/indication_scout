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
| `services/disease_helper.py` | Complete | LLM normalization; blocklist guard; PubMed count verification; file-based cache for both LLM results and PubMed counts |
| `services/pubmed_query.py` | Complete | Builds PubMed queries by normalizing disease name and combining with drug name |
| `services/retrieval.py` | Complete | `build_drug_profile`, `get_disease_synonyms`, `extract_organ_term`, `expand_search_terms`, `get_stored_pmids`, `fetch_new_abstracts`, `embed_abstracts`, `insert_abstracts`, `fetch_and_cache`, `semantic_search`, `synthesize`, `_normalize_disease_groups`, `get_drug_competitors` all implemented; `_normalize_disease_groups` normalizes disease names via `llm_normalize_disease` before the merge step; `get_drug_competitors` orchestrates raw fetch, normalization, LLM merge, sort, and top-10 slicing |
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
Services layer: llm.py, disease_helper.py, pubmed_query.py, retrieval.py
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
| `src/indication_scout/services/disease_helper.py` | LLM-driven disease term normalization; two-step strategy (normalize then verify/broaden); uses `cache_get`/`cache_set` from `utils/cache.py` |
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

---

## Update (2026-03-11)

### Implementation Status Changes
- `data_sources/pubmed.py` — Complete. Fixed PubMed date filter bug: split merged `datetype_maxdate` param into two correct params `datetype=pdat` and `maxdate=YYYY/MM/DD` in both `search()` and `get_count()` methods.
- `data_sources/open_targets.py` — Complete. Architectural fix: moved merge/remove/sort/cache logic out of `get_drug_competitors` and into `RetrievalService`; client now returns raw `CompetitorRawData` TypedDict without LLM calls.
- `services/retrieval.py` — Complete. Added `get_drug_competitors()` method that owns full pipeline (merge via `merge_duplicate_diseases` LLM call, remove/sort, cache keyed by drug_name).
- `tests/unit/data_sources/test_open_targets_client.py` — Complete. Removed patches for `merge_duplicate_diseases`, updated assertions to check `result["diseases"]`.
- `tests/unit/services/test_retrieval.py` — Complete. Added two new unit tests: alias-in-removed edge case and cache hit path for `get_drug_competitors()`.
- `docs/findings.md` — Complete. Appended two findings: PubMed date filter bug and `get_drug_competitors` layering fix.

### New Patterns / Decisions
- Client layer must never call the LLM; LLM calls belong in services layer.
- `drug_indications` must be passed alongside raw diseases because `merge_duplicate_diseases` needs it as context for the LLM prompt.
- Cache for `drug_competitors` lives in the service (keyed by `drug_name`), not the client — service owns the full merge+cache pipeline.

### Known Issues / Caveats Added
- `disease_helper.py` still uses `httpx` in `pubmed_count()` while the rest of the codebase uses `aiohttp` via `BaseClient` — no retry/backoff, inconsistency noted but not addressed.
- `sorted_data_2` on `open_targets.py` line 217 is assigned but never used — pre-existing dead code left in place.


## Update (2026-03-14)

### Implementation Status Changes
- `pyproject.toml` — Complete. Added `wandb` as a runtime dependency.
- `config.py` — Complete. Added `wandb_api_key` field (fixed pre-existing typo `wand_api_key`).
- `services/retrieval.py` `semantic_search()` — Complete. Added W&B logging: logs a `wandb.Table` with pmid, title, similarity per disease; scalar metrics namespaced as `semantic_search/{disease_key}/...` with spaces replaced by underscores.
- `runners/rag_runner.py` — Complete. Applied `@wandb_run(project='indication-scout')` decorator; removed inline `wandb.init` / `wandb.finish` calls.
- `utils/wandb_utils.py` — Complete. New utility module with `@wandb_run(project, tracked_param)` decorator for wrapping async functions in W&B run lifecycle (init/finish); extracts tracked param value via `inspect.signature`.
- `tests/unit/services/test_retrieval.py` — Complete. Added two unit tests: `test_semantic_search_logs_wandb_table_when_run_active` and `test_semantic_search_skips_wandb_log_when_no_run`.

### New Patterns / Decisions
- W&B init/finish belongs in the pipeline entry point (`rag_runner`), not inside the service layer.
- Decorator approach (`@wandb_run`) chosen over inline calls for reusability across future pipeline runners.
- Metric keys namespaced as `semantic_search/{disease_key}/...`; spaces in disease names replaced with underscores to avoid W&B UI rendering issues.
- `tracked_param` chosen as the generic decorator argument name (not `drug_param`) for future flexibility across different runners and tracked parameters.

### Known Issues / Caveats Added
- `OpenTargetsClient.get_drug_competitors` cache-hit path (line 133) returns old flat shape `{disease: set(drugs)}` instead of `CompetitorRawData` — caused `KeyError` on `raw["diseases"]` at runtime; resolved by deleting stale cache.
- `config.py` typo `wand_api_key` instead of `wandb_api_key` caused pydantic `ValidationError` blocking all unit tests at session start.
- W&B table keys with spaces in disease names caused "No rows to display" in UI — fixed by replacing spaces with underscores in metric keys.


---

## Update (2026-03-29)

### Implementation Status Changes
- `agents/clinical_trials.py` — Partial → Complete. Implemented `ClinicalTrialsAgent` using LangChain ReAct pattern with system prompt and `_parse_result()` method to reconstruct Pydantic models from LLM message history.
- `agents/clinical_trials_tools.py` — Partial → Complete. Created `build_clinical_trials_tools(date_before)` factory returning 4 `@tool` wrappers (`detect_whitespace`, `search_trials`, `get_landscape`, `get_terminated`) with date_before captured via closure.
- `agents/clinical_trials_model.py` — Stub → Complete. New file defining `ClinicalTrialsOutput` Pydantic model referencing models from `models/model_clinical_trials.py`.
- `tests/unit/agents/test_clinical_trials_tools.py` — Partial → Complete. Adapted to `build_clinical_trials_tools()` factory pattern; added `date_before` passthrough test.
- `tests/unit/agents/test_clinical_trials_agent.py` — Stub → Complete. New file with 5 unit tests covering `_parse_result` paths (whitespace, active trials, minimal, empty, block content).
- `tests/integration/agents/test_clinical_trials_agent.py` — Stub → Complete. New file with 3 end-to-end integration tests (whitespace, active trials, no data).
- `tests/integration/agents/test_clinical_trials_tools.py` — Partial → Complete. Adapted to factory pattern.
- `docs/agents.md` — Stub → Complete. Full architecture doc covering file layout, classes, tools, models, data flow, how to call, and how to add new agents.
- `config.py` — Partial → Complete. Added missing type annotation to `big_llm_model` field (was `PydanticUserError`).
- `pyproject.toml` — Complete. Added `langchain-anthropic>=0.3.0` to runtime dependencies.

### New Patterns / Decisions
- 3-file-per-agent pattern: `<name>.py` (agent class), `<name>_tools.py` (tool factory), `<name>_model.py` (output Pydantic model).
- Tools use closure pattern for `date_before` (and other parameters) rather than exposing as tool parameters — keeps LLM context clean.
- `get_landscape` passes `top_n=10` to keep LLM context manageable.
- LangChain `create_agent` uses parameter `system_prompt` (not `prompt`) — silent failure if wrong.
- Agent LLM set to Opus (`settings.big_llm_model`) instead of Haiku for this complex agent; `max_tokens` bumped from 1024 to 4096 for ReAct loop headroom.
- Tool responses currently use full `model_dump()` — identified as open issue requiring context-size optimization.

### Known Issues / Caveats Added
- Tool response payloads can be very large (e.g. 200 trials in `get_landscape`), wasteful for LLM context — needs slimming to condensed dicts for LLM while preserving full data for downstream.
- All 283 unit tests pass; integration tests may vary with real API data.

## Update (2026-03-29)

### Implementation Status Changes
- `models/model_clinical_trials.py` — Complete. Trimmed `Trial` model: removed `collaborators`, `study_type`, `results_posted` fields not consumed by agents. Trimmed `TerminatedTrial` model from 12 fields to 6: kept `nct_id`, `drug_name`, `indication`, `phase`, `why_stopped`, `stop_category`; removed `title`, `enrollment`, `sponsor`, `start_date`, `termination_date`, `references`.
- `data_sources/clinical_trials.py` — Complete. Updated `_parse_trial()` to remove parsing for deleted `Trial` fields; updated `_parse_terminated_trial()` to match slimmed `TerminatedTrial` model.
- `agents/clinical_trials.py` — Complete. Fixed `_parse_result()` bug: `AIMessage.name = None` (not absent), changed `hasattr` check to `getattr(msg, 'name', None)`; fixed tool response parsing to handle JSON strings via `json.loads()` fallback; added model_dump serialization boundary documentation.
- `tests/unit/models/test_clinical_trials_models.py` — Complete. Removed assertions for deleted fields from fixtures and test functions.
- `tests/integration/data_sources/test_clinical_trials.py` — Complete. Removed assertions for deleted fields from 4 test functions.
- `tests/unit/agents/test_clinical_trials_tools.py` — Complete. Removed deleted fields from `Trial` construction and assertions.
- `tests/unit/agents/test_clinical_trials_agent.py` — Complete. Removed deleted fields from test data dicts and `TerminatedTrial` assertions.
- `tests/integration/agents/test_clinical_trials_tools.py` — Complete. Removed deleted fields from assertions.
- `docs/agents.md` — Complete. Updated with model_dump serialization boundary explanation; added multi-tool collation details explaining slot-per-tool overwrite behavior; corrected `get_landscape` parameter name from `condition` to `indication`.

### New Patterns / Decisions
- Model trimming pattern: fetch all data from APIs (stay broad), but Pydantic models retain only fields consumed by agents/services — reduces LLM token cost via `model_dump()` serialization. Same principle applied to `DrugProfile` (projection of `RichDrugData`).
- `TerminatedTrial` only needs 6 fields for LLM red-flag assessment — full details not needed.
- `model_dump()` is a serialization boundary: Pydantic → dict for LLM → reconstruct Pydantic in `_parse_result()`. Agents must guard against JSON strings in tool responses (LangChain may stringify).
- Multi-tool collation: agent result dict slots are per-tool (no merging); second call to same tool overwrites first result.
- `Intervention` model retained on `Trial` for now (still needed by `_primary_drug()` helper) — open question whether to flatten into `drug_name`/`drug_type` fields.
- Cycled agent LLM through Opus → Sonnet → Haiku → Sonnet → Opus for user comparison testing; final model choice pending.

### Known Issues / Caveats Added
- `AIMessage.name = None` vs `hasattr` was a subtle bug — unit tests passed (SimpleNamespace has no `.name`) but integration tests failed (real AIMessage has `.name = None`).
- `get_terminated` returns 100 results with a single broad `query` string — if LLM passes just the drug name, gets all terminated trials across all indications instead of target disease only. Needs more context in tool input.
- 59 unit tests pass after model trimming; integration tests depend on final agent LLM model choice.


## Update (2026-03-29)

### Implementation Status Changes
- `agents/clinical_trials.py` — Stub → Complete. Implemented using LangChain ReAct pattern; `_parse_result()` reconstructs Pydantic models from message history; fixed `AIMessage.name = None` vs `hasattr` bug and JSON string parsing fallback for tool responses.
- `agents/clinical_trials_tools.py` — Partial → Complete. Factory `build_clinical_trials_tools(date_before)` returns 4 LangChain `@tool` wrappers with date_before captured via closure; adapted to support Opus/Sonnet/Haiku testing cycles.
- `agents/clinical_trials_model.py` — Stub → Complete. `ClinicalTrialsOutput` model referencing `WhitespaceResult`, `ConditionLandscape`, `TerminatedTrial` from data source models.
- `models/model_clinical_trials.py` — Complete. Trimmed `Trial` model: removed `collaborators`, `study_type`, `results_posted` (not used by agents/services). Trimmed `TerminatedTrial` from 12 → 6 fields: kept `nct_id`, `drug_name`, `indication`, `phase`, `why_stopped`, `stop_category`; removed `title`, `enrollment`, `sponsor`, `start_date`, `termination_date`, `references`.
- `data_sources/clinical_trials.py` — Complete. Updated `_parse_trial()` and `_parse_terminated_trial()` to match model changes; removed field assignments for deleted columns.
- `tests/unit/agents/test_clinical_trials_tools.py` — Complete. Updated for factory pattern; parametrized assertions; added `date_before` passthrough coverage.
- `tests/unit/agents/test_clinical_trials_agent.py` — Stub → Complete. 5 tests covering `_parse_result` paths: whitespace detection, active trials merging, minimal valid output, empty result, block content presence.
- `tests/integration/agents/test_clinical_trials_agent.py` — Stub → Complete. 3 end-to-end integration tests: whitespace scenario, active trials scenario, no data returns empty.
- `tests/integration/agents/test_clinical_trials_tools.py` — Complete. Updated for factory pattern; parametrized test coverage.
- `tests/unit/models/test_clinical_trials_models.py` — Complete. Removed assertions for deleted `Trial` fields; updated `TerminatedTrial` fixtures.
- `tests/integration/data_sources/test_clinical_trials.py` — Complete. Removed deleted field assertions from 4 test functions.
- `docs/agents.md` — Stub → Complete. Full architecture guide: 3-file-per-agent pattern, LangChain ReAct flow, `model_dump()` serialization boundary, multi-tool collation (slot-per-tool, no merging), tool result parsing, how to add new agents.
- `config.py` — Complete. Fixed missing type annotation on `big_llm_model` field (was `PydanticUserError`).
- `pyproject.toml` — Complete. Added `langchain-anthropic>=0.3.0` to runtime deps; `max_tokens` bumped 1024 → 4096 for agent headroom.

### New Patterns / Decisions
- 3-file-per-agent pattern is now standard: `<name>.py` (agent), `<name>_tools.py` (tools factory), `<name>_model.py` (output model).
- Model trimming pattern applied: data sources fetch broad API data; Pydantic models retain only fields consumed by agents/services to reduce LLM token cost on `model_dump()`.
- Tools use closure pattern for `date_before` and other parameters; parameters not exposed as tool inputs to keep LLM context clean.
- `_parse_result()` reconstructs Pydantic from message history: `AIMessage.name = None` (not absent), tool responses may be JSON strings requiring `json.loads()` fallback.
- `get_landscape` slices to `top_n=10` trials; agent LLM set to Opus (not Haiku) for complex reasoning.
- Multi-tool collation pattern: slot-per-tool in result dict; second call to same tool overwrites first (no merging logic).

### Known Issues / Caveats Added
- `get_terminated` API returns 100 results per broad `query`; LLM passing drug name alone retrieves all terminated trials across all indications instead of target disease — no indication-specific filtering applied.
- Tool `model_dump()` payloads are large (200+ trials); LLM context cost identified but not yet optimized (condensed dicts for LLM vs full data for downstream still open).
- LLM model choice (Opus vs Sonnet vs Haiku for agent) was cycling during session for quality comparison; final choice pending user evaluation.


---

## Update (2026-03-29)

### Implementation Status Changes
- `models/model_clinical_trials.py` — Complete. Removed unused fields from `Trial` (`collaborators`, `study_type`, `results_posted`) and `TerminatedTrial` (`title`, `enrollment`, `sponsor`, `start_date`, `termination_date`, `references`). `TerminatedTrial` now contains only 6 essential fields: `nct_id`, `drug_name`, `indication`, `phase`, `why_stopped`, `stop_category`.
- `data_sources/clinical_trials.py` — Complete. Updated `_parse_trial()` and `_parse_terminated_trial()` parsing logic to remove assignments for deleted fields.
- `agents/clinical_trials.py` — Complete. Fixed subtle `AIMessage.name` bug (property is `None`, not absent); changed `hasattr` to `getattr(msg, 'name', None)`. Added JSON string fallback parsing for tool responses (`json.loads()` before type checks). Updated `docs/agents.md` with `model_dump()` serialization boundary and multi-tool collation explanation (slot-per-tool, no merging).
- All test files updated to remove assertions/fixtures for deleted fields across unit and integration test suites (7 test files modified, 59 tests pass).

### New Patterns / Decisions
- Model trimming applied to `TerminatedTrial`: LLM red-flag assessment requires only 6 fields; full dataset fetched from API but Pydantic model projects down to what agents actually consume.
- `AIMessage.name = None` requires defensive `getattr()` — unit tests using `SimpleNamespace` passed but integration tests with real `AIMessage` failed with `hasattr` approach.
- Tool response parsing must handle JSON string stringification from LangChain: added `json.loads()` fallback before `isinstance` checks.

### Known Issues / Caveats Added
- `Intervention` model still on `Trial` (needed by `_primary_drug()` helper) — open question whether to flatten into scalar `drug_name`/`drug_type` fields for consistency with trimmed model philosophy.
- `get_terminated` broad query + LLM context issue remains: tool may retrieve all terminated trials across all indications if LLM passes only drug name without indication context.

---

## Update (2026-03-29)

### Implementation Status Changes
- `runners/rag_runner.py` — Complete. Parallelized disease-loop processing using `asyncio.gather` with three semaphores (`RAG_DISEASE_CONCURRENCY=4`, `RAG_LLM_CONCURRENCY=4`, `RAG_PUBMED_CONCURRENCY=3`). Extracted `_process_disease()` helper for per-disease pipeline encapsulation.
- `constants.py` — Complete. Added three new concurrency limit constants for RAG pipeline parallelization.
- `services/embeddings.py` — Complete. Implemented async `embed_async()` wrapper with `asyncio.Lock` to serialize access to shared `SentenceTransformer` singleton; made `RetrievalService.embed_abstracts()` async.
- `services/retrieval.py` — Complete. Updated `semantic_search()` to use new `embed_async()` instead of sync `embed`.
- All unit tests for `retrieval.py` — Complete. Updated mock targets from `retrieval.embed` to `retrieval.embed_async`; made test functions async.
- Integration test for embeddings — Complete. Updated to await now-async `embed_abstracts()` call.

### New Patterns / Decisions
- Parallelization strategy: cross-disease parallelism is preferred axis (4 diseases in parallel) since per-disease pipeline steps are inherently sequential.
- Three-semaphore design enforces rate limits: disease-level limits overall concurrency, LLM semaphore prevents Anthropic rate limits, PubMed semaphore respects NCBI.
- Embedding lock (not semaphore): `SentenceTransformer` model is a non-thread-safe global singleton — `asyncio.Lock` ensures only 1 concurrent `model.encode()`.
- Concurrency constants live in `constants.py` for easy tuning; tuned empirically from 2/2/2 up to 4/4/3.

### Known Issues / Caveats Added
- `SentenceTransformer` model lacks native async support — required wrapper with `asyncio.Lock` for safe concurrent access.
- After making `embed_abstracts()` async, all callers and mocks must be updated; transitional cache invalidation needed if deploying to running systems.
