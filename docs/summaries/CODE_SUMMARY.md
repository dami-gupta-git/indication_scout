# Code Summary — IndicationScout
Generated 2026-09-19 · commit c552361

## What this project does

IndicationScout takes a drug name and produces a drug-repurposing report. A supervisor LLM agent surfaces candidate diseases
from Open Targets competitor and mechanism seeds, then delegates per-disease investigation to three specialist sub-agents
(mechanism, literature, clinical trials) that query Open Targets, ClinicalTrials.gov, PubMed, Europe PMC, ChEMBL, and openFDA at
query time. The run emits a typed `SupervisorOutput` that is rendered to Markdown by the CLI, returned as JSON by the FastAPI
job API, and displayed by a React frontend. A `--date-before` holdout mode restricts PubMed and trial evidence to records dated
before a cutoff so a past state of the world can be replayed.

## Directory map

| Path | What it owns |
|------|--------------|
| `src/indication_scout/agents/` | The supervisor agent and the literature, clinical-trials, and mechanism sub-agents, plus the shared ReAct loop |
| `src/indication_scout/agents/supervisor/` | Supervisor tools, candidate dedup/merge, output model, run assembly |
| `src/indication_scout/agents/literature/` | Literature agent, its RAG tools, adverse-event PubMed search, output model |
| `src/indication_scout/agents/clinical_trials/` | Clinical-trials agent, its CT.gov tools, trial-relevance coverage, output model |
| `src/indication_scout/agents/mechanism/` | Mechanism agent, Open Targets tools, candidate scoring/selection, local OT score recomputation |
| `src/indication_scout/api/` | FastAPI app, async analyses job routes, drilldown routes, seed-example routes, request/response schemas |
| `src/indication_scout/cli/` | The Click-based `scout` command group |
| `src/indication_scout/data_sources/` | One async HTTP client per external API, all on a shared retrying base client |
| `src/indication_scout/db/` | SQLAlchemy declarative `Base`, engine and session-factory construction |
| `src/indication_scout/helpers/` | Drug-name normalization and the run-scoped `DrugIntake` seed |
| `src/indication_scout/ml_models/` | Trial-risk and trial-success modeling, plus offline repurposing/safety probes |
| `src/indication_scout/models/` | Pydantic contracts for every external data shape, one module per source |
| `src/indication_scout/prompts/` | LLM prompt templates as `.txt` files loaded by agents and services |
| `src/indication_scout/regression/` | Report-vs-report diff harness backing `scout diff-report` |
| `src/indication_scout/report/` | `SupervisorOutput` → Markdown rendering |
| `src/indication_scout/runners/` | Standalone async runners for the PubMed and RAG paths |
| `src/indication_scout/services/` | Business logic: RAG retrieval, LLM access, embeddings, disease/approval resolution, judging, persistence, jobs |
| `src/indication_scout/sqlalchemy/` | ORM tables: `pubmed_abstracts` (pgvector) and the analysis-run/attempt/event/cost tables |
| `src/indication_scout/utils/` | The shared SHA-keyed disk cache |
| `alembic/` | Database migrations for the run and abstract tables |
| `frontend/` | Vite + React UI: tabs, tables, charts, mechanism graph, polling hook |
| `scripts/` | Operational and validation scripts (seeding, precision/recall checks, probes, benchmarks) |
| `tests/` | `unit/`, `integration/`, `regression/` (contract cassettes, deterministic, structural, pipeline replay), and LLM harnesses |
| `observability/` | Prometheus scrape config and Grafana dashboards/provisioning |
| `docs/` | Architecture, feature, ops, and reference documentation |

## Architecture

### Entry layer

Both entry points converge on `services/analysis_runner.py`. `run_analysis` normalizes the drug name, calls
`data_sources.chembl.resolve_drug_name` as a fail-fast existence check, creates one session factory for the whole run, warms the
embedding model on a background task, builds the supervisor agent, runs it, and returns the `SupervisorOutput` together with the
formatted Markdown. `run_pair_analysis` handles the fixed drug+disease path used by `scout investigate`: it skips candidate
discovery and runs the mechanism, literature, and clinical-trials sub-agents directly on the given pair.

### Agent layer

`agents/_react_loop.py` builds a LangGraph state graph whose loop terminates the moment a `finalize_*` tool succeeds, checked by
a caller-supplied `finalize_done` predicate over the trailing `ToolMessage`s. The supervisor and clinical-trials agents use this
gated loop because their finalize tools can reject and must re-enter the model node; the literature and mechanism agents use
LangGraph's prebuilt `create_react_agent`. The loop also installs Anthropic ephemeral cache breakpoints on the system prompt
(`cached_system_message`) and on the growing history (`history_cache_pre_model_hook`).

Tools return `response_format="content_and_artifact"`: a short string the model reads plus a typed Pydantic artifact. After the
loop, `run_<name>_agent` walks the message history and reads artifacts off `ToolMessage.artifact`, so exact identifiers are never
re-typed by the model.

Supervisor tool set (`agents/supervisor/supervisor_tools.py`): `find_candidates`, `analyze_mechanism`, `analyze_literature`,
`analyze_clinical_trials`, `get_drug_briefing`, `critique_ranking`, `finalize_supervisor`. When `Settings.supervisor_fanout` is
true, `investigate_top_candidates` is inserted before finalize and the two per-candidate `analyze_*` tools are removed, forcing
the parallel path. The module also returns three closures — `get_merged_allowlist`, `get_auto_findings`, `get_approval_labels` —
that `run_supervisor_agent` uses to canonicalize disease names, merge fan-out artifacts that bypassed the loop, and stamp
approval relationships onto findings.

Sub-agent tool sets: literature — `build_drug_profile`, `expand_search_terms`, `fetch_and_cache`, `semantic_search`,
`safety_search`, `synthesize`, `finalize_analysis`; clinical trials — `check_fda_approval`, `search_trials`, `get_completed`,
`get_terminated`, `get_landscape`, `finalize_analysis`; mechanism — `get_drug`, `get_target_associations`, `finalize_analysis`.

### Service layer

`RetrievalService` (`services/retrieval.py`, ~2500 lines) is the RAG pipeline: build a drug profile, expand search terms, fetch
and embed PubMed abstracts into pgvector, run semantic search, judge each PMID for drug identity / disease treatment /
direction, and synthesize an `EvidenceSummary`. It also owns the two-tier safety path (`safety_search`, `summarize_safety`,
`classify_indication_harm`). Around it sit narrower services: disease normalization and MeSH resolution (`disease_helper.py`),
FDA approval labelling (`approval_check.py`), development-stage judging with a deterministic phase floor (`dev_stage.py`),
interpretive blurb judging (`judge_interpretive.py`), citation stripping (`citation_guard.py`), and LLM/embedding wrappers.

### Data-source layer

`BaseClient` is an abstract async context manager providing `_request`, `_rest_get`, `_graphql`, and `_rest_get_xml`, with
exponential-backoff retry, per-source API timing accounting, and terminal-failure logging to `cache/data_source_failures.log`.
Errors surface as `DataSourceError`. All clients read and write the shared disk cache in `utils/cache.py`
(`cache/<namespace>/<sha256>.json`, entry carries `cached_at` and `ttl`). Open Targets additionally persists per-target evidence
files and ChEMBL persists per-drug name files outside the namespace cache.

### Flow

```
scout find / POST /api/analyses
        │
        ▼
services/analysis_runner.run_analysis
        │  normalize name → resolve_drug_name → session factory → build_agent
        ▼
agents/supervisor: build_gated_react_loop
        │
        ├── find_candidates ──► competitor seeds (Open Targets) + mechanism seeds
        │                        └── candidate_dedup: collapse synonyms, merge, hierarchy dedup
        ├── analyze_mechanism ─► mechanism agent ─► OT associations ─► build_candidate_rows
        │                                                          └─► select_top_candidates
        ├── analyze_literature ► literature agent ─► RetrievalService ─► PubMed + pgvector + LLM
        ├── analyze_clinical_trials ► CT agent ─► ClinicalTrialsClient ─► derive_trial_signals
        ├── critique_ranking   (mandatory before finalize)
        └── finalize_supervisor
        │
        ▼
SupervisorOutput ──► report/format_report.format_report ──► Markdown
                └──► API JSON / run_repository persistence
```

## Main modules and functions

### Entry / orchestration

| Path | Signature | Purpose |
|------|-----------|---------|
| `services/analysis_runner.py` | `async run_analysis(drug_name: str, *, date_before: date \| None = None) -> tuple[SupervisorOutput, str]` | Full pipeline for one drug; returns output and Markdown |
| `services/analysis_runner.py` | `async run_pair_analysis(drug_name: str, disease_name: str, *, date_before: date \| None = None) -> tuple[SupervisorOutput, str]` | Runs the three sub-agents on a fixed drug+disease pair, skipping discovery |
| `services/analysis_runner.py` | `build_agent(db, session_factory=None, date_before=None, cache_dir=DEFAULT_CACHE_DIR) -> tuple[Any, Any, Any, Any]` | Constructs the LLM, retrieval service, and supervisor agent with its closures |

### Agents

| Path | Signature | Purpose |
|------|-----------|---------|
| `agents/_react_loop.py` | `build_gated_react_loop(llm, tools, prompt, finalize_done) -> Any` | Compiles a ReAct graph that exits as soon as `finalize_done` holds |
| `agents/_react_loop.py` | `cached_system_message(prompt: str) -> SystemMessage` | Wraps the system prompt with an Anthropic ephemeral cache breakpoint |
| `agents/_react_loop.py` | `history_cache_pre_model_hook(state: MessagesState) -> dict` | Adds a cache breakpoint to the growing message history before each model call |
| `agents/supervisor/supervisor_agent.py` | `build_supervisor_agent(llm, svc, db, session_factory=None, date_before=None)` | Builds the supervisor loop and returns it with allowlist/findings/approval closures |
| `agents/supervisor/supervisor_agent.py` | `async run_supervisor_agent(agent, get_merged_allowlist, drug_name, get_auto_findings=None, get_approval_labels=None, date_before=None) -> SupervisorOutput` | Invokes the loop, canonicalizes disease names, assembles the typed output |
| `agents/supervisor/supervisor_tools.py` | `build_supervisor_tools(...) -> tuple[list, Callable, Callable, Callable]` | Defines all supervisor tools over shared run state and returns the three snapshot closures |
| `agents/supervisor/candidate_dedup.py` | `async run_hierarchical_dedup(drug_name, mechanism_targets, candidates) -> HierarchyDedupOutput` | LLM pass collapsing parent/child disease candidates into one entry |
| `agents/supervisor/candidate_dedup.py` | `collapse_synonym_entries(allowed_diseases, allowed_efo_ids) -> list[tuple[str, str]]` | Deterministically merges candidates that share an EFO id |
| `agents/supervisor/candidate_dedup.py` | `async merge_mechanism_entries(drug_name, allowed_diseases, allowed_efo_ids, approved_indications) -> list[tuple[str, str]]` | Merges mechanism-sourced entries into the competitor allowlist |
| `agents/literature/literature_agent.py` | `build_literature_agent(llm, svc, db, date_before=None, approved_indications=None, drug_profile=None) -> CompiledStateGraph` | Builds the literature ReAct agent with its tools bound to a retrieval service |
| `agents/literature/literature_agent.py` | `async run_literature_agent(agent, drug_name, disease_name) -> LiteratureOutput` | Runs the agent and assembles `LiteratureOutput` from tool artifacts |
| `agents/literature/pubmed_ae.py` | `async search_adverse_events(drug_name, cache_dir=..., date_before=None, top_cited=AE_TOP_CITED, disease=None, disease_aliases=None) -> list[PubmedAbstract]` | Drug-level and disease-scoped adverse-event PubMed search, ranked by Europe PMC citation count |
| `agents/clinical_trials/clinical_trials_agent.py` | `build_clinical_trials_agent(llm, date_before=None, assigned_indication=None, target_drug=None, cache_dir=...) -> CompiledStateGraph` | Builds the gated CT agent scoped to one indication |
| `agents/clinical_trials/clinical_trials_agent.py` | `async run_clinical_trials_agent(agent, drug_name, disease_name, first_approval=None, approved_indications=None, cache_dir=...) -> ClinicalTrialsOutput` | Runs the CT agent and assembles its typed output plus relevance coverage |
| `agents/clinical_trials/clinical_trials_tools.py` | `build_clinical_trials_tools(...) -> list` | Defines `search_trials`, `get_completed`, `get_terminated`, `get_landscape`, `check_fda_approval`, `finalize_analysis` |
| `agents/_trial_signals.py` | `derive_trial_signals(ct, relevant_nct_ids=None, contaminated_nct_ids=None) -> TrialSignals` | Deterministic facts from the trial record (highest completed phase, terminations, active pivotal Phase 3) |
| `agents/_trial_signals.py` | `is_non_therapeutic_study(trial: Trial) -> bool` | Flags diagnostic/imaging/healthy-volunteer studies for exclusion |
| `agents/_trial_formatting.py` | `_format_trial_table(...)`, `_phase_distribution(trials)`, `_borda_rank_by_enrollment_and_recency(trials, k)` | Renders trial tables and selects which trials the model sees |
| `agents/mechanism/mechanism_agent.py` | `async run_mechanism_agent(agent, drug_name, *, approved_indications, date_before=None) -> MechanismOutput` | Runs the mechanism agent and assembles targets, MoAs, and scored candidates |
| `agents/mechanism/mechanism_row_builder.py` | `async build_candidate_rows(ot_client, target_id, action_types, top_n, date_before=None) -> list[dict]` | Pulls per-target OT associations and evidence into scoreable rows |
| `agents/mechanism/mechanism_candidates.py` | `select_top_candidates(rows, approved_diseases, limit) -> list[MechanismCandidate]` | Keeps rows where drug direction opposes disease-driving direction, drops approved diseases, sorts by ranking score |
| `agents/mechanism/mechanism_candidates.py` | `aggregate_directions(records, min_fraction=_MAJORITY_THRESHOLD)`, `classify_positive(action_types, directions_on_target, directions_on_trait)` | Majority-vote direction aggregation and the positive/negative classification rule |
| `agents/mechanism/ot_score.py` | `recompute_overall(datasource_scores: dict[str, float], exclude: set[str]) -> float` | Reproduces the Open Targets overall score locally so leaky datasources can be dropped in holdout mode |

### Services

| Path | Signature | Purpose |
|------|-----------|---------|
| `services/retrieval.py` | `RetrievalService.build_drug_profile(chembl_id) -> DrugProfile` | Assembles the run's drug profile from Open Targets rich drug data |
| `services/retrieval.py` | `RetrievalService.get_drug_competitors(chembl_id, date_before=None) -> dict[str, set[str]]` | Competitor-seeded candidate diseases keyed by source drug |
| `services/retrieval.py` | `RetrievalService.fetch_and_cache(queries, db, date_before=None, direct_query=None) -> list[str]` | Runs PubMed queries, fetches and embeds new abstracts, returns PMIDs |
| `services/retrieval.py` | `RetrievalService.semantic_search(disease, chembl_id, pmids, db, date_before=None) -> list[AbstractResult]` | pgvector similarity search plus LLM relevance gating over the fetched PMIDs |
| `services/retrieval.py` | `RetrievalService.synthesize(chembl_id, disease, top_abstracts, approved_indications=None) -> EvidenceSummary` | Produces the graded evidence summary, forcing strength/direction down when the basis is not drug-specific |
| `services/retrieval.py` | `RetrievalService.safety_search(chembl_id, date_before=None, disease=None) -> SafetySearchResult` | Drug-level and disease-scoped adverse-event retrieval, kept separate |
| `services/retrieval.py` | `RetrievalService.summarize_safety(chembl_id, disease, drug_profile, safety_abstracts, date_before=None) -> DrugSafetyAssessment` | Source-separated drug-wide safety assessment (label, OT warnings, FAERS) |
| `services/retrieval.py` | `RetrievalService.classify_indication_harm(chembl_id, disease, safety_abstracts) -> tuple[bool \| None, str, list[str]]` | Per-indication harm adjudication requiring a quote present in the supplied text |
| `services/retrieval.py` | `RetrievalService.expand_search_terms(chembl_id, disease_name, drug_profile)`, `extract_organ_term(disease_name)` | LLM query expansion and organ-term extraction for PubMed queries |
| `services/approval_check.py` | `async get_approved_indications(drug_name, candidate_diseases, as_of) -> set[str]` | Approved indications for the drug as of a date |
| `services/approval_check.py` | `async get_fda_approved_disease_mapping(drug_name, candidate_diseases, approved_indications, cache_dir=...) -> dict[str, ApprovalLabel]` | Four-way approval relationship per candidate (approved / combination_only / contaminated / none) |
| `services/approval_check.py` | `async extract_approved_from_labels(label_texts, candidate_diseases, cache_dir=...) -> set[str]`, `async list_approved_indications_from_labels(label_texts, cache_dir=...) -> list[str]` | LLM extraction of approved indications from openFDA label text |
| `services/disease_helper.py` | `async llm_normalize_disease(raw_term)`, `async llm_normalize_disease_batch(raw_terms)` | Canonicalize disease strings, singly and in batch |
| `services/disease_helper.py` | `async merge_duplicate_diseases(diseases, drug_indications) -> MergeResult` | One LLM pass merging synonymous disease names across the candidate list |
| `services/disease_helper.py` | `async resolve_mesh_id(indication) -> tuple[str, str] \| None` | Resolves an indication to a MeSH descriptor used for CT.gov queries |
| `services/disease_helper.py` | `async normalize_for_pubmed(raw_term, drug_name=None)`, `async normalize_batch(terms, drug_name=None)`, `async pubmed_count(query)` | PubMed-facing term normalization and hit counting |
| `services/dev_stage.py` | `async judge_dev_stage(relevant_trials, cache_dir, *, drug="", indication="") -> StageJudgment` | LLM-judged development-stage tier, raised by `_enforce_tier_floor` when the trial record proves a higher tier |
| `services/dev_stage.py` | `dev_stage_phrase(sig: TrialSignals \| None) -> str \| None` | Renders the stage phrase shown in the report |
| `services/judge_interpretive.py` | `async judge_interpretive(*, stage, active_programs, literature, relationship, approved_indication, trial_evidence, closure, terminations, cache_dir, drug="", indication="") -> InterpretiveJudgment \| None` | Isolated LLM pass interpreting resolved facts so blurbs cannot contradict the stage |
| `services/clinical_trials_summary.py` | `async judge_ct_summary(relevant_trials, *, stage, active_programs, coverage=None, first_approval, cache_dir, drug="", indication="") -> CTSummary \| None` | LLM summary of the trial picture for one candidate |
| `services/trial_target.py` | `async judge_trials_treat_disease(drug, disease, trials, cache_dir=...) -> dict[str, bool]` | Per-trial relevance judgement keyed by NCT id |
| `services/citation_guard.py` | `unknown_pmids(text, allowed)`, `unknown_nct_ids(text, allowed)`, `strip_findings_with_unknown_pmids(findings, allowed, *, context)` | Removes model-generated sentences citing identifiers not present in the tool artifacts |
| `services/llm.py` | `async query_llm(prompt, system="")`, `async query_small_llm(prompt, system="", max_tokens=None)`, `async query_big_llm(prompt, system="")` | Anthropic Messages API wrappers for the three configured model tiers |
| `services/llm.py` | `parse_llm_response(response)`, `parse_last_json_array(response)`, `parse_last_json_object(response)`, `strip_markdown_fences(text)` | Tolerant parsing of model output into lists and objects |
| `services/embeddings.py` | `embed(texts) -> list[list[float]]`, `async embed_async(texts) -> list[list[float]]` | BioLORD-2023 SentenceTransformer encoding with a lazy, lock-guarded model load |
| `services/condition_extraction.py` | `async extract_conditions(drug, articles) -> ExtractionResult` | One small-LLM call per Europe PMC abstract extracting stated treated conditions |
| `services/condition_grouping.py` | `async group_conditions(conditions, approved_indications) -> list[GroupedCondition]` | Groups extracted condition strings into canonical candidates, dropping approved ones |
| `services/run_repository.py` | `AnalysisRunRepository.create_run(...)`, `.start_attempt(...)`, `.append_event(...)`, `.complete_validated_attempt(...)`, `.fail_attempt(...)`, `.request_cancellation(run_id)`, `.record_attempt_cost(...)` | Transactional lifecycle for runs, attempts, progress events, and per-candidate costs |
| `services/job_store.py` | `JobStore.create(drug_name, *, job_id=None) -> Job`, `JobStore.get(job_id)`, `Job.emit(phase, message)` | In-memory async-job model backing the polling analyses API |
| `services/progress.py` | `set_emitter(emit)`, `emit_progress(phase, message)` | Context-var progress channel used by tools to report pipeline phases |
| `services/cost_tracking.py` | `calculate_usage(*, model, input_tokens, output_tokens, cache_read_tokens, cache_write_5m_tokens, cache_write_1h_tokens) -> LlmUsage`, `record_anthropic_response(response)`, `candidate_cost_scope(candidate)` | Token and dollar accounting per model and per candidate |
| `services/precision_metrics.py` | `score_ranked_candidate_precision(reports, spec) -> CandidatePrecisionResult`, `score_reviews(predictions, reviews) -> PrecisionSummary` | Scores candidate precision against labelled specs and expert reviews, with Wilson intervals |
| `services/seed_reports.py` | `load_fresh_seed_report(drug_name) -> SupervisorOutput \| None` | Loads a committed seed report when `SEED_REPORTS_ENABLED` and the file is fresh |

### Data sources

| Path | Signature | Purpose |
|------|-----------|---------|
| `data_sources/base_client.py` | `BaseClient._request(method, url, *, params=None, json_body=None, headers=None, as_text=False) -> Any` | Retrying HTTP core (429/5xx, exponential backoff) shared by every client |
| `data_sources/base_client.py` | `log_data_source_failure(source, url, context, error, cache_dir=...) -> None` | Appends a terminal failure record to `cache/data_source_failures.log` |
| `data_sources/open_targets.py` | `get_drug(chembl_id)`, `get_rich_drug_data(chembl_id)`, `get_target_data(target_id)`, `get_target_evidences(target_id, efo_ids)` | GraphQL fetches for drug, target, and per-disease evidence records |
| `data_sources/open_targets.py` | `get_drug_competitors(chembl_id, min_stage="PHASE_3", date_before=None) -> CompetitorRawData`, `rank_competitor_siblings(...) -> CompetitorRanking` | Competitor seed discovery through shared-target sibling drugs |
| `data_sources/open_targets.py` | `resolve_disease_id(name)`, `get_disease_synonyms(disease_name)`, `get_disease_drugs(disease_id)` | Disease name → EFO id resolution and disease-scoped drug lists |
| `data_sources/clinical_trials.py` | `search_trials(drug, mesh_term, date_before=None) -> SearchTrialsResult` | Paginated CT.gov v2 search for one drug × MeSH-resolved indication |
| `data_sources/clinical_trials.py` | `get_completed_trials(...)`, `get_terminated_trials(...)`, `get_landscape(mesh_term, date_before=None, top_n=50) -> IndicationLandscape`, `get_trial(nct_id) -> Trial` | Completed/terminated slices, the competitor landscape aggregate, and single-trial lookup |
| `data_sources/pubmed.py` | `search(query, max_results=None, date_before=None)`, `search_complete(query, page_size=None, date_before=None)`, `get_count(query, date_before=None)` | E-utilities esearch with a `sortpubdate` post-guard for holdout runs |
| `data_sources/pubmed.py` | `fetch_abstracts(pmids, batch_size=None) -> list[PubmedAbstract]`, `fetch_pubtypes(pmids, batch_size=None)` | Batched efetch plus XML parsing into typed abstracts |
| `data_sources/europe_pmc.py` | `fetch_citation_counts(pmids, batch_size=...) -> dict[str, int]` | Citation counts used to rank safety literature |
| `data_sources/europe_pmc.py` | `search_by_drug(drug_name, date_before=None)`, `search_filtered_by_drug(drug, ...)`, `fetch_disease_annotations(pmids, ...)` | Drug-scoped literature retrieval and text-mined disease annotations |
| `data_sources/chembl.py` | `async resolve_drug_name(drug_name, cache_dir=...) -> str`, `async get_all_drug_names(chembl_id, cache_dir=...) -> list[str]`, `async get_drug_family_chembl_ids(chembl_id, cache_dir=...) -> set[str]` | Module-level drug-identity resolution backed by per-drug JSON caches |
| `data_sources/chembl.py` | `ChEMBLClient.get_molecule(chembl_id)`, `.get_atc_description(atc_code)` | Molecule metadata and ATC code descriptions |
| `data_sources/fda.py` | `get_label_indications(drug_name)`, `get_all_label_indications(drug_names)`, `get_label_safety(drug_name)`, `get_all_label_safety(drug_names)` | openFDA label indication text and boxed-warning/safety records |
| `data_sources/drugbank.py` | `DrugBankClient.get_drug(drug_name)`, `.get_interactions(drug_id)` | Stub client; no implementation behind it |

### Models, report, regression, infrastructure

| Path | Signature | Purpose |
|------|-----------|---------|
| `models/` | `Trial`, `SearchTrialsResult`, `IndicationLandscape`, `ApprovalCheck`, `DrugData`, `TargetData`, `Association`, `EvidenceRecord`, `RichDrugData`, `PubmedAbstract`, `EuropePMCArticle`, `MoleculeData`, `DrugProfile`, `EvidenceSummary`, `FDALabelSafetyRecord`, `DrugSafetyAssessment` | Pydantic contracts; every model carries a `coerce_nones` before-validator |
| `agents/supervisor/supervisor_output.py` | `SupervisorOutput`, `CandidateFindings`, `CandidateBlurb` | The run's final structured object: candidates, per-disease findings, ranked top diseases, summary |
| `report/format_report.py` | `format_report(output: SupervisorOutput) -> str` | Renders the full Markdown report, including blurb splicing and PMID/NCT linkification |
| `regression/harness.py` | `compare_reports(golden, current) -> list[Diff]`, `has_errors(diffs)`, `render_diffs(diffs)` | Field-by-field report diff over candidates, mechanism, per-disease findings, and trial counts |
| `helpers/drug_helpers.py` | `normalize_drug_name(name) -> str`, `async seed_drug_intake(drug_name, cache_dir, date_before=None) -> DrugIntake` | Name normalization and the once-per-run resolution of ChEMBL id, aliases, first approval, approved indications |
| `utils/cache.py` | `cache_key(namespace, params)`, `cache_get(namespace, params, cache_dir)`, `cache_set(namespace, params, data, cache_dir, ttl=None)` | SHA-256-keyed JSON disk cache with read-time expiry |
| `config.py` | `Settings(BaseSettings)`, `get_settings() -> Settings` | Frozen settings loaded from `.env` then `CONSTANTS_FILE` (default `.env.constants`); tunable limits have no defaults |
| `constants.py` | module-level constants | URLs, timeouts, phase and MeSH vocabularies, adverse-event query legs, OT datasource weights, blocklists |
| `observability.py` | `configure_logging(level)`, `log_context(**fields)`, `bind_log_context(**fields)` | JSON log formatter and context-var log enrichment including trace id |
| `tracing.py` | `setup_tracing()`, `shutdown_tracing()` | Opt-in OpenTelemetry → Langfuse export with LangChain instrumentation |
| `metrics.py` | `record_http_request(...)`, `analysis_started(...)`, `analysis_finished(...)`, `record_dependency_request(...)` | Prometheus counters, histograms, and gauges exposed at `/metrics` |
| `markers.py` | `no_review`, `is_no_review(obj)` | Marks code excluded from the code-review agent |

### ML models

`ml_models/trial_risk/` trains and applies a trial-risk classifier: `data.load_labeled_trials`, `features.build_features` and
`vectorize`, `literature.compute_signals` / `signals_for_trial` (pgvector-backed literature features around a trial's cutoff),
`train.main_async`, `score.score_trial_async`, and `inspect.main_async` for per-trial inspection. `ml_models/success_classifier/`
holds `features.build_features` and `labels.load_labeled_pairs` for a trial-success model. `ml_models/repurposing_probe/probe.py`
and `ml_models/safety_signal_probe.py` are standalone offline probes with their own `main()`.

## Entry points

| Entry point | How it is run | Notes |
|-------------|---------------|-------|
| `scout` CLI | `scout find -d <drug> [--out-dir DIR] [--no-write] [--date-before YYYY-MM-DD]` | Console script `indication_scout.cli.cli:main`; writes Markdown to `snapshots/` (holdouts to `snapshots/holdouts/`) and JSON to `test_reports/` |
| `scout investigate` | `scout investigate -d <drug> -i <indication> [--out-dir DIR] [--no-write] [--date-before ...]` | Fixed pair; no candidate discovery, no JSON payload |
| `scout render` | `scout render -i <payload.json> [--out-dir DIR] [--no-write]` | Re-renders a saved `SupervisorOutput` JSON to Markdown |
| `scout diff-report` | `scout diff-report <golden.json> <current.json>` | Diffs two report payloads via `regression.harness.compare_reports` |
| FastAPI app | `uvicorn indication_scout.api.main:app --reload` | `GET /health`, `/metrics` (Prometheus ASGI mount), and the routers below; serves `frontend/dist` at `/` when built |
| Analyses API | `POST /api/analyses`, `GET /api/analyses/{job_id}`, `GET /api/analyses/{job_id}/report.md`, `DELETE /api/analyses/{job_id}` | `_execute` runs the job in the background, persisting attempts, progress events, and costs via `AnalysisRunRepository` |
| Drilldown API | `GET /api/trials/{nct_id}`, `GET /api/pubmed/{pmid}`, `GET /api/targets/{target_id}` | Direct typed lookups against CT.gov, PubMed, and Open Targets |
| Examples API | `GET /api/examples/{drug}`, `GET /api/examples/{drug}/report.md` | Serves committed seed reports from `seed_examples/`, cached in-process |
| React frontend | `cd frontend && npm run dev` (Vite, port 5173, proxies `/api` to 8000) | `App.tsx` + `useAnalysis.ts` poll the job API; four tabs (Overview, Mechanism, Clinical Trials, Literature) |
| Streamlit app | `streamlit run app.py` | Alternative single-page UI calling `run_analysis` directly |
| Runners | `indication_scout.runners.rag_runner.run_rag(drug_name, db, cache_dir)`; `runners.pubmed_runner.run_candidate(drug_name)` | Standalone RAG and PubMed paths outside the agent loop |
| Makefile | `make check` (lint, format-check, typecheck, tests, contract, regression, frontend), `make ci`, `make regression` | Also `create-tables`, `prefetch-model`, `seed-examples`, `container-smoke`, `observability-up/down` |
| Migrations | `alembic upgrade head` | Five revisions creating the abstract and analysis-run tables |

`main.py` at the repository root is an empty file.

## External dependencies

| Service | Protocol | Used by | Auth |
|---------|----------|---------|------|
| Open Targets Platform | GraphQL | `OpenTargetsClient`, `_OTSearchClient` (drug-name resolution) | none |
| ClinicalTrials.gov v2 | REST | `ClinicalTrialsClient` | none |
| PubMed / NCBI E-utilities | REST + XML | `PubMedClient`, `disease_helper` MeSH resolution | optional `NCBI_API_KEY` / `PUBMED_API_KEY` |
| Europe PMC | REST | `EuropePMCClient` | none |
| ChEMBL | REST | `ChEMBLClient` | none |
| openFDA | REST | `FDAClient` | optional `OPENFDA_API_KEY` |
| Anthropic Messages API | REST | `services/llm.py`, `ChatAnthropic` in the agent loops | `ANTHROPIC_API_KEY` required at first call |
| PostgreSQL + pgvector | SQL | `pubmed_abstracts` embeddings, analysis-run persistence | `DATABASE_URL`, `DB_PASSWORD` |
| Hugging Face | model download | BioLORD-2023 via SentenceTransformer | none |
| Langfuse (OTLP) | HTTP | `tracing.py`, opt-in via `TRACING_ENABLED` | `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` |
| ip-api geolocation | HTTP | `api/main.py` visitor logging middleware | none |

## Doc drift

- `docs/ARCHITECTURE.md` states that `investigate_top_candidates` is added to the supervisor tool set "in holdout mode
  (`date_before` set) or when `supervisor_fanout` is on". In `agents/supervisor/supervisor_tools.py` the tool set depends only on
  `get_settings().supervisor_fanout`; a comment there records that holdout runs are no longer special-cased.
- `docs/ARCHITECTURE.md` and `CLAUDE.md` describe `condition_extraction.py` and `condition_grouping.py` as built but not wired.
  That still holds: no module under `src/` or `scripts/` imports either. Correspondingly, `EuropePMCClient.search_by_drug` and
  `search_filtered_by_drug` have no callers in `src/`; the only production use of the client is
  `fetch_citation_counts` from `agents/literature/pubmed_ae.py`.
- `CLAUDE.md` says the supervisor coordinates sub-agents via LangGraph's prebuilt `create_react_agent`. The supervisor and
  clinical-trials agents use the custom gated loop in `agents/_react_loop.py`; only the literature and mechanism agents use
  `create_react_agent`. `docs/ARCHITECTURE.md` describes this correctly.
- `README.md` lists `frontend/`, but not the root `app.py` Streamlit UI, which also calls `run_analysis` and is a second
  interactive entry point.
- `README.md` links `GLOSSARY.md`, `DEV_SETUP.md`, and `holdout.md` at the repository root; the tracked files are
  `docs/reference/glossary.md`, `docs/ops/dev_setup.md`, and `docs/reference/holdout.md`.

## Coverage notes

`services/retrieval.py` (2492 lines), `agents/supervisor/supervisor_tools.py` (2285 lines), `constants.py` (761 lines), and
`data_sources/open_targets.py` (1185 lines) were read at the level of their public surface and control flow, not line by line;
private helpers inside them are summarized rather than enumerated. The `frontend/`, `scripts/`, `notebooks/`, `alembic/`, and
`tests/` trees were mapped by file and entry point rather than read in full.
