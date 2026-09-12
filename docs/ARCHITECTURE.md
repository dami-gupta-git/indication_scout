# Architecture

## Project Overview

IndicationScout is an agentic drug repurposing system. A drug name goes in; coordinated AI agents query multiple biomedical data sources (Open Targets, ClinicalTrials.gov, PubMed, Europe PMC, ChEMBL, openFDA) and produce a repurposing report identifying candidate indications worth investigating.

### Directory Structure

```
indication_scout/
├── src/indication_scout/          # Main source code
│   ├── __init__.py                # Package initialization
│   ├── config.py                  # Application settings (pydantic-settings; .env + .env.constants)
│   ├── constants.py               # URLs, timeouts, lookup maps, vaccine keywords, MeSH constants
│   ├── markers.py                 # `no_review` marker (excludes items from the code-review agent)
│   ├── agents/                    # Sub-agents and supervisor (custom gated ReAct loop, _react_loop.py)
│   │   ├── base.py                # BaseAgent ABC (legacy, unused by ReAct agents)
│   │   ├── _trial_formatting.py   # Shared trial table / phase distribution helpers
│   │   ├── _trial_signals.py      # Deterministic trial FACTS (highest_completed_phase, phase3_terminated_for_cause)
│   │   ├── supervisor/            # Top-level supervisor agent
│   │   ├── literature/            # PubMed retrieval + synthesis sub-agent
│   │   ├── clinical_trials/       # ClinicalTrials.gov sub-agent
│   │   └── mechanism/             # Open Targets mechanism sub-agent
│   ├── api/                       # FastAPI app (/health + async analyses, drilldown, examples routes; serves built React frontend in prod)
│   ├── cli/                       # `scout` CLI entry point (cli.py)
│   ├── data_sources/              # Async API clients
│   │   ├── base_client.py         # BaseClient: aiohttp + retry/backoff
│   │   ├── open_targets.py        # OpenTargetsClient (GraphQL)
│   │   ├── clinical_trials.py     # ClinicalTrialsClient (REST v2)
│   │   ├── pubmed.py              # PubMedClient (NCBI E-utilities)
│   │   ├── europe_pmc.py          # EuropePMCClient (citation counts + drug-scoped literature search)
│   │   ├── chembl.py              # ChEMBLClient + drug-name resolution helpers
│   │   ├── fda.py                 # FDAClient (openFDA labels)
│   │   └── drugbank.py            # DrugBankClient (stub)
│   ├── db/                        # SQLAlchemy session factory
│   ├── helpers/                   # `normalize_drug_name`, etc.
│   ├── ml_models/                 # Optional: success_classifier, trial_risk modules
│   ├── models/                    # Pydantic data contracts
│   │   ├── model_open_targets.py
│   │   ├── model_clinical_trials.py
│   │   ├── model_pubmed_abstract.py
│   │   ├── model_europe_pmc.py
│   │   ├── model_chembl.py
│   │   ├── model_drug_profile.py
│   │   └── model_evidence_summary.py
│   ├── prompts/                   # LLM prompt templates (.txt files; incl. pmid_direction, extract_fda_approval_single, synthesize)
│   ├── regression/                # Snapshot regression harness (diff.py, harness.py) — backs `scout diff-report`
│   ├── report/                    # `format_report` — SupervisorOutput → markdown
│   ├── runners/                   # Standalone runner scripts (pubmed_runner, rag_runner)
│   ├── services/                  # Business logic
│   │   ├── llm.py                 # Anthropic SDK wrappers (query_llm, query_small_llm)
│   │   ├── embeddings.py          # BioLORD-2023 embeddings
│   │   ├── disease_helper.py      # LLM disease normalization + MeSH descriptor resolver
│   │   ├── pubmed_query.py        # Query building
│   │   ├── retrieval.py           # RAG: drug profile, semantic search, synthesis
│   │   ├── condition_extraction.py # Europe PMC abstract → stated treated-conditions (not yet wired in)
│   │   ├── condition_grouping.py  # Extracted conditions → candidate indications (not yet wired in)
│   │   ├── approval_check.py      # openFDA label + LLM approval extraction
│   │   ├── dev_stage.py           # LLM-judged development-stage tier from trial facts
│   │   ├── judge_interpretive.py  # LLM interpretation of resolved facts (keeps blurbs consistent with dev_stage)
│   │   ├── analysis_runner.py     # `run_analysis` — shared CLI/API entry point (owns DB session, drug-name normalization)
│   │   ├── job_store.py           # In-memory async-job model + polling (backs POST/GET /api/analyses)
│   │   └── progress.py            # Pipeline phase definitions + `emit_progress`
│   ├── sqlalchemy/                # ORM models (pubmed_abstracts with pgvector)
│   └── utils/                     # cache.py (shared file cache)
├── tests/                         # Test suite
│   ├── unit/                      # Unit tests (no network)
│   ├── integration/               # Integration tests (hits real APIs)
│   └── conftest.py                # Shared fixtures
├── docs/                          # Documentation
├── for_me/                        # Personal notes (findings.md is source of truth)
├── cache/                        # Disk cache for API/LLM responses (per-namespace JSON, config-driven TTL (currently 60 days))
└── pyproject.toml                 # Project metadata & dependencies
```

### Current State

| Component | Status | Description |
|-----------|--------|-------------|
| Data Sources | **Complete** | OpenTargetsClient, ClinicalTrialsClient, PubMedClient, EuropePMCClient, ChEMBLClient, FDAClient; DrugBankClient is a stub |
| Data Models | **Complete** | Pydantic models for all data contracts (Open Targets, ClinicalTrials, PubMed, Europe PMC, ChEMBL, DrugProfile, EvidenceSummary) |
| BaseClient | **Complete** | Retry with exponential backoff; persistent failure log via `log_data_source_failure` |
| File Cache | **Complete** | Shared `utils/cache.py` used by all clients and services (`cache/<namespace>/<sha>.json`, config-driven TTL (currently 60 days)) |
| Services | **Complete** | `llm.py`, `embeddings.py`, `disease_helper.py`, `pubmed_query.py`, `approval_check.py`, `retrieval.py` (build_drug_profile, expand_search_terms, extract_organ_term, fetch_new_abstracts, embed_abstracts, fetch_and_cache, semantic_search, synthesize, get_drug_competitors) |
| Agents | **Complete** | Supervisor + literature, clinical_trials, mechanism sub-agents — all built on the custom gated ReAct loop (`agents/_react_loop.py`). `BaseAgent` ABC still exists in `agents/base.py` but is unused. |
| API | **Complete** | FastAPI app: `/health` plus async `analyses` (POST/GET/report.md/DELETE), `drilldown`, and `examples` routers (in `api/routes/`); CORS + visitor/bot logging; serves the built React frontend in prod |
| CLI | **Complete** | `scout find` (run pipeline), `scout investigate` (run pipeline on a fixed drug+disease pair, no candidate discovery), `scout render` (re-render saved JSON), `scout diff-report` (diff two JSON snapshots) — in `cli/cli.py` |
| Literature candidate sourcing | **Built, not wired** | Europe PMC retrieval, condition extraction and grouping are implemented and tested; nothing calls them. Invocation point undecided — see `design_europe_pmc.md` and `PLAN_europe_pmc.md` |

---

## Layered Architecture

```
CLI / API ──> Supervisor agent ──> {Literature, ClinicalTrials, Mechanism} sub-agents
                   │                              │
                   └─────── Services ─────────────┤
                            (RetrievalService,    │
                             approval_check,      │
                             disease_helper,      │
                             llm, embeddings)     │
                                                  ▼
                                   Data source clients (async)
                                   ──────────────────────────
                                   OpenTargetsClient (GraphQL)
                                   ClinicalTrialsClient (REST v2)
                                   PubMedClient (E-utilities)
                                   ChEMBLClient (REST)
                                   FDAClient (openFDA)
                                                  │
                                                  ▼
                                   Pydantic models (models/) — only contracts that cross
                                   module boundaries
```

Agents never see raw API responses — all data crosses module boundaries as Pydantic `BaseModel` instances.

---

## Agent Layer

The supervisor and clinical-trials agents are built using a custom gated ReAct loop
(`build_gated_react_loop` in `agents/_react_loop.py`) that ends the loop as soon as the
agent's `finalize_*` tool succeeds. The literature and mechanism agents use LangGraph's
prebuilt `create_react_agent`. `BaseAgent` (in `agents/base.py`) is a legacy ABC and is not
used by the active ReAct-style agents.

The split is driven by whether the agent's `finalize_*` tool has a reject path. Supervisor's
`finalize_supervisor` and clinical-trials' `finalize_analysis` can fail validation (e.g.
empty-summary/critique-not-run; not every shown trial classified) and must loop back to the
model to retry — that requires a graph that can re-enter the model node, hence the custom
gated loop. Literature's and mechanism's finalize tools are pure termination signals with no
reject path, so the simpler prebuilt `create_react_agent` (with `return_direct=True`)
suffices. The gated loop also ends the moment finalize *succeeds*, skipping the prebuilt's
extra trailing model turn — assembly reads typed artifacts off the ToolMessages, never the
final AIMessage, so that turn would be pure latency.

### Supervisor (`agents/supervisor/`)

`build_supervisor_agent(llm, svc, db, date_before)` returns
`(compiled_agent, get_merged_allowlist, get_auto_findings, get_approval_labels)`. The supervisor wraps each
sub-agent as a tool and orchestrates the run via a gated ReAct loop
(`build_gated_react_loop`). After the loop finishes, `run_supervisor_agent` walks the
message history, canonicalises disease names against the merged competitor + mechanism
allowlist, and assembles a `SupervisorOutput`.

Tools available to the supervisor (all in `supervisor_tools.py`):

| Tool | Purpose |
|------|---------|
| `find_candidates` | Surface competitor + mechanism disease candidates (runs the merge/dedup over both seed sources) |
| `analyze_mechanism` | Run the mechanism sub-agent (returns `MechanismOutput`); buffers raw mechanism candidates |
| `analyze_literature` | Run the literature sub-agent for one disease |
| `analyze_clinical_trials` | Run the clinical-trials sub-agent for one disease |
| `investigate_top_candidates` | Holdout/fan-out only: parallel fan-out over top candidates |
| `get_drug_briefing` | Read-only view of accumulated drug-level facts |
| `critique_ranking` | Audit the draft ranking order; mandatory before `finalize_supervisor` |
| `finalize_supervisor` | Last action; returns the supervisor's narrative summary + top-5 blurbs |

`investigate_top_candidates` is added to the tool set in holdout mode (`date_before` set) or
when `supervisor_fanout` is on; in pure fan-out mode the per-candidate `analyze_literature` /
`analyze_clinical_trials` tools are removed so the LLM must use the parallel path.
`finalize_supervisor` is rejected until `critique_ranking` has run this turn.

Candidate discovery and mechanism analysis share one run-scoped drug-intake task keyed by the
normalized drug name. The task resolves the ChEMBL ID, aliases, first approval year, and FDA-approved
indications once. Both seed tools await the same result, which is written to the existing supervisor
drug-facts entry before either path uses those facts. This task is held in memory for one supervisor
run; persistent API and model-response caching remains in the data-source and service layers.

When `date_before` is set, the supervisor (which always loads `prompts/supervisor.txt`)
forwards the cutoff to the literature and clinical-trials sub-agents. Mechanism analysis
(Open Targets) is always current because there is no date-filtering API.

### Sub-agents

Each sub-agent has the same shape:

```
agents/<name>/
  <name>_agent.py    # build_<name>_agent + run_<name>_agent
  <name>_tools.py    # @tool definitions, response_format="content_and_artifact"
  <name>_output.py   # Pydantic output model
```

| Agent | Tools | Output |
|-------|-------|--------|
| **Literature** | `build_drug_profile`, `expand_search_terms`, `fetch_and_cache`, `semantic_search`, `safety_search`, `synthesize`, `finalize_analysis` | `LiteratureOutput` |
| **Clinical Trials** | `check_fda_approval`, `search_trials`, `get_completed`, `get_terminated`, `get_landscape`, `finalize_analysis` | `ClinicalTrialsOutput` |
| **Mechanism** | `get_drug`, `get_target_associations`, `finalize_analysis` | `MechanismOutput` |

The mechanism agent additionally has `mechanism_candidates.py` (`select_top_candidates`) and
`mechanism_row_builder.py` (`build_candidate_rows`) for post-LLM candidate scoring,
filtered against an FDA-approved disease set and trimmed to `MECHANISM_TOP_CANDIDATES`.
`select_top_candidates` aggregates `direction_on_target` × `direction_on_trait` by majority vote
(`_MAJORITY_THRESHOLD = 0.8`) to label each (target, disease) pair as LoF-driven / GoF-driven /
inconclusive, keeps only rows where the drug's action direction opposes the disease-driving direction
(LoF drug ↔ GoF-driven, GoF drug ↔ LoF-driven), drops diseases that are FDA-approved via
`get_fda_approved_disease_mapping` or in `BROADENING_BLOCKLIST`, and sorts by `ranking_score`.

`mechanism/ot_score.py` (`recompute_overall`) is a pure-function local reproduction of Open
Targets' overall association score (a weighted harmonic sum over per-datasource scores,
normalized by `OT_PLATFORM_DATASOURCE_COUNT`; weights from `OT_DATASOURCE_WEIGHTS`, default
`OT_DEFAULT_DATASOURCE_WEIGHT`). It exists so the leaky `clinical_precedence` datasource can be
dropped in **holdout** mode — that channel encodes current trials/approvals and would leak
post-cutoff signal. Holdout ranks by the recomputed `ranking_score` (leak-free); production
ranks by OT's published `overall_score`. Reproduces the published score to ~0.01 MAE.

After each sub-agent run, `run_<name>_agent` walks the message history and pulls each
tool's typed artifact off `ToolMessage.artifact`, assembling them into the typed output.

#### Why `content_and_artifact` instead of `response_format`

Tools use `response_format="content_and_artifact"`: each tool returns a short LLM-visible
`content` string plus a typed Pydantic `artifact`. The LLM only ever reads `content`; the
artifact is pulled straight off `ToolMessage.artifact` by plain Python after the loop ends
(see above). This was chosen over LangGraph's built-in `response_format=<Model>` (structured
output generated by the LLM itself at the end of the loop) for two reasons:

- **No re-typing of exact values through the LLM.** With `response_format`, the model has to
  restate the final structured object from its memory of the conversation — fine for prose
  (a paraphrased weather summary, a one-sentence city blurb) but not for exact identifiers.
  This project's tools return NCT IDs, PMIDs, and OT scores; having the LLM retype those from
  memory risks silent corruption (a transposed digit in an NCT ID or PMID is a wrong
  citation). `content_and_artifact` keeps those values sourced from the tool's own return
  value, never from LLM recall — consistent with the project rule against fabricating or
  reconstructing scientific/clinical data.
- **No extra trailing turn.** `response_format` requires one more model turn after the last
  tool call to produce the structured response. The gated ReAct loop (see above) already ends
  the instant `finalize_*` succeeds, so assembly never depends on a final AIMessage — adding a
  `response_format` turn back in would reintroduce the exact latency that loop is designed to
  avoid.

#### The floor/cap pattern

The model proposes, deterministic code clamps. Two instances: `services/dev_stage.py`'s
`_enforce_tier_floor` raises the LLM's proposed development-stage tier when the trial record
proves a higher tier than the LLM guessed (e.g. an active pure Phase-3 trial on record raises
an LLM guess of `early_phase` up to `active_phase3`); it never lowers a tier the LLM got right
or overcalled. `services/retrieval.py`'s `PubMedRetriever.synthesize` forces evidence
`strength`/`direction` *down* to `"none"` whenever `evidence_basis != "drug_specific"` (see
`EvidenceSummary` below). Neither lets the model contradict the underlying data.

### SupervisorOutput

```
SupervisorOutput
 |-- drug_name: str
 |-- candidate_diseases: list[str]          # Diseases in the merged allowlist
 |-- mechanism: MechanismOutput | None
 |-- disease_findings: list[CandidateFindings]   # top_diseases first (rank order), then the rest
 |        |-- disease: str
 |        |-- source: "competitor" | "mechanism" | "both"
 |        |-- approval_relationship: "contaminated" | "combination_only" | "none"  # label-grounded, set upstream
 |        |-- literature: LiteratureOutput | None
 |        |-- clinical_trials: ClinicalTrialsOutput | None
 |        +-- blurb: CandidateBlurb | None        # structured per-candidate synthesis (top 3 only)
 |-- top_diseases: list[str]                # Ranked top diseases (max 5); subset of disease_findings
 +-- summary: str                           # Supervisor's narrative
```

`report/format_report.py` renders this into markdown for the CLI.

---

## BaseClient Infrastructure

All data source clients inherit from `BaseClient`, which provides common infrastructure for
reliable API communication.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              BaseClient                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Request methods                                                             │
│  ├── _request()        — Low-level HTTP with retry                          │
│  ├── _graphql()        — GraphQL POST                                       │
│  ├── _rest_get()       — REST GET (JSON)                                    │
│  └── _rest_get_xml()   — REST GET returning XML text                        │
│                                                                              │
│  Retry logic                                                                 │
│  └── Exponential backoff (1s, 2s, 4s, capped at 30s), max 3 retries        │
│  └── Retries on: 429, 500, 502, 503, 504                                    │
│                                                                              │
│  Failure logging                                                             │
│  └── log_data_source_failure() appends a tab-separated line to             │
│      cache/data_source_failures.log on terminal failure.                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

Configuration values come from `Settings` (`default_timeout`, `default_max_retries`).

---

## Disk Cache

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Disk Cache                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Layout: cache/<namespace>/<sha256>.json                           │
│  Key:    SHA-256 of {"ns": namespace, **params} (JSON, sorted keys)│
│  Entry:  {"data": ..., "cached_at": <iso>, "ttl": <secs>}           │
│  TTL:    CACHE_TTL (currently 60 days) unless overridden per-call   │
│  Expiry: checked on read; expired/corrupt entries auto-deleted      │
│                                                                      │
│  Namespaces in active use:                                           │
│  ├── drug, target, disease_drugs, competitors_raw,                  │
│  │   disease_id_resolver                       (OpenTargets)        │
│  ├── ct_search, ct_completed, ct_terminated,                        │
│  │   ct_landscape                                (ClinicalTrials)    │
│  ├── pubmed_search                              (PubMed)            │
│  ├── atc_description, resolve_drug_name         (ChEMBL)            │
│  ├── fda_label, fda_label_indications,          (FDA / approval)    │
│  │   fda_approval_check                                              │
│  ├── disease_norm, disease_merge,               (disease_helper)    │
│  │   pubmed_count, mesh_resolver                                     │
│  ├── competitors_merged, synthesize, organ_term  (retrieval)         │
│  └── europe_pmc_search,                          (Europe PMC        │
│      europe_pmc_extraction                        sourcing)          │
│                                                                      │
│  In addition, OpenTargetsClient persists per-target evidence files  │
│  via _save_target_evidences (separate JSON), and ChEMBLClient       │
│  persists drug-name caches (_save_chembl_names).                    │
│                                                                      │
│  The two Europe PMC namespaces are written against the shared       │
│  helper but carry no per-call TTL yet, so they take the global      │
│  value; retrieval goes stale as papers publish while extraction is  │
│  keyed on an immutable record id, so the two may not share it.      │
│  Undecided — see PLAN_europe_pmc.md section 9.                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data-source contracts

Per-source data models, client methods, and caching are documented in their own files:

| Source | Doc |
|--------|-----|
| Open Targets (`DrugData`, `TargetData`, `RichDrugData`, associations, evidence) | `reference/open_targets.md` |
| ClinicalTrials.gov (`Trial`, search/completed/terminated/landscape results, MeSH filtering) | `features/clinical_trials_agent.md` |
| PubMed (`PubmedAbstract`, esearch/efetch, `EvidenceSummary`) | `reference/pubmed.md`, `reference/rag.md` |
| ChEMBL (`MoleculeData`, `ATCDescription`) | `reference/chembl.md` |

The approval-relationship label (`approved` / `contaminated` / `combination_only` / `none`) that gates candidates is
decided once upstream from the FDA label and threaded through every layer; see `features/approval_awareness.md`.

---

## Drug Safety

The literature agent's `safety_search` tool produces a **two-tier** safety signal per drug-disease
pair, merged into `EvidenceSummary` (see `reference/rag.md`) by `synthesize` and rendered by
`report/format_report.py`.

### Retrieval — `agents/literature/pubmed_ae.py::search_adverse_events`

Two query modes, both **citation-ranked** (not PubMed relevance) and **holdout-clean** (drug/disease
names + generic MeSH tags only, no future knowledge):

- **Drug-level** (`disease=None`): `"{drug}/adverse effects"[Majr]` — PubMed's curated "primarily
  about this drug's harms" major-topic tag. Precise (no 500-cap saturation); catches specific-named
  signals (agranulocytosis, bladder cancer) without knowing the event name. Falls back to a
  title/abstract query when MeSH indexing is sparse (`< AE_FALLBACK_MIN_HITS`).
- **Disease-scoped** (`disease` set): drug leg `AND` adverse-event vocabulary leg `AND` disease leg —
  recovers indication-specific safety papers the drug-level pool misses (e.g. rofecoxib × colorectal
  → the APPROVe trial). Query legs are `constants.py::AE_DISEASE_*`.

Ranking uses `EuropePMCClient.fetch_citation_counts` (`data_sources/europe_pmc.py`): PubMed's
term-frequency relevance sort buries landmark safety papers (APPROVe ranks outside PubMed's top 300
for a toxicity query despite 1,600+ citations); Europe PMC's `citedByCount` draws on a broader
citation graph than NCBI esummary's (blank) `pmcrefcount`. Holdout also applies the
`_filter_pmids_by_date` `sortpubdate` post-guard (PubMed's `maxdate` trusts the unreliable `pdat`).

`RetrievalService.safety_search` preserves both collections. Its combined view is used only for
drug-wide synthesis; the indication classifier receives only disease-scoped abstracts.

### Summarization

- **Drug-level** — `RetrievalService.summarize_safety` returns a typed source-separated assessment.
  Production boxed-warning text comes from the latest openFDA record per label set identifier.
  Open Targets warning rows remain metadata, and FAERS values remain non-causal associations.
  Holdout runs omit current label, Open Targets, and current pharmacovigilance data.
- **Disease-specific** — `RetrievalService.classify_indication_harm` adjudicates each disease-scoped
  PMID as confirmed harm, safety assessed only, irrelevant, or unclear. Confirmed harm requires a
  named adverse outcome and an exact supporting quote present in the supplied title or abstract.

No disease-scoped evidence, incomplete model output, and unverified source quotes produce an
unavailable result. A reviewed negative is not rendered as evidence that the drug is safe.

### Report rendering & ranking

- The DRUG-LEVEL blurb is collapsed (pick-first non-empty across candidates) and rendered ONCE in a
  top-level `## Drug Safety` section.
- The DISEASE-SPECIFIC signal renders per candidate: a `⚠️ **Indication-specific safety:**` block in
  the Literature section, and a `⚠️ safety signal reported for this indication` flag in the Summary
  table (via `supervisor_tools._safety_flag`, keyed on `indication_harm`).
- The flag reaches the ranking critic's FACT block, so the LLM weighs it (it can demote a flagged
  candidate below a comparable unflagged one — no hard rule). The drug-level `withdrawn` severity is
  deliberately NOT surfaced as a ranking flag: a historical drug-level withdrawal (e.g. thalidomide)
  was mis-attributed by the LLM as foreclosing each specific disease; `indication_harm` is
  per-candidate and cannot mis-attribute.

---

## ChEMBL & FDA

| Client | Methods |
|--------|---------|
| `ChEMBLClient` | `get_atc_description(atc_code)`, `get_molecule(chembl_id)` |
| (module-level) | `resolve_drug_name(drug_name)` → ChEMBL ID; `get_all_drug_names(chembl_id)` → list of synonyms |
| `FDAClient` | `get_label_indications(drug_name)`, `get_all_label_indications(drug_names)` |

ChEMBL IDs and drug-name lists are persisted in dedicated per-drug JSON files under
`cache/` (separate from the namespace cache).

---

## Services

| Service | Public surface |
|---------|----------------|
| `llm.py` | `query_llm`, `query_small_llm`, `parse_llm_response`, `parse_last_json_array`, `parse_last_json_object`, `strip_markdown_fences` |
| `embeddings.py` | `embed`, `embed_async` (BioLORD-2023 via SentenceTransformer) |
| `disease_helper.py` | `llm_normalize_disease`, `llm_normalize_disease_batch`, `merge_duplicate_diseases`, `pubmed_count`, `normalize_for_pubmed`, `normalize_batch`, `resolve_mesh_id` |
| `pubmed_query.py` | `get_pubmed_query(drug_name, disease_name)` |
| `retrieval.py` | `RetrievalService` — `build_drug_profile`, `get_drug_competitors`, `fetch_new_abstracts`, `embed_abstracts`, `fetch_and_cache`, `semantic_search`, `synthesize` (takes `approved_indications`; per-PMID directions via `_judge_pmid_directions`), `extract_organ_term`, `expand_search_terms` |
| `condition_extraction.py` | `build_prompt`, `parse_response`, `extract_conditions` — one small-LLM call per Europe PMC abstract, cached per article, failures counted in `ExtractionResult.skipped` rather than recorded as NONE |
| `condition_grouping.py` | `group_conditions` — one `merge_duplicate_diseases` call over all extracted names, returning `GroupedCondition` (canonical name, aliases, article keys) sorted by paper count, with already-approved conditions removed |
| `approval_check.py` | `get_approved_indications`, `list_approved_indications_at`, `list_approved_indications_from_labels`, `extract_approved_from_labels`, `get_all_fda_approved_diseases`, `get_fda_approved_disease_mapping` |
| `dev_stage.py` | `judge_dev_stage`, `dev_stage_phrase` (`DEV_STAGE_PHRASE` / `DEV_STAGE_TIERS`) — LLM-judged development-stage tier with a deterministic phase-band floor |
| `judge_interpretive.py` | `judge_interpretive` — isolated LLM call interpreting already-resolved facts so blurb fields don't contradict the authoritative dev_stage |
| `analysis_runner.py` | `run_analysis`, `build_agent` — shared CLI/API entry point; normalizes drug name, owns DB session lifecycle, threads `date_before` |
| `job_store.py` | `Job`, `JobStore` — in-memory async-job model backing the polling `analyses` API |
| `progress.py` | `emit_progress` + phase constants (`PHASE_CANDIDATES`, `PHASE_MECHANISM`, `PHASE_TRIALS`, `PHASE_LITERATURE`, `PHASE_SUMMARY`) |

---

## External Integrations

| Service | Type | Endpoint | Authentication |
|---------|------|----------|-----------------|
| Open Targets Platform | GraphQL | https://api.platform.opentargets.org/api/v4/graphql | None |
| ClinicalTrials.gov | REST v2 | https://clinicaltrials.gov/api/v2/ | None |
| PubMed / NCBI E-utilities | REST | https://eutils.ncbi.nlm.nih.gov/entrez/eutils/ | API key (optional) |
| Europe PMC | REST | https://www.ebi.ac.uk/europepmc/webservices/rest/search | None |
| ChEMBL | REST | https://www.ebi.ac.uk/chembl/api/data | None |
| openFDA | REST | https://api.fda.gov/ | API key (optional) |
| Anthropic | REST | Anthropic Messages API | API key required |

---

## Configuration

Application settings via `pydantic_settings.BaseSettings`. Two env files are loaded in
order: `.env` (secrets, DB credentials, model names) and `.env.constants` (tunable numeric
limits). Environment variables override both. The constants file path can be swapped via
`CONSTANTS_FILE=...`.

```python
Settings:
    # Database
    database_url: str
    db_password: str
    test_database_url: str | None

    # API keys
    openai_api_key: str = ""
    pubmed_api_key: str = ""
    anthropic_api_key: str = ""
    ncbi_api_key: str = ""
    openfda_api_key: str = ""

    # LLM
    llm_model: str = "claude-sonnet-4-6"
    small_llm_model: str = "claude-sonnet-4-6"
    big_llm_model: str = "claude-opus-4-6"
    embedding_model: str = "FremyCompany/BioLORD-2023"
    llm_max_tokens: int                # from .env.constants
    small_llm_max_tokens: int

    # App
    debug: bool = False
    log_level: str = "INFO"
    seed_reports_enabled: bool = True   # serve a committed seed report instead of running agents

    # Tracing (OpenTelemetry -> Langfuse; opt-in, all optional)
    tracing_enabled: bool = False
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_base_url: str = "https://us.cloud.langfuse.com"

    # Tunable limits (no defaults — must be present in .env.constants)
    default_timeout: float
    default_max_retries: int
    literature_top_k: int
    semantic_search_top_k: int
    pubmed_max_results: int
    pubmed_search_default_max_results: int
    pubmed_esummary_batch_size: int
    pubmed_efetch_batch_size: int
    rag_llm_concurrency: int
    rag_pubmed_concurrency: int
    rag_disease_concurrency: int
    europe_pmc_extraction_concurrency: int  # in-flight condition-extraction LLM calls
    clinical_trials_landscape_max_trials: int
    supervisor_candidate_cap: int          # trims the final ranked candidate list
    supervisor_investigation_cap: int      # how many top candidates the deep-dive fan-out investigates
    supervisor_fanout: bool                # expose investigate_top_candidates in non-holdout runs
    mechanism_signal_threshold: float
    mechanism_associations_cap: int
    mechanism_associations_per_target: int # top OT associations pulled per target before filtering
    mechanism_top_candidates: int          # final count of positive candidates surfaced
    disease_pubmed_min_results: int
    open_targets_page_size: int
    open_targets_competitor_prefetch_max: int
    open_targets_association_min_score: float
```

`Settings` is `frozen=True` and accessed via the cached `get_settings()` accessor.

---

## CLI

```bash
scout find -d <drug> [--out-dir DIR] [--no-write] [--date-before YYYY-MM-DD]
scout investigate -d <drug> -i <indication> [--out-dir DIR] [--no-write]
scout render -i <payload.json> [--out-dir DIR] [--no-write]
scout diff-report <golden.json> <current.json>
```

Defined in `cli/cli.py`. `find` loads `.env` and `.env.constants`, normalizes the drug name,
and delegates to `services.analysis_runner.run_analysis` (which builds the `ChatAnthropic`
LLM and supervisor agent and runs them). It writes the markdown report under `snapshots/`
(or `snapshots/holdouts/` when `--date-before` is set) and, for non-holdout runs, a
structured `SupervisorOutput` JSON dump under `test_reports/`. `render` re-renders a saved
JSON payload to markdown without re-running the pipeline; `diff-report` diffs two JSON
snapshots via `regression.harness.compare_reports` (the regression-harness comparison).

---

## Design Principles

1. **Separation of Concerns** — Data sources (clients) separate from domain logic (agents/services); agents never see raw API responses.
2. **Async-First** — All I/O is async via aiohttp; clients are async context managers.
3. **Graceful Degradation** — Retry with exponential backoff on 429/5xx; `DataSourceError` carries source name and context; terminal failures are logged to `cache/data_source_failures.log`.
4. **Shared Disk Cache** — JSON files in `cache/<namespace>/` with config-driven TTL (currently 60 days), SHA-256-keyed; used by all data source clients and services via `utils/cache.py`.
5. **Type Safety** — Full Pydantic validation with `coerce_nones` model validator on every external data model; Python 3.10+ type hints throughout.
6. **Model-Driven** — GraphQL/REST responses parsed into typed Pydantic models; Pydantic `BaseModel` contracts at every module boundary.
7. **No Fallbacks for Clinical Data** — Missing scientific/clinical values return `None` / empty structures, never defaults; this is a clinical genomics tool.
8. **Accuracy over Coverage** — Error by omission is acceptable; inaccurate output is not. Reject paths and allowlist guards are not loosened to "rescue" missing candidates.
