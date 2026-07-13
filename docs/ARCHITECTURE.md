# Architecture

## Project Overview

IndicationScout is an agentic drug repurposing system. A drug name goes in; coordinated AI agents query multiple biomedical data sources (Open Targets, ClinicalTrials.gov, PubMed, ChEMBL, openFDA) and produce a repurposing report identifying candidate indications worth investigating.

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
| Data Sources | **Complete** | OpenTargetsClient, ClinicalTrialsClient, PubMedClient, ChEMBLClient, FDAClient; DrugBankClient is a stub |
| Data Models | **Complete** | Pydantic models for all data contracts (Open Targets, ClinicalTrials, PubMed, ChEMBL, DrugProfile, EvidenceSummary) |
| BaseClient | **Complete** | Retry with exponential backoff; persistent failure log via `log_data_source_failure` |
| File Cache | **Complete** | Shared `utils/cache.py` used by all clients and services (`cache/<namespace>/<sha>.json`, config-driven TTL (currently 60 days)) |
| Services | **Complete** | `llm.py`, `embeddings.py`, `disease_helper.py`, `pubmed_query.py`, `approval_check.py`, `retrieval.py` (build_drug_profile, expand_search_terms, extract_organ_term, fetch_new_abstracts, embed_abstracts, fetch_and_cache, semantic_search, synthesize, get_drug_competitors) |
| Agents | **Complete** | Supervisor + literature, clinical_trials, mechanism sub-agents — all built on the custom gated ReAct loop (`agents/_react_loop.py`). `BaseAgent` ABC still exists in `agents/base.py` but is unused. |
| API | **Complete** | FastAPI app: `/health` plus async `analyses` (POST/GET/report.md/DELETE), `drilldown`, and `examples` routers (in `api/routes/`); CORS + visitor/bot logging; serves the built React frontend in prod |
| CLI | **Complete** | `scout find` (run pipeline), `scout investigate` (run pipeline on a fixed drug+disease pair, no candidate discovery), `scout render` (re-render saved JSON), `scout diff-report` (diff two JSON snapshots) — in `cli/cli.py` |

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
| **Literature** | `build_drug_profile`, `expand_search_terms`, `fetch_and_cache`, `semantic_search`, `synthesize`, `finalize_analysis` | `LiteratureOutput` |
| **Clinical Trials** | `check_fda_approval`, `search_trials`, `get_completed`, `get_terminated`, `get_landscape`, `finalize_analysis` | `ClinicalTrialsOutput` |
| **Mechanism** | `get_drug`, `get_target_associations`, `finalize_analysis` | `MechanismOutput` |

The mechanism agent additionally has `mechanism_candidates.py` (`select_top_candidates`) and
`mechanism_row_builder.py` (`build_candidate_rows`) for post-LLM candidate scoring,
filtered against an FDA-approved disease set and trimmed to `MECHANISM_TOP_CANDIDATES`.

`mechanism/ot_score.py` (`recompute_overall`) is a pure-function local reproduction of Open
Targets' overall association score (a weighted harmonic sum over per-datasource scores,
normalized by `OT_PLATFORM_DATASOURCE_COUNT`; weights from `OT_DATASOURCE_WEIGHTS`, default
`OT_DEFAULT_DATASOURCE_WEIGHT`). It exists so the leaky `clinical_precedence` datasource can be
dropped in **holdout** mode — that channel encodes current trials/approvals and would leak
post-cutoff signal. Holdout ranks by the recomputed `ranking_score` (leak-free); production
ranks by OT's published `overall_score`. Reproduces the published score to ~0.01 MAE.

After each sub-agent run, `run_<name>_agent` walks the message history and pulls each
tool's typed artifact off `ToolMessage.artifact`, assembling them into the typed output.

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
│  └── competitors_merged, synthesize, organ_term  (retrieval)         │
│                                                                      │
│  In addition, OpenTargetsClient persists per-target evidence files  │
│  via _save_target_evidences (separate JSON), and ChEMBLClient       │
│  persists drug-name caches (_save_chembl_names).                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Open Targets Data Structure

The `OpenTargetsClient` provides three primary entry points: `get_drug()` for drug data,
`get_target_data()` for target data, and `get_rich_drug_data()` which combines both.
Results are cached independently per namespace.

### DrugData

Drug names (pref_name, synonyms, trade names) are not on `DrugData` — they are fetched
separately via `data_sources.chembl.get_all_drug_names(chembl_id)` so a single source of
truth (ChEMBL) handles them and the result can be cached per ChEMBL ID. ChEMBL ID is the sole drug identifier.

```
DrugData
 |-- chembl_id: str = ""
 |-- drug_type: str | None = None
 |-- maximum_clinical_stage: str | None = None  # APPROVAL, PHASE_3, PHASE_2, etc.
 |-- mechanisms_of_action: list[MechanismOfAction] = []
 |        |-- mechanism_of_action: str = ""
 |        |-- action_type: str | None = None    # INHIBITOR, AGONIST, ANTAGONIST, etc.
 |        |-- target_ids: list[str] = []
 |        +-- target_symbols: list[str] = []
 |-- warnings: list[DrugWarning] = []
 |        |-- warning_type: str = ""
 |        |-- description: str | None = None
 |        |-- toxicity_class: str | None = None
 |        |-- country: str | None = None
 |        |-- year: int | None = None
 |        +-- efo_id: str | None = None
 |-- indications: list[Indication] = []
 |        |-- id: str = ""
 |        |-- disease_id: str = ""
 |        |-- disease_name: str = ""
 |        +-- max_clinical_stage: str | None = None
 |-- targets: list[DrugTarget] = []
 |        |-- target_id: str = ""  ──────────> use with get_target_data()
 |        |-- target_symbol: str = ""
 |        |-- mechanism_of_action: str = ""
 |        +-- action_type: str | None = None
 |-- adverse_events: list[AdverseEvent] = []
 |        |-- name: str = ""
 |        |-- meddra_code: str | None = None
 |        |-- count: int | None = None
 |        +-- log_likelihood_ratio: float | None = None
 |-- adverse_events_critical_value: float | None = None
 +-- atc_classifications: list[str] = []
```

### TargetData

```
TargetData
 |-- target_id: str = ""
 |-- symbol: str = ""
 |-- name: str = ""
 |-- function_descriptions: list[str] = []   # UniProt function paragraphs
 |-- associations: list[Association] = []
 |        |-- disease_id: str = ""
 |        |-- disease_name: str = ""
 |        |-- disease_description: str = ""
 |        |-- overall_score: float | None = None
 |        |-- datatype_scores: dict[str, float] = {}     # e.g. {"genetic_association": 0.8}
 |        |-- datasource_scores: dict[str, float] = {}   # per-datasource, e.g. {"clinical_precedence": 0.99}; consumed by ot_score.recompute_overall
 |        +-- therapeutic_areas: list[str] = []
 |-- pathways: list[Pathway] = []
 |        |-- pathway_id: str = ""
 |        |-- pathway_name: str = ""
 |        +-- top_level_pathway: str = ""
 |-- interactions: list[Interaction] = []
 |        |-- interacting_target_id: str = ""
 |        |-- interacting_target_symbol: str = ""
 |        |-- interaction_score: float | None = None
 |        |-- source_database: str = ""
 |        |-- biological_role: str = ""
 |        |-- evidence_count: int | None = None
 |        +-- interaction_type: str | None = None   # derived via INTERACTION_TYPE_MAP
 |-- drug_summaries: list[DrugSummary] = []         # One row per drug with nested diseases
 |        |-- id: str = ""
 |        |-- drug_id: str = ""
 |        |-- drug_name: str = ""
 |        |-- drug_type: str | None = None
 |        |-- max_clinical_stage: str | None = None
 |        +-- diseases: list[ClinicalDisease] = []
 |                 |-- disease_from_source: str = ""
 |                 |-- disease_id: str | None = None
 |                 +-- disease_name: str | None = None
 |-- expressions: list[TissueExpression] = []
 |        |-- tissue_id: str = ""
 |        |-- tissue_name: str = ""
 |        |-- tissue_anatomical_system: str = ""
 |        |-- rna: RNAExpression | None = None
 |        |        |-- value: float | None = None
 |        |        |-- quantile: int | None = None
 |        |        +-- unit: str = "TPM"
 |        +-- protein: ProteinExpression | None = None
 |                 |-- level: int | None = None
 |                 |-- reliability: bool | None = None
 |                 +-- cell_types: list[CellTypeExpression] = []
 |-- mouse_phenotypes: list[MousePhenotype] = []
 |        |-- phenotype_id: str = ""
 |        |-- phenotype_label: str = ""
 |        |-- phenotype_categories: list[str] = []
 |        +-- biological_models: list[BiologicalModel] = []
 |-- safety_liabilities: list[SafetyLiability] = []
 |        |-- event: str | None = None
 |        |-- event_id: str | None = None
 |        |-- effects: list[SafetyEffect] = []
 |        |-- datasource: str | None = None
 |        |-- literature: str | None = None
 |        +-- url: str | None = None
 +-- genetic_constraint: list[GeneticConstraint] = []
          |-- constraint_type: str = ""
          |-- oe: float | None = None
          |-- oe_lower: float | None = None
          |-- oe_upper: float | None = None
          |-- score: float | None = None
          +-- upper_bin: int | None = None
```

**Key design**: `DrugTarget` (inside `DrugData`) only holds lightweight reference data. To
get the full target data, call `get_target_data(target_id)` separately. This allows targets
to be cached independently and shared across drugs.

### EvidenceRecord

Pulled from Open Targets' `target.evidences(efoIds: [...])` endpoint. Persisted in a
per-target JSON file (separate from the `cache/<ns>/<sha>.json` layout) via
`_save_target_evidences`.

```
EvidenceRecord
 |-- disease_id: str = ""
 |-- datatype_id: str = ""                    # genetic_association, animal_model, etc.
 |-- score: float | None = None
 |-- direction_on_target: str | None = None   # GoF / LoF
 |-- direction_on_trait: str | None = None    # risk / protect
 +-- variant_functional_consequence: VariantFunctionalConsequence | None = None
          |-- id: str = ""
          +-- label: str = ""
```

### RichDrugData

```
RichDrugData
 |-- drug: DrugData | None = None
 +-- targets: list[TargetData] = []
```

Returned by `get_rich_drug_data()`. `DrugProfile` (in `models/model_drug_profile.py`) is a
flat LLM-facing projection.

### DiseaseSynonyms

```
DiseaseSynonyms
 |-- disease_id: str = ""           # EFO/MONDO identifier
 |-- disease_name: str = ""
 |-- parent_names: list[str] = []
 |-- exact: list[str] = []          # hasExactSynonym
 |-- related: list[str] = []        # hasRelatedSynonym
 |-- narrow: list[str] = []         # hasNarrowSynonym
 +-- broad: list[str] = []          # hasBroadSynonym

 Property: all_synonyms -> exact + related + parent_names (excludes broad and narrow)
```

---

## Helper Properties on DrugData

```python
drug.approved_disease_ids      # set[str] — disease IDs with max_clinical_stage == "APPROVAL"
drug.investigated_disease_ids  # set[str] — all disease IDs being investigated
```

---

## Three Disease Links

| Path | What it answers |
|------|-----------------|
| `drug.indications` | What diseases is **this drug** being tested for? |
| `target.associations` | What diseases is **this target** linked to (by any evidence)? |
| `target.drug_summaries[*].diseases` | What **other drugs** target this protein, and for what diseases? |

---

## OpenTargetsClient API

```python
async with OpenTargetsClient() as client:
    # All drug accessors take a ChEMBL ID, not a drug name.
    # Resolve a name → ChEMBL ID via data_sources.chembl.resolve_drug_name(name).

    # Drug-level
    rich = await client.get_rich_drug_data("CHEMBL1201496")
    drug = await client.get_drug("CHEMBL1201496")
    indications = await client.get_drug_indications("CHEMBL1201496")
    competitors = await client.get_drug_competitors("CHEMBL1201496")
    target_competitors = await client.get_drug_target_competitors("CHEMBL1201496")

    # Target-level (filtered against open_targets_association_min_score from Settings)
    target = await client.get_target_data("ENSG00000112164")
    associations = await client.get_target_data_associations(target_id)
    pathways = await client.get_target_data_pathways(target_id)
    interactions = await client.get_target_data_interactions(target_id)
    known_drugs = await client.get_target_data_drug_summaries(target_id)
    expressions = await client.get_target_data_tissue_expression(target_id)
    phenotypes = await client.get_target_data_mouse_phenotypes(target_id)
    safety = await client.get_target_data_safety_liabilities(target_id)
    constraints = await client.get_target_data_genetic_constraints(target_id)
    evidences = await client.get_target_evidences(target_id, efo_ids=[...])

    # Disease-level
    disease_drugs = await client.get_disease_drugs("EFO_0003847")
    synonyms = await client.get_disease_synonyms("non-alcoholic steatohepatitis")
    disease_id = await client.resolve_disease_id("non-alcoholic steatohepatitis")
```

### GraphQL Queries

Queries hit `https://api.platform.opentargets.org/api/v4/graphql`:

| Query | Purpose | Variables |
|-------|---------|-----------|
| `SEARCH_QUERY` | Resolve drug name to ChEMBL ID | `q: str` |
| `DRUG_QUERY` | Fetch full drug data | `id: str` (ChEMBL ID) |
| `TARGET_QUERY` | Fetch full target data | `id: str` (Ensembl ID) |
| `ASSOCIATIONS_PAGE_QUERY` | Paginate associations (if > 500) | `id, index, size` |

Associations are paginated through `_paginate_associations` once `len(associations) >= 500`.

### Interaction Type Mapping

`interaction_type` is derived from `source_database` via `INTERACTION_TYPE_MAP`:

| source_database | interaction_type |
|-----------------|------------------|
| `intact` | `physical` |
| `string` | `functional` |
| `signor` | `signalling` |
| `reactome` | `enzymatic` |

---

## ClinicalTrials.gov Data Structure

The `ClinicalTrialsClient` exposes five public methods:

| Method | Purpose | Returns |
|--------|---------|---------|
| `get_trial(nct_id)` | Fetch a single trial by NCT ID | `Trial` |
| `search_trials(drug, mesh_term, date_before=None)` | All-status pair query: count + top-50 exemplars | `SearchTrialsResult` |
| `get_completed_trials(drug, mesh_term, date_before=None)` | COMPLETED pair query | `CompletedTrialsResult` |
| `get_terminated_trials(drug, mesh_term, date_before=None)` | TERMINATED pair query | `TerminatedTrialsResult` |
| `get_landscape(mesh_term, date_before=None, top_n=50)` | Competitive landscape for an indication | `IndicationLandscape` |

### Server-side MeSH filtering

Each indication-scoped method takes a `mesh_term` (the MeSH preferred heading) rather than a
raw disease string. The indication is filtered server-side via CT.gov's
`AREA[ConditionMeshTerm]"<mesh_term>"` syntax — a precise descriptor match with no free-text
noise; the drug side stays free-text via `query.intr`. The agent tool layer
(`agents/clinical_trials/clinical_trials_tools.py`) resolves the indication → `(mesh_id,
mesh_term)` via `services.disease_helper.resolve_mesh_id` (NCBI MeSH ATM lookup) and passes the
resolved term on every call; when resolution fails it returns an empty result.

> Known bug (flagged in `clinical_trials.py`): `AREA[ConditionMeshTerm]` also matches trials
> whose MeSH *ancestors* include the term, so e.g. "Hypertension" can still pull in pulmonary
> hypertension trials. The intended fix (post-filter on direct `mesh_conditions` keyed on MeSH
> ID) is not yet implemented.

### Trial

Core trial record parsed from ClinicalTrials.gov API `protocolSection` (plus
`derivedSection.conditionBrowseModule` for MeSH terms):

```
Trial
 |-- nct_id: str = ""
 |-- title: str = ""
 |-- brief_summary: str | None = None
 |-- phase: str = ""                         # "Phase 1", "Phase 2", "Phase 1/Phase 2", etc.
 |-- overall_status: str = ""                # "Recruiting", "Completed", "Terminated", etc.
 |-- why_stopped: str | None = None          # only for Terminated/Withdrawn/Suspended
 |-- indications: list[str] = []
 |-- mesh_conditions: list[MeshTerm] = []    # from conditionBrowseModule.meshes
 |-- mesh_ancestors: list[MeshTerm] = []     # from conditionBrowseModule.ancestors
 |        |-- id: str = ""                       # MeSH D-number
 |        +-- term: str = ""
 |-- interventions: list[Intervention] = []
 |        |-- intervention_type: str = ""       # "Drug", "Biological", "Device", etc.
 |        |-- intervention_name: str = ""
 |        +-- description: str | None = None
 |-- sponsor: str = ""
 |-- enrollment: int | None = None
 |-- start_date: str | None = None
 |-- completion_date: str | None = None
 |-- primary_outcomes: list[PrimaryOutcome] = []
 |        |-- measure: str = ""
 |        +-- time_frame: str | None = None
 +-- references: list[str] = []                  # PMIDs
```

### Pair-scoped result models

These follow a consistent count + top-50 exemplars pattern. `total_count` is the exact
number of matching trials (via `countTotal`); `trials` is the top 50 by enrollment for the
agent to inspect. Stop-category classification is derived on read at the tool layer (no
separate field stored).

```
SearchTrialsResult
 |-- total_count: int = 0
 |-- by_status: dict[str, int] = {}    # RECRUITING, ACTIVE_NOT_RECRUITING, WITHDRAWN, UNKNOWN
 +-- trials: list[Trial] = []

CompletedTrialsResult
 |-- total_count: int = 0
 +-- trials: list[Trial] = []          # phase information read off each Trial

TerminatedTrialsResult
 |-- total_count: int = 0
 +-- trials: list[Trial] = []          # each carries `why_stopped` text
```

### IndicationLandscape

Competitive landscape for an indication — all drug/biologic trials grouped by sponsor +
drug. Vaccines are excluded (matched by `VACCINE_NAME_KEYWORDS`) since they are not
mechanism competitors.

```
IndicationLandscape
 |-- total_trial_count: int | None = None
 |-- competitors: list[CompetitorEntry] = []   # ranked by max_phase desc, then most_recent_start desc
 |        |-- sponsor: str = ""
 |        |-- drug_name: str = ""
 |        |-- drug_type: str | None = None
 |        |-- max_phase: str = ""
 |        |-- trial_count: int = 0
 |        |-- statuses: set[str] = set()
 |        |-- total_enrollment: int = 0
 |        +-- most_recent_start: str | None = None
 |-- phase_distribution: dict[str, int] = {}
 +-- recent_starts: list[RecentStart] = []     # trials starting >= CLINICAL_TRIALS_RECENT_START_YEAR
          |-- nct_id: str = ""
          |-- sponsor: str = ""
          |-- drug: str = ""
          +-- phase: str = ""
```

`get_landscape` filters to drug/biologic interventions only:
1. Fetches up to `clinical_trials_landscape_max_trials` trials for the indication, sorted by
   start date descending.
2. Skips trials without a Drug or Biological intervention type.
3. Excludes biologics whose name matches `VACCINE_NAME_KEYWORDS`.
4. Groups remaining by sponsor + drug.
5. Ranks by `max_phase` desc, then `most_recent_start` desc.
6. Returns top N competitors.

### ApprovalCheck

Result of an FDA-label lookup for a drug × indication pair. Computed in
`services.approval_check` and surfaced as a tool by the clinical-trials agent.

```
ApprovalCheck
 |-- is_approved: bool = False
 |-- label_found: bool = False
 |-- matched_indication: str | None = None
 +-- drug_names_checked: list[str] = []
```

`is_approved` is True when the indication appears on a current FDA label for any known name
of the drug. `label_found` distinguishes "no label exists for this drug in openFDA" (e.g.
withdrawn drugs) from "label exists but indication not present".

---

## Approval-Relationship Labeling

A repurposing candidate's relationship to the drug's existing FDA approvals is decided **once,
upstream, from the FDA label** — never authored by an LLM blurb — and threaded down to every layer
that consumes it. The contract is a four-way `ApprovalLabel` (`services/approval_check.py`):

| Label | Meaning | Disposition |
|-------|---------|-------------|
| `approved` | Drug is already approved for this exact indication | **Dropped** upstream; never reaches findings |
| `combination_only` | Approved for this disease ONLY as a fixed-dose combination product, never as monotherapy | **Demoted** — not a monotherapy repurposing lead |
| `contaminated` | A real, broader repurposing target whose trial/literature counts are polluted by an approved narrower subset | **Kept and ranked**, but trial tables suppressed |
| `none` | No relationship to an approved indication | **Kept**, clean signal |

`get_fda_approved_disease_mapping(drug_name, candidate_diseases)` returns one `ApprovalLabel` per
candidate via a two-tier lookup:

1. **Curated short-circuit** — exact, case-sensitive match against the drug's curated lists in
   `constants.py` (`CURATED_FDA_APPROVED_CANDIDATES`, `CURATED_FDA_COMBINATION_ONLY_CANDIDATES`,
   `CURATED_FDA_CONTAMINATED_CANDIDATES`, `CURATED_FDA_REJECTED_CANDIDATES`). Skips the LLM. Used
   where the bare disease term mis-leads the label-grounded LLM (e.g. `sotorasib` × colorectal
   cancer is approved only for KRAS-G12C mCRC → `contaminated`; `bupropion` × obesity is approved
   only as Contrave → `combination_only`).
2. **LLM fallback** — remaining candidates: the drug is expanded to all ChEMBL aliases, all matching
   openFDA labels are fetched, and the candidates are batched into one label-grounded LLM call.

Labels are cached per-drug (`cache/fda_drug_disease_approval/`, one file per drug, per-disease TTL).
`_coerce_label` validates every value from both the LLM-parse and the cache loader — an invalid or
legacy (bool-shaped) value is **skipped, never silently coerced**. Any failure (ChEMBL, FDA fetch,
LLM parse) defaults a candidate to `none` so a failure can never *drop* a candidate.

### Threading the label down

- **Supervisor level.** `get_fda_approved_disease_mapping` runs inside `supervisor_tools.py` during
  candidate surfacing. `approved` candidates are dropped immediately. `contaminated` /
  `combination_only` labels are buffered in a `{lowercase_disease: label}` map exposed via
  `get_approval_labels()`. After the ReAct loop, `run_supervisor_agent` sets
  `CandidateFindings.approval_relationship` from this map (matching on both the canonical and the
  finding's own disease string) — **not** from any LLM-authored field. The approval relationship was
  removed from `CandidateBlurb` entirely; the model never decides it.
- **Trial level.** A `contaminated` candidate has its trial tables suppressed in
  `format_report.py` (the verbatim total counts stay, but the example trial list is replaced with
  "overlaps an approved related indication and cannot be cleanly separated").

### Approval-aware relevance at the trial & literature level

The drug's **approved-indication list** (collected in the supervisor store as
`entry["approved_indications"]`, fully seeded before fan-out) is threaded into both per-trial and
per-paper relevance checks so an approved sub-indication's evidence can no longer count toward the
broader repurposing candidate. This is necessary because CT.gov's `AREA[ConditionMeshTerm]` filter
matches via MeSH **ancestors**, so a broad umbrella query (e.g. "mood disorder") pulls in trials
whose direct condition is an approved child (e.g. Seasonal Affective Disorder, a child of Mood
Disorders).

- the **clinical-trials relevance gate** (`prompts/clinical_trials.txt`) — an ordered first-match
  test: TEST 1 a trial whose condition IS an approved indication (or a narrower form of one) →
  CONTAMINATION, overriding the "narrower subtype rolls up" rule; TEST 2 distinct-disease /
  wrong-drug → CONTAMINATION; TEST 3 otherwise → RELEVANT. Replaced the former
  `CURATED_CONTAMINATED_NCTS` hardcode with a general rule. `approved_indications` is threaded down
  via `analyze_clinical_trials` → `run_clinical_trials_agent`.
- the **literature strength judge** — merged into `RetrievalService.synthesize`, which now takes an
  `approved_indications` argument. A paper studying the drug for an approved sub-indication is
  marked `evidence_basis="approved"` (strength capped at weak/none, direction none) on the
  `EvidenceSummary` and excluded from the broader candidate's signal. (There is no standalone
  `literature_strength.py` / `judge_literature_strength`.)

Deliberately **not** excluded (these roll up as relevant): *siblings* of an approved indication
(T1DM vs approved T2DM) and trials/papers *broader* than a minority-biomarker approval (all-comers
NSCLC vs approved EGFR-mutated NSCLC). This is the accuracy-over-coverage principle applied at the
evidence layer: an approved sub-indication's evidence must not prop up the broader repurposing
candidate, but genuinely-broader or sibling evidence still counts.

---

## PubMed Data Structure

The `PubMedClient` provides access to scientific literature via NCBI E-utilities.

### PubmedAbstract

```
PubmedAbstract
 |-- pmid: str = ""
 |-- title: str = ""
 |-- abstract: str | None = None
 |-- authors: list[str] = []
 |-- journal: str | None = None
 |-- pub_date: str | None = None     # YYYY or YYYY-MM or YYYY-MM-DD
 |-- mesh_terms: list[str] = []
 +-- keywords: list[str] = []
```

### PubMedClient methods

| Method | Description | Returns |
|--------|-------------|---------|
| `search(query, max_results, date_before)` | Search for PMIDs (cached under `pubmed_search`) | `list[str]` |
| `get_count(query, date_before)` | Count results without fetching | `int` |
| `fetch_abstracts(pmids, batch_size)` | Fetch abstract details by PMID | `list[PubmedAbstract]` |

PMIDs are persisted to Postgres (`sqlalchemy.pubmed_abstracts.PubmedAbstracts`, with
pgvector embeddings) by `RetrievalService.fetch_and_cache` so the literature agent can run
semantic search against stored abstracts.

### EvidenceSummary

The synthesized literature output produced by `RetrievalService.synthesize` (in
`models/model_evidence_summary.py`). `strength` grades evidence quantity/quality;
`direction` grades which way it points — the two are independent. The PMID buckets are built
**in code** (`retrieval.py`) from the per-PMID verdict map the synthesize call emits, not by the
LLM directly.

```
EvidenceSummary
 |-- summary: str = ""
 |-- study_count: int = 0
 |-- strength: "strong" | "moderate" | "weak" | "none" = "none"
 |-- direction: "supports" | "contradicts" | "mixed" | "none" = "none"
 |-- evidence_basis: "drug_specific" | "approved" | "class_level" | "none" = "none"
 |        # "class_level" → relevant RCTs are sibling-drug only (strength forced off "strong")
 |        # "approved"    → only this-drug evidence studies an APPROVED sub-indication (strength forced none)
 |-- is_observational: bool | None = None   # True if ≥1 RCT; False if purely observational; None = undetermined
 |-- key_findings: list[str] = []
 |-- supporting_pmids: list[str] = []        # supporting + mixed
 |-- contradicting_pmids: list[str] = []     # contradicting + mixed
 |-- relevant_pmids: list[str] = []          # graded this-drug-this-disease evidence
 |-- contaminated_pmids: list[str] = []      # excluded (wrong drug/disease or approved sub-indication)
 +-- neutral_pmids: list[str] = []           # relevant but no efficacy result (PK/safety/mechanism)
```

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
    small_llm_model: str = "claude-haiku-4-5-20251001"
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
    clinical_trials_landscape_max_trials: int
    clinical_trials_cap: int
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
