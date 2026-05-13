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
│   ├── markers.py                 # Cross-cutting markers (e.g. holdout flags)
│   ├── agents/                    # Sub-agents and supervisor (LangGraph create_react_agent)
│   │   ├── base.py                # BaseAgent ABC (legacy, unused by ReAct agents)
│   │   ├── _trial_formatting.py   # Shared trial table / phase distribution helpers
│   │   ├── supervisor/            # Top-level supervisor agent
│   │   ├── literature/            # PubMed retrieval + synthesis sub-agent
│   │   ├── clinical_trials/       # ClinicalTrials.gov sub-agent
│   │   └── mechanism/             # Open Targets mechanism sub-agent
│   ├── api/                       # FastAPI application (/health only; routes/, schemas/ stubs)
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
│   ├── prompts/                   # LLM prompt templates (.txt files)
│   ├── report/                    # `format_report` — SupervisorOutput → markdown
│   ├── runners/                   # Standalone runner scripts (pubmed_runner, rag_runner)
│   ├── services/                  # Business logic
│   │   ├── llm.py                 # Anthropic SDK wrappers (query_llm, query_small_llm)
│   │   ├── embeddings.py          # BioLORD-2023 embeddings
│   │   ├── disease_helper.py      # LLM disease normalization + MeSH descriptor resolver
│   │   ├── pubmed_query.py        # Query building
│   │   ├── retrieval.py           # RAG: drug profile, semantic search, synthesis
│   │   └── approval_check.py      # openFDA label + LLM approval extraction
│   ├── sqlalchemy/                # ORM models (pubmed_abstracts with pgvector)
│   └── utils/                     # cache.py (shared file cache), wandb_utils.py
├── tests/                         # Test suite
│   ├── unit/                      # Unit tests (no network)
│   ├── integration/               # Integration tests (hits real APIs)
│   └── conftest.py                # Shared fixtures
├── docs/                          # Documentation (findings.md is source of truth)
├── _cache/                        # Disk cache for API/LLM responses (per-namespace JSON, 5-day TTL)
└── pyproject.toml                 # Project metadata & dependencies
```

### Current State

| Component | Status | Description |
|-----------|--------|-------------|
| Data Sources | **Complete** | OpenTargetsClient, ClinicalTrialsClient, PubMedClient, ChEMBLClient, FDAClient; DrugBankClient is a stub |
| Data Models | **Complete** | Pydantic models for all data contracts (Open Targets, ClinicalTrials, PubMed, ChEMBL, DrugProfile, EvidenceSummary) |
| BaseClient | **Complete** | Retry with exponential backoff; persistent failure log via `log_data_source_failure` |
| File Cache | **Complete** | Shared `utils/cache.py` used by all clients and services (`_cache/<namespace>/<sha>.json`, 5-day TTL) |
| Services | **Complete** | `llm.py`, `embeddings.py`, `disease_helper.py`, `pubmed_query.py`, `approval_check.py`, `retrieval.py` (build_drug_profile, expand_search_terms, extract_organ_term, fetch_new_abstracts, embed_abstracts, fetch_and_cache, semantic_search, synthesize, get_drug_competitors) |
| Agents | **Complete** | Supervisor + literature, clinical_trials, mechanism sub-agents — all built on LangGraph `create_react_agent`. `BaseAgent` ABC still exists in `agents/base.py` but is unused. |
| API | Minimal | FastAPI with `/health` endpoint only; `routes/` and `schemas/` are empty packages |
| CLI | **Complete** | `scout find -d <drug> [--out-dir DIR] [--no-write] [--date-before YYYY-MM-DD]` (in `cli/cli.py`) |

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

All four agents are built using `langgraph.prebuilt.create_react_agent`. `BaseAgent` (in
`agents/base.py`) is a legacy ABC and is not used by the active ReAct-style agents.

### Supervisor (`agents/supervisor/`)

`build_supervisor_agent(llm, svc, db, date_before)` returns
`(compiled_agent, get_merged_allowlist, get_auto_findings)`. The supervisor wraps each
sub-agent as a tool and orchestrates the run. After the LangGraph loop finishes,
`run_supervisor_agent` walks the message history, canonicalises disease names against the
merged competitor + mechanism allowlist, and assembles a `SupervisorOutput`.

Tools available to the supervisor (all in `supervisor_tools.py`):

| Tool | Purpose |
|------|---------|
| `find_candidates` | Surface competitor-derived disease candidates from Open Targets |
| `analyze_mechanism` | Run the mechanism sub-agent (returns `MechanismOutput`) |
| `analyze_literature` | Run the literature sub-agent for one disease |
| `analyze_clinical_trials` | Run the clinical-trials sub-agent for one disease |
| `investigate_top_candidates` | Holdout-only: parallel fan-out over top candidates |
| `get_drug_briefing` | Read-only view of accumulated drug-level facts |
| `finalize_supervisor` | Last action; returns the supervisor's narrative summary |

When `date_before` is set, the supervisor loads `prompts/supervisor_holdout.txt` instead of
`supervisor.txt` and forwards the cutoff to the literature and clinical-trials sub-agents.
Mechanism analysis (Open Targets) is always current because there is no date-filtering API.

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
| **Literature** | `expand_search_terms`, `fetch_and_cache`, `semantic_search`, `synthesize`, `finalize_analysis` | `LiteratureOutput` |
| **Clinical Trials** | `check_fda_approval`, `search_trials`, `get_completed`, `get_terminated`, `get_landscape`, `finalize_analysis` | `ClinicalTrialsOutput` |
| **Mechanism** | `get_drug`, `get_target_associations`, `finalize_analysis` | `MechanismOutput` |

The mechanism agent additionally has `mechanism_candidates.py` (`select_top_candidates`) and
`mechanism_row_builder.py` (`build_candidate_rows`) for post-LLM candidate scoring,
filtered against an FDA-approved disease set and trimmed to `MECHANISM_TOP_CANDIDATES`.

After each sub-agent run, `run_<name>_agent` walks the message history and pulls each
tool's typed artifact off `ToolMessage.artifact`, assembling them into the typed output.

### SupervisorOutput

```
SupervisorOutput
 |-- drug_name: str
 |-- candidates: list[str]                  # Diseases in the merged allowlist
 |-- mechanism: MechanismOutput | None
 |-- findings: list[CandidateFindings]
 |        |-- disease: str
 |        |-- source: "competitor" | "mechanism" | "both"
 |        |-- literature: LiteratureOutput | None
 |        +-- clinical_trials: ClinicalTrialsOutput | None
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
│      _cache/data_source_failures.log on terminal failure.                   │
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
│  Layout: _cache/<namespace>/<sha256>.json                           │
│  Key:    SHA-256 of {"ns": namespace, **params} (JSON, sorted keys)│
│  Entry:  {"data": ..., "cached_at": <iso>, "ttl": <secs>}           │
│  TTL:    5 days (CACHE_TTL = 5 * 86400) unless overridden per-call  │
│  Expiry: checked on read; expired/corrupt entries auto-deleted      │
│                                                                      │
│  Namespaces in active use:                                           │
│  ├── drug, target, disease_drugs, competitors_raw,                  │
│  │   disease_id_resolver                       (OpenTargets)        │
│  ├── ct_terminated, ct_completed                (ClinicalTrials)    │
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
truth (ChEMBL) handles them and the result can be cached per ChEMBL ID. See
`docs/findings.md` → "ChEMBL ID is the sole drug identifier".

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
 |        |-- datatype_scores: dict[str, float] = {}
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
per-target JSON file (separate from the `_cache/<ns>/<sha>.json` layout) via
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
| `search_trials(drug, indication, target_mesh_id=None)` | All-status pair query: count + top-50 exemplars | `SearchTrialsResult` |
| `get_completed_trials(drug, indication, target_mesh_id=None)` | COMPLETED pair query | `CompletedTrialsResult` |
| `get_terminated_trials(drug, indication, target_mesh_id=None)` | TERMINATED pair query | `TerminatedTrialsResult` |
| `get_landscape(indication, target_mesh_id=None)` | Competitive landscape for an indication | `IndicationLandscape` |

### MeSH post-filtering

Every indication-filtered method accepts an optional `target_mesh_id` (a MeSH D-number,
e.g. `D006973` for hypertension). When supplied, results are post-filtered to trials whose
`mesh_conditions` or `mesh_ancestors` include that ID. This compensates for
ClinicalTrials.gov's Essie engine being recall-first — `query.cond=hypertension` returns
trials for glaucoma, portal hypertension, and pulmonary hypertension. The agent tool layer
(`agents/clinical_trials/clinical_trials_tools.py`) resolves the indication → MeSH ID via
`services.disease_helper.resolve_mesh_id` and forwards it on every call. See
`docs/findings.md` for the full rationale.

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

---

## ChEMBL & FDA

| Client | Methods |
|--------|---------|
| `ChEMBLClient` | `get_atc_description(atc_code)`, `get_molecule(chembl_id)` |
| (module-level) | `resolve_drug_name(drug_name)` → ChEMBL ID; `get_all_drug_names(chembl_id)` → list of synonyms |
| `FDAClient` | `get_label_indications(drug_name)`, `get_all_label_indications(drug_names)` |

ChEMBL IDs and drug-name lists are persisted in dedicated per-drug JSON files under
`_cache/` (separate from the namespace cache).

---

## Services

| Service | Public surface |
|---------|----------------|
| `llm.py` | `query_llm`, `query_small_llm`, `parse_llm_response`, `parse_last_json_array`, `parse_last_json_object`, `strip_markdown_fences` |
| `embeddings.py` | `embed`, `embed_async` (BioLORD-2023 via SentenceTransformer) |
| `disease_helper.py` | `llm_normalize_disease`, `llm_normalize_disease_batch`, `merge_duplicate_diseases`, `pubmed_count`, `normalize_for_pubmed`, `normalize_batch`, `resolve_mesh_id` |
| `pubmed_query.py` | `get_pubmed_query(drug_name, disease_name)` |
| `retrieval.py` | `RetrievalService` — `build_drug_profile`, `get_drug_competitors`, `fetch_new_abstracts`, `embed_abstracts`, `fetch_and_cache`, `semantic_search`, `synthesize`, `extract_organ_term`, `expand_search_terms` |
| `approval_check.py` | `get_approved_indications`, `list_approved_indications_at`, `list_approved_indications_from_labels`, `extract_approved_from_labels`, `get_all_fda_approved_diseases`, `get_fda_approved_disease_mapping` |

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
    wandb_api_key: str = ""

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
    mechanism_signal_threshold: float
    mechanism_associations_cap: int
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
```

Defined in `cli/cli.py`. Loads `.env` and `.env.constants`, normalizes the drug name,
constructs a `ChatAnthropic` LLM, builds the supervisor agent, runs it, and writes a
markdown report and a structured JSON dump under `snapshots/` (or `snapshots/holdouts/`
when `--date-before` is set).

---

## Design Principles

1. **Separation of Concerns** — Data sources (clients) separate from domain logic (agents/services); agents never see raw API responses.
2. **Async-First** — All I/O is async via aiohttp; clients are async context managers.
3. **Graceful Degradation** — Retry with exponential backoff on 429/5xx; `DataSourceError` carries source name and context; terminal failures are logged to `_cache/data_source_failures.log`.
4. **Shared Disk Cache** — JSON files in `_cache/<namespace>/` with 5-day TTL, SHA-256-keyed; used by all data source clients and services via `utils/cache.py`.
5. **Type Safety** — Full Pydantic validation with `coerce_nones` model validator on every external data model; Python 3.10+ type hints throughout.
6. **Model-Driven** — GraphQL/REST responses parsed into typed Pydantic models; Pydantic `BaseModel` contracts at every module boundary.
7. **No Fallbacks for Clinical Data** — Missing scientific/clinical values return `None` / empty structures, never defaults; this is a clinical genomics tool.
8. **Accuracy over Coverage** — Error by omission is acceptable; inaccurate output is not. Reject paths and allowlist guards are not loosened to "rescue" missing candidates.
