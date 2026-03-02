# Architecture

## Project Overview

IndicationScout is an agentic drug repurposing system. A drug name goes in; coordinated AI agents query multiple biomedical data sources (Open Targets, ClinicalTrials.gov, PubMed, ChEMBL) and produce a repurposing report identifying candidate indications not yet approved or explored.

### Directory Structure

```
indication_scout/
├── src/indication_scout/          # Main source code
│   ├── __init__.py                # Package initialization
│   ├── config.py                  # Application settings (pydantic-settings)
│   ├── constants.py               # URLs, timeouts, lookup maps
│   ├── agents/                    # AI agent layer (all stubs)
│   ├── api/                       # FastAPI application (/health only)
│   ├── data_sources/              # Async API clients (OpenTargets, ClinicalTrials, PubMed, ChEMBL, DrugBank)
│   ├── db/                        # SQLAlchemy session factory
│   ├── helpers/                   # Drug name normalization
│   ├── models/                    # Pydantic data contracts
│   ├── prompts/                   # LLM prompt templates
│   ├── runners/                   # Standalone runner scripts
│   ├── scripts/                   # Session management, exploratory pipelines
│   ├── services/                  # Business logic (LLM, embeddings, disease normalization, retrieval)
│   ├── sqlalchemy/                # ORM models (pubmed_abstracts with pgvector)
│   └── utils/                     # Shared file-based cache utility
├── tests/                         # Test suite
│   ├── unit/                      # Unit tests (no network)
│   ├── integration/               # Integration tests (hits real APIs)
│   └── conftest.py               # Shared fixtures
├── docs/                          # Documentation
├── _cache/                        # Disk cache for API responses (SHA-256-keyed JSON, 5-day TTL)
└── pyproject.toml                 # Project metadata & dependencies
```

### Current State

| Component | Status | Description |
|-----------|--------|-------------|
| Data Sources | **Complete** | OpenTargetsClient, ClinicalTrialsClient, PubMedClient, ChEMBLClient; DrugBankClient is a stub |
| Data Models | **Complete** | Pydantic models for all data contracts (Open Targets, ClinicalTrials, PubMed, ChEMBL, DrugProfile) |
| BaseClient | **Complete** | Retry with exponential backoff, session management via aiohttp |
| File Cache | **Complete** | Shared `utils/cache.py` used by all clients and services (`_cache/` dir, SHA-256 keys, 5-day TTL) |
| Services | **Partial** | `llm.py`, `embeddings.py`, `disease_normalizer.py`, `pubmed_query.py` complete; `retrieval.py` partial (expand/extract done, fetch_and_cache/semantic_search/synthesize stubbed) |
| Agents | Stub | Orchestrator, LiteratureAgent, ClinicalTrialsAgent, MechanismAgent, SafetyAgent -- all `run()` raise `NotImplementedError` |
| API | Minimal | FastAPI with `/health` endpoint only; routes/ and schemas/ are empty |
| CLI | Referenced | Defined in pyproject.toml but CLI module does not exist |

---

## BaseClient Infrastructure

All data source clients inherit from `BaseClient`, which provides common infrastructure for reliable API communication.

### Core Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              BaseClient                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Request Methods                                                             │
│  ├── _request()   - Low-level HTTP with retry                               │
│  ├── _graphql()   - GraphQL POST requests                                   │
│  ├── _rest_get()  - REST GET requests                                       │
│  └── _rest_get_xml() - REST GET returning XML text                          │
│                                                                              │
│  Retry Logic                                                                 │
│  └── Exponential backoff (1s, 2s, 4s, capped at 30s), max 3 retries        │
│  └── Retries on: 429, 500, 502, 503, 504                                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Configuration

```python
BaseClient(
    timeout=30.0,        # DEFAULT_TIMEOUT
    max_retries=3,       # DEFAULT_MAX_RETRIES
)
```

---

## Open Targets Data Structure

The `OpenTargetsClient` provides three primary entry points: `get_drug()` for drug data, `get_target_data()` for target data, and `get_rich_drug_data()` which combines both. Results are cached independently per namespace.

### DrugData

```
DrugData
 |-- chembl_id: str = ""
 |-- name: str = ""
 |-- synonyms: list[str] = []
 |-- trade_names: list[str] = []
 |-- drug_type: str = ""
 |-- is_approved: bool | None = None
 |-- max_clinical_phase: float | None = None
 |-- year_first_approved: int | None = None
 |-- warnings: list[DrugWarning] = []
 |        |-- warning_type: str = ""
 |        |-- description: str | None = None
 |        |-- toxicity_class: str | None = None
 |        |-- country: str | None = None
 |        |-- year: int | None = None
 |        +-- efo_id: str | None = None
 |-- indications: list[Indication] = []
 |        |-- disease_id: str = ""
 |        |-- disease_name: str = ""
 |        |-- max_phase: float | None = None
 |        +-- references: list[dict] = []
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
 |-- associations: list[Association] = []
 |        |-- disease_id: str = ""
 |        |-- disease_name: str = ""
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
 |        +-- interaction_type: str | None = None
 |-- drug_summaries: list[DrugSummary] = []
 |        |-- drug_id: str = ""
 |        |-- drug_name: str = ""
 |        |-- disease_id: str = ""
 |        |-- disease_name: str = ""
 |        |-- phase: float | None = None
 |        |-- status: str | None = None
 |        |-- mechanism_of_action: str = ""
 |        +-- clinical_trial_ids: list[str] = []
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
 |                          |-- name: str = ""
 |                          |-- level: int | None = None
 |                          +-- reliability: bool | None = None
 |-- mouse_phenotypes: list[MousePhenotype] = []
 |        |-- phenotype_id: str = ""
 |        |-- phenotype_label: str = ""
 |        |-- phenotype_categories: list[str] = []
 |        +-- biological_models: list[BiologicalModel] = []
 |                 |-- allelic_composition: str = ""
 |                 |-- genetic_background: str = ""
 |                 |-- literature: list[str] = []
 |                 +-- model_id: str = ""
 |-- safety_liabilities: list[SafetyLiability] = []
 |        |-- event: str | None = None
 |        |-- event_id: str | None = None
 |        |-- effects: list[SafetyEffect] = []
 |        |        |-- direction: str = ""
 |        |        +-- dosing: str | None = None
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

**Key design**: `DrugTarget` (inside `DrugData`) only holds lightweight reference data. To get the full target data, call `get_target_data(target_id)` separately. This allows targets to be cached independently and shared across drugs.

### RichDrugData

```
RichDrugData
 |-- drug: DrugData | None = None   # Full drug data
 +-- targets: list[TargetData] = [] # Full target data for each of drug.targets
```

Returned by `get_rich_drug_data()`. Provides everything Open Targets knows about a drug and all its targets in a single object. `DrugProfile` (in `models/model_drug_profile.py`) is the flat LLM-facing projection of `RichDrugData`.

### DiseaseSynonyms

```
DiseaseSynonyms
 |-- disease_id: str = ""           # EFO/MONDO identifier
 |-- disease_name: str = ""         # Canonical disease name
 |-- parent_names: list[str] = []   # Parent disease names in the ontology
 |-- exact: list[str] = []          # Exact synonyms (hasExactSynonym)
 |-- related: list[str] = []        # Related synonyms (hasRelatedSynonym)
 |-- narrow: list[str] = []         # Narrow synonyms (hasNarrowSynonym)
 +-- broad: list[str] = []          # Broad synonyms (hasBroadSynonym)

 Property: all_synonyms -> exact + related + parent_names (excludes broad and narrow)
```

---

## Helper Properties on DrugData

```python
drug.approved_disease_ids      # set[str] - Disease IDs with phase 4+
drug.investigated_disease_ids  # set[str] - All disease IDs being pursued
```

---

## Three Disease Links

| Path | What it answers |
|------|-----------------|
| `drug.indications` | What diseases is **this drug** being tested for? |
| `target.associations` | What diseases is **this target** linked to (by any evidence)? |
| `target.drug_summaries` | What **other drugs** target this protein, and for what diseases? |

---

## Finding Whitespace Indications

The system finds repurposing opportunities by:

1. Get `target.associations` - diseases linked to the target
2. Filter out `drug.indications` - diseases already being pursued
3. What remains = potential new indications

Example: If GLP1R has a high association score with NASH, but semaglutide's `indications` list doesn't include NASH, that's a whitespace opportunity.

---

## OpenTargetsClient Implementation

### Data Flow

```
                          ┌─────────────────────────────────────────────────┐
                          │             OpenTargetsClient                   │
                          │                                                 │
  get_drug("semaglutide") │   1. _resolve_drug_name() ──> SEARCH_QUERY     │
           │              │          └─> returns ChEMBL ID                  │
           ▼              │                                                 │
   ┌───────────────┐      │   2. Check disk cache (_cache/<hash>.json)     │
   │  Drug Name    │──────│          └─> if hit, deserialize & return      │
   │  "semaglutide"│      │                                                 │
   └───────────────┘      │   3. _fetch_drug() ──> DRUG_QUERY              │
                          │          └─> parse response ──> DrugData       │
                          │          └─> store in disk cache               │
                          └─────────────────────────────────────────────────┘

get_target_data("ENSG...") follows the same pattern with TARGET_QUERY
```

### Disk Cache

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Disk Cache                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Location: _cache/<sha256>.json  (shared across sessions)           │
│                                                                      │
│  Key: SHA-256 of {"ns": namespace, **params} (JSON, sorted keys)   │
│  Namespaces:                                                        │
│  ├── "drug"              (OpenTargets)  key: chembl_id              │
│  ├── "target"            (OpenTargets)  key: target_id              │
│  ├── "disease_drugs"     (OpenTargets)  key: disease_id             │
│  ├── "disease_synonyms"  (OpenTargets)  key: disease_id             │
│  ├── "pubmed_search"     (PubMed)       key: query+max+date        │
│  ├── "atc_description"   (ChEMBL)       key: atc_code              │
│  ├── "disease_norm"      (normalizer)   key: raw_term              │
│  ├── "pubmed_count"      (normalizer)   key: query                 │
│  ├── "organ_term"        (retrieval)    key: disease_name           │
│  └── "expand_search_terms" (retrieval)  key: drug+disease          │
│                                                                      │
│  Entry: {"data": <model_dump>, "cached_at": <iso>, "ttl": <secs>}  │
│                                                                      │
│  TTL: 5 days (CACHE_TTL = 5 * 86400)                                │
│  Expiry: checked on read; expired/corrupt entries auto-deleted      │
│  Disable: pass cache_dir=None to OpenTargetsClient()                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### GraphQL Queries

The client uses four GraphQL queries against `https://api.platform.opentargets.org/api/v4/graphql`:

| Query | Purpose | Variables |
|-------|---------|-----------|
| `SEARCH_QUERY` | Resolve drug name to ChEMBL ID | `q: str` |
| `DRUG_QUERY` | Fetch full drug data | `id: str` (ChEMBL ID) |
| `TARGET_QUERY` | Fetch full target data | `id: str` (Ensembl ID) |
| `ASSOCIATIONS_PAGE_QUERY` | Paginate associations (if > 500) | `id, index, size` |

### Parsing Pipeline

Raw GraphQL responses are parsed into Pydantic models via dedicated parser methods:

```
GraphQL Response                    Parser Method                  Pydantic Model
─────────────────                   ─────────────                  ──────────────
drug { ... }                   ──>  _parse_drug_data()        ──>  DrugData
  └── mechanismsOfAction       ──>    (inline)                ──>    DrugTarget
  └── indications              ──>    (inline)                ──>    Indication
  └── drugWarnings             ──>    (inline)                ──>    DrugWarning
  └── adverseEvents            ──>  _parse_adverse_event()    ──>    AdverseEvent

target { ... }                 ──>  _parse_target_data()      ──>  TargetData
  └── associatedDiseases       ──>  _parse_association()      ──>    Association
  └── pathways                 ──>  _parse_pathway()          ──>    Pathway
  └── interactions             ──>  _parse_interaction()      ──>    Interaction
  └── knownDrugs               ──>  _parse_drug_summary()     ──>    DrugSummary
  └── expressions              ──>  _parse_expression()       ──>    TissueExpression
  └── mousePhenotypes          ──>  _parse_phenotype()        ──>    MousePhenotype
  └── safetyLiabilities        ──>  _parse_safety_liability() ──>    SafetyLiability
  └── geneticConstraint        ──>  _parse_constraint()       ──>    GeneticConstraint
```

### Interaction Type Mapping

The `interaction_type` field is derived from `source_database` using `INTERACTION_TYPE_MAP`:

| source_database | interaction_type |
|-----------------|------------------|
| `intact` | `physical` |
| `string` | `functional` |
| `signor` | `signalling` |
| `reactome` | `enzymatic` |

### Pagination

Associations are paginated when count exceeds 500:

```python
# In _fetch_target():
if len(target_data.associations) >= 500:
    target_data.associations = await self._paginate_associations(target_id)

# _paginate_associations() loops through pages of 500 until all fetched
```

---

## Client API

```python
async with OpenTargetsClient() as client:
    # Fetch drug + all target data in one call
    rich = await client.get_rich_drug_data("semaglutide")

    # Or fetch drug data alone (cached)
    drug = await client.get_drug("semaglutide")

    # Fetch target data separately (cached independently)
    target = await client.get_target_data("ENSG00000112164")

    # Accessor methods for specific target data
    associations = await client.get_target_data_associations(target_id, min_score=0.1)
    pathways = await client.get_target_data_pathways(target_id)
    interactions = await client.get_target_data_interactions(target_id)
    known_drugs = await client.get_target_data_drug_summaries(target_id)
    expressions = await client.get_target_data_tissue_expression(target_id)
    phenotypes = await client.get_target_data_mouse_phenotypes(target_id)
    safety = await client.get_target_data_safety_liabilities(target_id)
    constraints = await client.get_target_data_genetic_constraints(target_id)

    # Drug-specific accessors
    indications = await client.get_drug_indications("semaglutide")
    competitors = await client.get_drug_competitors("semaglutide")
    target_competitors = await client.get_drug_target_competitors("semaglutide")

    # Disease-specific accessors
    disease_drugs = await client.get_disease_drugs("EFO_0003847")
    synonyms = await client.get_disease_synonyms("non-alcoholic steatohepatitis")
```

---

## ClinicalTrials.gov Data Structure

The `ClinicalTrialsClient` provides four public methods for clinical trial data:

| Method | Purpose | Returns |
|--------|---------|---------|
| `search_trials(drug, condition)` | Find trials for a drug-condition pair | `list[Trial]` |
| `detect_whitespace(drug, condition)` | Is this pair unexplored? | `WhitespaceResult` |
| `get_landscape(condition)` | Competitive landscape for a condition | `ConditionLandscape` |
| `get_terminated(query)` | Failed trials for a drug/condition | `list[TerminatedTrial]` |

### Trial

Core trial record parsed from ClinicalTrials.gov API `protocolSection`:

```
Trial
 |-- nct_id: str                    # NCT identifier (e.g. "NCT04375669")
 |-- title: str                     # Brief title of the study
 |-- brief_summary: str | None      # Study description
 |-- phase: str                     # "Phase 1", "Phase 2", "Phase 1/Phase 2", etc.
 |-- overall_status: str            # "Recruiting", "Completed", "Terminated", etc.
 |-- why_stopped: str | None        # Free text reason (only for Terminated/Withdrawn/Suspended)
 |-- conditions: list[str]          # Disease/conditions being studied
 |-- interventions: list[Intervention]
 |        |-- intervention_type: str    # "Drug", "Biological", "Device", "Procedure", etc.
 |        |-- intervention_name: str    # e.g. "Semaglutide"
 |        +-- description: str | None   # Dosing, administration details
 |-- sponsor: str                   # Lead sponsor organization (e.g. "Novo Nordisk")
 |-- collaborators: list[str]       # Partner organizations
 |-- enrollment: int | None         # Number of participants
 |-- start_date: str | None         # Study start date
 |-- completion_date: str | None    # Primary completion date
 |-- study_type: str                # "Interventional" or "Observational"
 |-- primary_outcomes: list[PrimaryOutcome]
 |        |-- measure: str              # What's being measured
 |        +-- time_frame: str | None    # e.g. "72 weeks"
 |-- results_posted: bool | None    # Whether results are available (None if unknown)
 +-- references: list[str]          # PubMed IDs (PMIDs)
```

### WhitespaceResult

Result of whitespace detection — is this drug-condition pair unexplored?

```
WhitespaceResult
 |-- is_whitespace: bool | None     # True if no exact matches found
 |-- exact_match_count: int | None  # Trials with both drug AND condition
 |-- drug_only_trials: int | None   # Trials with drug (any condition)
 |-- condition_only_trials: int | None  # Trials with condition (any drug)
 +-- condition_drugs: list[ConditionDrug]  # Other drugs tested for this condition (up to 20)
          |-- nct_id: str
          |-- drug_name: str
          |-- condition: str
          |-- phase: str
          +-- status: str
```

### ConditionLandscape

Competitive landscape for a condition — all drug/biologic trials grouped by sponsor + drug:

```
ConditionLandscape
 |-- total_trial_count: int | None  # All trials fetched for condition
 |-- competitors: list[CompetitorEntry]   # Ranked by phase, then enrollment (top_n)
 |        |-- sponsor: str              # Lead sponsor organization
 |        |-- drug_name: str            # Primary drug intervention
 |        |-- drug_type: str | None     # "Drug" or "Biological"
 |        |-- max_phase: str            # Highest phase reached
 |        |-- trial_count: int | None    # Number of trials for this sponsor+drug
 |        |-- statuses: set[str]        # All statuses seen (Recruiting, Completed, etc.)
 |        |-- total_enrollment: int | None  # Sum of enrollment across trials
 |        +-- most_recent_start: str | None  # Latest start date
 |-- phase_distribution: dict[str, int]  # Count of trials per phase
 +-- recent_starts: list[RecentStart]  # Trials started in last 2 years
          |-- nct_id: str = ""
          |-- sponsor: str = ""
          |-- drug: str = ""
          +-- phase: str = ""
```

### TerminatedTrial

A terminated, withdrawn, or suspended trial with stop classification:

```
TerminatedTrial
 |-- nct_id: str
 |-- title: str
 |-- drug_name: str | None          # Primary drug intervention
 |-- condition: str | None          # First condition listed
 |-- phase: str | None
 |-- why_stopped: str | None        # Free text reason from sponsor
 |-- stop_category: str | None      # Classified: safety, efficacy, business, enrollment, other, unknown
 |-- enrollment: int | None
 |-- sponsor: str | None
 |-- start_date: str | None
 |-- termination_date: str | None
 +-- references: list[str]          # PMIDs
```

### Stop Category Classification

The `stop_category` is derived from `why_stopped` using keyword matching:

| Keywords | Category |
|----------|----------|
| efficacy, futility, lack of efficacy, no benefit | `efficacy` |
| safety, adverse, toxicity, side effect | `safety` |
| enrollment, accrual, recruitment | `enrollment` |
| business, strategic, funding, commercial | `business` |
| (no match) | `other` |
| (no why_stopped text) | `unknown` |

### Filtering

`get_landscape` filters to drug/biologic interventions only:
1. Fetches all trials for the condition (no API-level filter)
2. Skips trials without Drug or Biological intervention type
3. Groups remaining by sponsor + drug
4. Ranks by phase (descending), then enrollment (descending)
5. Returns top N competitors (default 50)

---

## PubMed Data Structure

The `PubMedClient` provides access to scientific literature via NCBI E-utilities.

### PubmedAbstract

```
PubmedAbstract
 |-- pmid: str = ""                # PubMed ID
 |-- title: str | None = None      # Article title
 |-- abstract: str | None = None   # Abstract text (may have labelled sections)
 |-- authors: list[str] = []       # Author names ("Last, First" format)
 |-- journal: str | None = None    # Journal name
 |-- pub_date: str | None = None   # Publication date (YYYY or YYYY-MM or YYYY-MM-DD)
 |-- mesh_terms: list[str] = []    # MeSH descriptor terms
 +-- keywords: list[str] = []      # Author-supplied keywords
```

### PubMedClient Data Flow

```
Input: query (str), max_results (int)
  |
search(query, max_results, date_before)       [cached under "pubmed_search"]
  +-- REST query NCBI E-utilities
      +-- esearch.fcgi -> list[str] (PMIDs)
  |
fetch_abstracts(pmids, batch_size)
  +-- efetch.fcgi -> Parse XML -> PubmedAbstract[]
  |
Output: list[PubmedAbstract]

Alternative:
  - get_count(query) -> quick result count (no parsing)
```

### Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `search(query, max_results, date_before)` | Search for PMIDs (cached) | `list[str]` |
| `get_count(query, date_before)` | Count results without fetching | `int` |
| `fetch_abstracts(pmids, batch_size)` | Fetch abstract details by PMID | `list[PubmedAbstract]` |

---

## External Integrations

| Service | Type | Endpoint | Authentication |
|---------|------|----------|-----------------|
| Open Targets Platform | GraphQL | https://api.platform.opentargets.org/api/v4/graphql | None |
| ClinicalTrials.gov | REST v2 | https://clinicaltrials.gov/api/v2/ | None |
| PubMed/NCBI | REST (E-utilities) | https://eutils.ncbi.nlm.nih.gov/entrez/eutils/ | API key (optional) |
| ChEMBL | REST | https://www.ebi.ac.uk/chembl/api/data | None |
| Anthropic | REST | Anthropic Messages API | API key required |

---

## Configuration

Application settings via `pydantic_settings.BaseSettings` (loads from `.env`):

```python
Settings:
  database_url: str               # PostgreSQL connection string
  db_password: str                # Database password
  test_database_url: str | None   # Separate DB for integration tests
  anthropic_api_key: str          # For Claude LLM calls
  pubmed_api_key: str             # For PubMed E-utilities (optional)
  ncbi_api_key: str               # NCBI API key (optional)
  openfda_api_key: str            # OpenFDA API key (optional)
  llm_model: str                  # Default: "claude-sonnet-4-6"
  small_llm_model: str            # Default: "claude-haiku-4-5-20251001"
  embedding_model: str            # Default: "FremyCompany/BioLORD-2023"
  debug: bool                     # Debug mode
  log_level: str                  # Logging level
```

---

## Design Principles

1. **Separation of Concerns**: Data sources (clients) separate from domain logic (agents/services); agents never see raw API responses
2. **Async-First**: All I/O operations are async using aiohttp; clients are async context managers
3. **Graceful Degradation**: Retry with exponential backoff on 429/5xx; `DataSourceError` with source name and context
4. **Shared Disk Cache**: JSON files in `_cache/` with 5-day TTL, SHA-256-keyed; used by all data source clients and services via `utils/cache.py`
5. **Type Safety**: Full Pydantic validation with `coerce_nones` model validator on every external data model; Python 3.10+ type hints throughout
6. **Model-Driven**: GraphQL/REST responses parsed into typed Pydantic models; Pydantic `BaseModel` contracts at every module boundary
7. **No Fallbacks for Clinical Data**: Missing scientific/clinical values return None, never defaults; this is a clinical genomics tool