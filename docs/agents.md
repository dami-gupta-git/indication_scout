# IndicationScout — Agents

## Overview

IndicationScout uses a multi-agent architecture where specialist agents each own a slice of the drug repurposing analysis. A central Orchestrator coordinates them, routing data through the pipeline and assembling the final report.

All agents extend `BaseAgent` and expose a single async entry point:

```python
async def run(self, input_data: dict[str, Any]) -> dict[str, Any]
```

**Current status:** All agents are planned/stubbed. Data source clients (Open Targets, ClinicalTrials.gov, PubMed, openFDA) are fully implemented and tested.

---

## BaseAgent

**File:** [src/indication_scout/agents/base.py](../src/indication_scout/agents/base.py)

```python
class BaseAgent(ABC):
    @abstractmethod
    async def run(self, input_data: dict[str, Any]) -> dict[str, Any]:
        ...
```

All agents are async, accept a shared state dict, and return an updated state dict. This interface is kept minimal to allow flexible state passing through the pipeline.

---

## Agent Catalogue

### Orchestrator

**File:** [src/indication_scout/agents/orchestrator.py](../src/indication_scout/agents/orchestrator.py)

Coordinates all specialist agents in a deterministic sequence. Responsible for:

- Accepting a drug name as input
- Calling the Drug Profile Agent to resolve the drug and fetch its profile
- Fanning out to the Discovery Agent (both paths in parallel)
- Iterating over top-N candidates and gathering literature and clinical trial evidence
- Invoking the Report Generator to produce the final Markdown output
- Collecting and surfacing errors from downstream agents

**Planned flow:**

```
drug_name
    │
    ▼
Drug Profile Agent     ← resolves drug name, fetches targets and indications
    │
    ▼ (if drug found)
Discovery Agent        ← runs Path 1 and Path 2, merges and ranks candidates
    │
    ▼ (for each top-N candidate)
┌─────────────────────────────────┐
│ Literature Agent                │  ← PubMed search + RAG rerank
│ Clinical Trials Agent           │  ← ClinicalTrials.gov evidence
└─────────────────────────────────┘
    │
    ▼
Report Generator       ← synthesises Markdown report
```

**State (TypedDict):**

| Key | Type | Description |
|-----|------|-------------|
| `drug_name` | `str` | Input drug name |
| `drug` | `dict \| None` | Serialised drug profile |
| `candidates` | `list[dict]` | Merged repurposing candidates |
| `literature` | `dict[str, dict]` | EFO ID → abstracts + count |
| `trials` | `dict[str, dict]` | EFO ID → trial count, phases, status |
| `current_step` | `str` | Pipeline position for error context |
| `errors` | `list[str]` | Non-fatal errors accumulated |
| `report` | `str \| None` | Final Markdown output |

State uses plain dicts (not Pydantic models) to satisfy LangGraph's serialisation requirement.

---

### LiteratureAgent

**File:** [src/indication_scout/agents/literature.py](../src/indication_scout/agents/literature.py)

Searches PubMed for evidence linking a drug to a candidate indication and reranks results using vector similarity (RAG).

**Responsibilities:**
- Query PubMedClient with `drug_name + disease_name` search terms
- Fetch and parse abstracts via `PubMedClient.fetch_articles()`
- Embed abstracts and store in pgvector
- At retrieval time, embed the therapeutic query and rerank by cosine similarity
- Return the top-K most relevant abstracts to the Orchestrator

**Data source:** `PubMedClient` → `PubMedArticle` models

**Why RAG matters here:**
A raw PubMed search for *bupropion + obesity* returns mostly depression papers that mention obesity in passing. RAG surfaces papers specifically about Contrave (naltrexone/bupropion) for weight management — the clinically relevant signal.

---

### ClinicalTrialsAgent

**File:** [src/indication_scout/agents/clinical_trials.py](../src/indication_scout/agents/clinical_trials.py)

Searches ClinicalTrials.gov for drug–disease trial evidence.

**Responsibilities:**
- Call `ClinicalTrialsClient.search_trials()` for each candidate
- Detect whitespace opportunities via `detect_whitespace()`
- Classify terminated trials via `get_terminated_trials()`
- Summarise trial landscape: phase distribution, active vs. completed, enrolment size
- Return structured evidence per candidate indication

**Data source:** `ClinicalTrialsClient` → `Trial`, `WhitespaceResult`, `ConditionLandscape`, `TerminatedTrial` models

---

### MechanismAgent

**File:** [src/indication_scout/agents/mechanism.py](../src/indication_scout/agents/mechanism.py)

Analyses the drug's mechanism of action to assess biological plausibility for candidate indications.

**Planned responsibilities:**
- Retrieve target–pathway information from Open Targets
- Identify pathway overlap between the drug's known targets and the candidate disease's implicated biology
- Score mechanistic plausibility per candidate

**Data source:** `OpenTargetsClient` → `TargetData`, `Association` models

---

### SafetyAgent

**File:** [src/indication_scout/agents/safety.py](../src/indication_scout/agents/safety.py)

Analyses the drug's safety profile to flag risks for candidate indications.

**Planned responsibilities:**
- Query openFDA FAERS for adverse event counts and reaction profiles
- Flag adverse reactions that would be contraindicated for the candidate population
- Surface serious adverse events (SAEs) from clinical trial data
- Return a structured safety summary per candidate

**Data source:** `FDAClient` → `FAERSEvent`, `FAERSReactionCount` models

---

## Discovery Paths

The Discovery Agent implements two independent search strategies. Candidates found by both paths are ranked highest.

### Path 1 — Target-Disease Associations

```
Drug → known targets → associated diseases (via Open Targets)
```

For each of the drug's molecular targets, query Open Targets `associatedDiseases` to find diseases with biological evidence (genetic, pathway, literature) implicating that target. This surfaces mechanistically grounded candidates even without prior clinical use.

### Path 2 — Drug Class Analogy

```
Drug → known targets → sibling drugs on same targets → sibling drug indications
```

For each target, query Open Targets `knownDrugs` to find other drugs that act on the same target. Collect those sibling drugs' approved indications. This surfaces candidates supported by existing clinical translation — if a sibling drug is already approved for an indication, the repurposing hypothesis has prior clinical validation.

### Merge and Rank

| Signal | Weight |
|--------|--------|
| Found by both paths | Highest |
| Open Targets biology score | Secondary |
| Number of sibling drugs with indication | Tertiary |

Candidates are deduplicated by EFO disease ID, tagged with which path(s) surfaced them, and ranked by this composite signal. The top-N (planned: 10) proceed to the evidence-gathering phase.

---

## Data Flow

```
CLI / API
    │  drug_name
    ▼
Orchestrator
    │
    ├── OpenTargetsClient ──────→ DrugData, TargetData, Association
    │   (drug profile + discovery)
    │
    ├── PubMedClient ───────────→ PubMedArticle
    │   (literature evidence)
    │
    ├── ClinicalTrialsClient ───→ Trial, WhitespaceResult, ConditionLandscape, TerminatedTrial
    │   (trial evidence)
    │
    └── FDAClient ─────────────→ FAERSEvent, FAERSReactionCount
        (safety evidence)
              │
              ▼
        Pydantic models (models/)
              │
              ▼
        Agent state (dicts)
              │
              ▼
        Markdown report
```

Data source clients return validated Pydantic models. Agents consume these models and write results into the shared state dict. No raw API responses cross module boundaries.

---

## Validated Test Drugs

The agent pipeline will be evaluated against 10 ground-truth repurposings with known outcomes:

| Drug | Original Indication | Known Repurposing | Primary Path |
|------|--------------------|--------------------|--------------|
| Baricitinib | Rheumatoid arthritis | Alopecia areata, atopic dermatitis | Path 2 |
| Imatinib | CML | GIST | Path 2 |
| Sildenafil | Erectile dysfunction | Pulmonary arterial hypertension | Path 2 |
| Bupropion | Depression | Smoking cessation, ADHD | Path 2 |
| Thalidomide | Sedative | Multiple myeloma | Path 2 |
| Rituximab | Lymphoma | Rheumatoid arthritis | Path 2 |
| Duloxetine | Depression | Chronic pain, fibromyalgia | Path 2 |
| Metformin | Type 2 diabetes | PCOS | Path 2 |
| Colchicine | Gout | Cardiovascular prevention | Path 2 |
| Empagliflozin | Type 2 diabetes | Heart failure | Path 2 |

Success metrics: **Recall@10** (known repurposing in top-10 candidates) and **MRR** (mean reciprocal rank).

---

## Implementation Roadmap

| Sprint | Focus | Agents |
|--------|-------|--------|
| Sprint 1 | RAG pipeline + Path 2 validation | LiteratureAgent (embedding/retrieval) |
| Sprint 2 | All agents as LangGraph tools | All specialist agents |
| Sprint 3 | ReAct orchestrator + report generation | Orchestrator |
| Sprint 4 | Evaluation + Streamlit UI | All (integration) |
