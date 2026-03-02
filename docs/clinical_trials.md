# ClinicalTrials.gov Client — Data Contracts

Five methods. The Trial Agent consumes methods 1 and 2, the Landscape Agent consumes method 3, and the Critique Agent consumes method 4. Method 5 (`get_trial`) fetches a single trial by NCT ID.

**API:** ClinicalTrials.gov v2 REST API (`https://clinicaltrials.gov/api/v2/studies`)
**Auth:** None required
**Rate limit:** ~50 requests/min per IP
**Temporal holdout:** Supported via date range filters on `query.term` using `AREA[StudyFirstPostDate]`

---

## 1. `search_trials(drug, condition, date_before, phase_filter, max_results) → list[Trial]`

The primary search. "What trials exist for this drug in this condition?"

**In:**
- `drug` — drug name or synonym (e.g. `"semaglutide"`)
- `condition` — disease/condition (e.g. `"NASH"`, `"non-alcoholic steatohepatitis"`)
- `date_before` — optional date, for temporal holdout
- `phase_filter` — optional, e.g. `"PHASE3"` or `"(PHASE2 OR PHASE3)"`
- `max_results` — maximum number of results to return (default 200)

**API call:** `GET /api/v2/studies?query.cond={condition}&query.intr={drug}&pageSize=100&countTotal=true`

**Out:** list of trials, each containing:
- `nct_id` — the canonical trial identifier (e.g. `"NCT02970942"`)
- `title` — official trial title (from `protocolSection.identificationModule.briefTitle`)
- `brief_summary` — short description of what the trial is testing
- `phase` — string: `"Phase 1"`, `"Phase 2"`, `"Phase 1/Phase 2"`, `"Phase 3"`, `"Phase 4"`, `"Early Phase 1"`, `"Not Applicable"`
- `overall_status` — this is hugely important for the agents:
  - `"RECRUITING"` — active, hasn't finished
  - `"ACTIVE_NOT_RECRUITING"` — enrolled, running
  - `"COMPLETED"` — finished, results may exist
  - `"TERMINATED"` — **stopped early** (the Critique Agent wants to know *why*)
  - `"WITHDRAWN"` — never started
  - `"SUSPENDED"` — paused
  - `"UNKNOWN"` — lost contact
- `why_stopped` — free text, only populated for Terminated/Suspended/Withdrawn (e.g. `"Lack of efficacy"`, `"Safety concerns"`, `"Business decision"`)
- `conditions` — list of conditions being studied (a trial can target multiple)
- `interventions` — list of:
  - `intervention_type` — `"Drug"`, `"Biological"`, `"Device"`, etc.
  - `intervention_name` — e.g. `"Semaglutide"`, `"Placebo"`
  - `description` — dosing info, formulation
- `sponsor` — lead sponsor name (e.g. `"Novo Nordisk"`)
- `collaborators` — list of collaborating organizations
- `enrollment` — number of participants (int)
- `start_date` — when the trial started
- `completion_date` — primary completion date (actual or estimated)
- `study_type` — `"INTERVENTIONAL"` vs `"OBSERVATIONAL"` (we mostly care about interventional)
- `primary_outcomes` — list of:
  - `measure` — what they're measuring (e.g. `"Change in NAS score"`)
  - `time_frame` — e.g. `"72 weeks"`
- `results_posted` — bool, whether results are available on ClinicalTrials.gov
- `references` — list of PMIDs linked to this trial (bridge back to PubMed client)

**v2 API response path mapping:**
```
nct_id           → protocolSection.identificationModule.nctId
title            → protocolSection.identificationModule.briefTitle
brief_summary    → protocolSection.descriptionModule.briefSummary
phase            → protocolSection.designModule.phases
overall_status   → protocolSection.statusModule.overallStatus
why_stopped      → protocolSection.statusModule.whyStopped
conditions       → protocolSection.conditionsModule.conditions
interventions    → protocolSection.armsInterventionsModule.interventions
sponsor          → protocolSection.sponsorCollaboratorsModule.leadSponsor.name
collaborators    → protocolSection.sponsorCollaboratorsModule.collaborators[].name
enrollment       → protocolSection.designModule.enrollmentInfo.count
start_date       → protocolSection.statusModule.startDateStruct.date
completion_date  → protocolSection.statusModule.primaryCompletionDateStruct.date
study_type       → protocolSection.designModule.studyType
primary_outcomes → protocolSection.outcomesModule.primaryOutcomes
results_posted   → hasResults
references       → protocolSection.referencesModule.references[].pmid
```

**Why the agents need all of this:**

The Trial Agent uses `phase`, `overall_status`, and `enrollment` to assess how mature the clinical evidence is. A completed Phase 3 with 3,000 patients means something very different from an early Phase 1 with 20.

The Critique Agent zeroes in on `overall_status == "TERMINATED"` + `why_stopped`. If a semaglutide NASH trial was terminated for safety concerns, that's a critical finding that should surface as an adversarial concern.

The Supervisor uses `references` (PMIDs) to hand to the Literature Agent — "here are papers directly linked to clinical trials for this drug-indication pair, go fetch and synthesize them."

---

## 2. `detect_whitespace(drug, condition, date_before) → WhitespaceResult`

This is the novel signal the system detects. "Does a clinical trial exist for this drug-condition pair?"

**In:**
- `drug` — drug name
- `condition` — candidate indication
- `date_before` — optional date, for temporal holdout

**Out:**
- `is_whitespace` — bool. `True` = no trials found = potential opportunity
- `exact_match_count` — number of trials that match both drug AND condition
- `drug_only_trials` — count of trials for this drug in *any* condition (tells you the drug is actively being studied)
- `condition_only_trials` — count of trials for this condition with *any* drug (tells you the condition is being pursued)
- `condition_drugs` — when whitespace exists, returns other drugs being tested for the same condition. Each entry includes: `nct_id`, `drug_name`, `condition`, `phase`, `status`

**Implementation notes:**

This is a derived method — it makes three concurrent calls under the hood:
1. Exact match: trials with both drug AND condition (max 50)
2. Drug-only count: total trials with this drug (any condition)
3. Condition-only count: total trials with this condition (any drug)

When whitespace exists (no exact matches), it fetches condition trials and populates `condition_drugs`:
- **Filters to Phase 2+ only** (excludes noisy Phase 1 data for meaningful efficacy signal)
- **Ranks by phase (descending) then active status** (recruiting preferred)
- **Deduplicates by drug_name**, keeping the highest-ranked trial per drug
- **Returns top 50 unique drugs**

**Why this matters:** Whitespace detection is one of the system's key differentiators. But raw whitespace (zero trials) isn't enough — you need context. If `drug_only_trials` is high and `condition_only_trials` is high but `exact_match_count` is zero, that's a genuinely interesting gap. If `drug_only_trials` is zero, maybe the drug just isn't being developed at all (less interesting). The `condition_drugs` tell you what other drugs are being tested for this condition.

---

## 3. `get_landscape(condition, date_before, top_n) → ConditionLandscape`

The Landscape Agent's primary data source. "Who's working on this condition and how far along are they?"

**In:**
- `condition` — disease/condition name
- `date_before` — optional date, for temporal holdout
- `top_n` — number of top competitors to return (default 50)

**API call:** `GET /api/v2/studies?query.cond={condition}&query.term=AREA[Phase](EARLY_PHASE1 OR PHASE1 OR PHASE2 OR PHASE3 OR PHASE4)&pageSize=100&countTotal=true`

**Out:**
- `total_trial_count` — how active this condition is overall
- `competitors` — list of, grouped by sponsor + drug:
  - `sponsor` — company name
  - `drug_name` — intervention name
  - `drug_type` — `"Drug"` or `"Biological"`
  - `max_phase` — highest phase reached
  - `trial_count` — how many trials this sponsor is running
  - `statuses` — set of statuses across their trials (e.g. `{"COMPLETED", "RECRUITING"}`)
  - `total_enrollment` — sum of enrollment across their trials
  - `most_recent_start` — date of their newest trial
- `phase_distribution` — dict showing how many trials are in each phase:
  - `{"Phase 1": 12, "Phase 2": 23, "Phase 3": 8, "Phase 4": 4}`
- `recent_starts` — trials started in 2024+ (signal for whether the field is heating up or cooling down)
  - list of `{"nct_id", "sponsor", "drug", "phase"}`

**Implementation notes:**
- Fetches all trials for the condition with phase filter `(EARLY_PHASE1 OR PHASE1 OR PHASE2 OR PHASE3 OR PHASE4)`
- Filters client-side to `intervention_type` in `("Drug", "Biological")` only
- Groups by sponsor + drug
- Ranks by max phase (descending) then total enrollment (descending)
- Returns top N competitors

**Why the agents need this:** The Landscape Agent turns this into a competitive narrative: "NASH is a crowded space with 47 active trials. Resmetirom (Madrigal) is furthest along at Phase 3. There are 8 GLP-1 agonists in various phases, suggesting the mechanism is validated but competition is fierce." The `phase_distribution` tells you whether the field is early-stage exploratory or late-stage competitive. The `recent_starts` signal whether investment is growing or shrinking.

---

## 4. `get_terminated(query, date_before, max_results) → list[TerminatedTrial]`

The Critique Agent's weapon. "What has failed before in this space?"

**In:**
- `query` — either a drug name, a drug class, or a condition. The Critique Agent calls this multiple ways:
  - `"semaglutide"` — has this specific drug failed anywhere?
  - `"GLP-1 receptor agonist"` — has this drug class failed?
  - `"NASH"` — what has failed in this condition?
- `date_before` — optional date, for temporal holdout
- `max_results` — maximum number of results to return (default 100)

**API call:** `GET /api/v2/studies?query.term={query}&filter.overallStatus=TERMINATED,WITHDRAWN,SUSPENDED&pageSize=100`

**Out:** list of terminated/withdrawn trials:
- `nct_id` — trial ID
- `title` — trial title
- `drug_name` — what was being tested (first Drug/Biological intervention found)
- `condition` — what it was being tested for (first condition listed)
- `phase` — how far it got
- `why_stopped` — the critical field. Free text from ClinicalTrials.gov
- `stop_category` — one of: `"safety"`, `"efficacy"`, `"business"`, `"enrollment"`, `"other"`, `"unknown"`
- `enrollment` — how many patients were enrolled before termination
- `sponsor` — who was running it
- `start_date` — when it started
- `termination_date` — when it stopped (from `statusModule.primaryCompletionDateStruct`)
- `references` — PMIDs, if any (papers explaining why it failed are gold for the Critique Agent)

**Stop category classification:**

The `stop_category` is derived from `why_stopped` using keyword matching:

| Keywords | Category |
|----------|----------|
| efficacy, futility, lack of efficacy, no benefit | `efficacy` |
| safety, adverse, toxicity, side effect | `safety` |
| enrollment, accrual, recruitment | `enrollment` |
| business, strategic, funding, commercial | `business` |
| (no match) | `other` |
| (no why_stopped text) | `unknown` |

**Why this exists separately from method 1:** You *could* filter `search_trials` by `overall_status == "TERMINATED"`, but the Critique Agent needs to search more broadly — not just for this exact drug, but for the whole drug class and the whole condition. A terminated Phase 3 trial for *obeticholic acid* in NASH is relevant critique even though it's a different drug, because it might reveal that the condition itself is hard to treat or that a specific endpoint is problematic.

---

## Data quality notes

**Condition names are not standardized.** Unlike Open Targets which uses EFO disease IDs, ClinicalTrials.gov uses free-text condition names. "NASH", "Non-alcoholic Steatohepatitis", "Nonalcoholic Steatohepatitis", "NAFLD/NASH" are all different strings that mean the same thing. The search method needs to handle synonyms, either by:
- Querying multiple variants
- Using ClinicalTrials.gov's built-in MeSH term expansion (the API does some of this automatically)
- Maintaining a condition synonym map for high-value disease areas

**The `why_stopped` field is sparse and inconsistent.** Many terminated trials have no `why_stopped` text at all. When present, it ranges from detailed ("Pre-specified interim analysis showed lack of efficacy on primary endpoint") to useless ("Business decision"). Plan for ~40% of terminated trials having no useful stop reason.

**Pagination.** The v2 API returns max 100 studies per page. For active disease areas, `get_landscape` may need to paginate. Use the `nextPageToken` from the response to fetch subsequent pages.

**Date filtering for temporal holdout.** Use `AREA[StudyFirstPostDate]` in the query term to restrict to trials posted before a cutoff date. Example: `query.term=AREA[StudyFirstPostDate]RANGE[MIN, 2019-06-01]` limits to trials first posted before June 2019.

**Phase filtering.** Use `AREA[Phase]` in the query term to restrict by phase. Example: `query.term=AREA[Phase](PHASE2 OR PHASE3)` limits to Phase 2 and Phase 3 trials.

---

## Pydantic models

```python
class Intervention(BaseModel):
    intervention_type: str       # "Drug", "Biological", "Device", etc.
    intervention_name: str       # e.g. "Semaglutide"
    description: str | None = None

class PrimaryOutcome(BaseModel):
    measure: str                 # what they're measuring
    time_frame: str | None = None

class Trial(BaseModel):
    nct_id: str
    title: str
    brief_summary: str | None = None
    phase: str
    overall_status: str
    why_stopped: str | None = None
    conditions: list[str] = []
    interventions: list[Intervention] = []
    sponsor: str = ""
    collaborators: list[str] = []
    enrollment: int | None = None
    start_date: str | None = None
    completion_date: str | None = None
    study_type: str = "Interventional"
    primary_outcomes: list[PrimaryOutcome] = []
    results_posted: bool = False
    references: list[str] = []        # PMIDs

class ConditionDrug(BaseModel):
    """A drug being tested for the same condition (when whitespace exists)."""
    nct_id: str
    drug_name: str
    condition: str
    phase: str
    status: str

class WhitespaceResult(BaseModel):
    is_whitespace: bool
    exact_match_count: int
    drug_only_trials: int
    condition_only_trials: int
    condition_drugs: list[ConditionDrug] = []

class CompetitorEntry(BaseModel):
    sponsor: str
    drug_name: str
    drug_type: str | None = None
    max_phase: str
    trial_count: int
    statuses: set[str]
    total_enrollment: int = 0
    most_recent_start: str | None = None

class ConditionLandscape(BaseModel):
    total_trial_count: int
    competitors: list[CompetitorEntry] = []
    phase_distribution: dict[str, int] = {}
    recent_starts: list[RecentStart] = []

class TerminatedTrial(BaseModel):
    nct_id: str
    title: str
    drug_name: str | None = None
    condition: str | None = None
    phase: str | None = None
    why_stopped: str | None = None
    stop_category: str = "unknown"     # safety, efficacy, business, enrollment, other, unknown
    enrollment: int | None = None
    sponsor: str | None = None
    start_date: str | None = None
    termination_date: str | None = None
    references: list[str] = []         # PMIDs
```

---

## Data flow from ClinicalTrials.gov into the agents

```
Trial Agent
  │
  ├── search_trials("semaglutide", "NASH")
  │     → active/completed trials, phase progression
  │
  ├── detect_whitespace("semaglutide", "NASH")
  │     → is this a gap? what other drugs are being tested?
  │
  └── OUTPUT: trial_evidence + whitespace_signal


Landscape Agent
  │
  ├── get_landscape("NASH")
  │     → who's competing, how far along
  │
  └── OUTPUT: competitive_map with phase distribution


Critique Agent
  │
  ├── get_terminated("semaglutide")
  │     → has this drug failed anywhere?
  │
  ├── get_terminated("GLP-1 receptor agonist")
  │     → has this class failed?
  │
  ├── get_terminated("NASH")
  │     → what's failed in this condition?
  │
  └── OUTPUT: list of concerns, each with:
        - what failed
        - why it failed (safety/efficacy/business)
        - severity score
        - supporting PMIDs
```

---

## Agent-to-method mapping

| Agent | Methods consumed | What it gets |
|-------|-----------------|--------------|
| **Trial** | 1, 2 | Active/completed trials for the drug-condition pair, whitespace signals |
| **Landscape** | 3 | Competitive map across all drugs for the condition, phase distribution |
| **Critique** | 4 | Failed trials for the drug, drug class, and condition — with stop reasons |
| **Supervisor** | (indirect) | Gets PMIDs from Trial Agent's results to hand to Literature Agent |
