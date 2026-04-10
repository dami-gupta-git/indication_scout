# Clinical Trials Agent

Assesses the clinical trial landscape for a drug-disease pair. Returns structured data
(trial records, whitespace signal, competitive landscape, terminated trials) plus a
2–3 sentence natural language summary.

---

## Architecture

```
ClinicalTrialsAgent.run()
    └─ build_clinical_trials_tools()  ← wraps ClinicalTrialsClient methods
    └─ LangChain ReAct agent (ChatAnthropic + tools)
         ├─ detect_whitespace(drug, indication)
         ├─ search_trials(drug, indication)
         ├─ get_landscape(indication)
         └─ get_terminated(drug, indication)
    └─ _parse_result()  ← extracts structured data from message history
    └─ ClinicalTrialsOutput
```

### Files

| File | Role |
|---|---|
| `agents/clinical_trials.py` | Agent orchestration, system prompt, result parsing |
| `agents/clinical_trials_tools.py` | LangChain `@tool` wrappers around the client |
| `data_sources/clinical_trials.py` | ClinicalTrials.gov API v2 client |
| `models/model_clinical_trials.py` | Pydantic models between client and agents |
| `agents/model_clinical_trials_agent.py` | `ClinicalTrialsOutput` — the agent's return type |

---

## Entry Point

```python
class ClinicalTrialsAgent(BaseAgent):
    async def run(self, input_data: dict[str, Any]) -> dict[str, Any]
```

**Input:**

| Key | Type | Required |
|---|---|---|
| `drug_name` | `str` | Yes |
| `disease_name` | `str` | Yes |
| `date_before` | `date \| None` | No — temporal holdout cutoff |

Both names are lowercased before use. `date_before` is captured via closure and applied to
every tool call, so all queries in a session operate on a consistent time window.

**Output:** `{"clinical_trials_output": ClinicalTrialsOutput}`

---

## ReAct Loop

The agent uses `langchain.agents.create_agent` with `temperature=0` and a `RECURSION_LIMIT`
of 15. The system prompt instructs this call order:

1. **Always** start with `detect_whitespace`.
2. If trials exist → `search_trials`, then `get_landscape`.
3. If whitespace → `get_terminated` (check for prior failures), then optionally `get_landscape`.
4. If `get_terminated` finds safety/efficacy failures → that is a strong negative signal; may skip `get_landscape`.

The agent ends with a 2–3 sentence plain-text assessment covering:
- Whether trials exist for the pair
- Competitive landscape (crowded vs. open)
- Red flags from terminated trials
- Overall opportunity assessment

---

## Tools

Tools are thin wrappers that call `ClinicalTrialsClient` and return dicts/lists so the LLM
can read the results. All tools share the `date_before` cutoff via closure.

### `detect_whitespace(drug, indication) → dict`

Detects whether the drug-indication pair has been explored. Runs three concurrent queries:

- **Exact match**: trials with both drug AND indication (cap: `CLINICAL_TRIALS_WHITESPACE_EXACT_MAX = 50`)
- **Drug-only count**: total trials with this drug (count only, 1 record fetched)
- **Indication-only count**: total trials with this indication (count only, 1 record fetched)

When no exact matches exist (whitespace), also fetches up to `CLINICAL_TRIALS_WHITESPACE_INDICATION_MAX = 200`
indication trials filtered to `CLINICAL_TRIALS_WHITESPACE_PHASE_FILTER = "(PHASE2 OR PHASE3 OR PHASE4)"`,
deduplicates by drug name (keeping the highest-phase trial per drug), and returns top
`CLINICAL_TRIALS_WHITESPACE_TOP_DRUGS = 50` unique drugs as `indication_drugs`.

Returns a `WhitespaceResult` dict.

### `search_trials(drug, indication) → list[dict]`

Fetches trial records for the pair. Fixed to `max_results=50`. No sort order is passed,
so results are in ClinicalTrials.gov relevance order.

Returns a list of `Trial` dicts.

### `get_landscape(indication) → dict`

Fetches up to `CLINICAL_TRIALS_LANDSCAPE_MAX_TRIALS = 50` trials for the indication
sorted by enrollment descending, filters to Drug/Biological interventions only, and
aggregates into top `10` competitors grouped by sponsor + drug.

Returns an `IndicationLandscape` dict with `competitors`, `phase_distribution`, and `recent_starts`.

### `get_terminated(drug, indication) → list[dict]`

Runs two concurrent queries:

- **Drug query**: terminated trials for this drug (any indication), capped at
  `CLINICAL_TRIALS_TERMINATED_DRUG_PAGE_SIZE = 20`. Only `stop_category` in
  `{"safety", "efficacy"}` are returned — business/enrollment/unknown terminations are dropped as noise.
- **Indication query**: all terminated trials in this indication space (any drug), capped at `20`.

Returns the union deduplicated by `nct_id`.

---

## Client: `ClinicalTrialsClient`

Extends `BaseClient`. Base URL: `https://clinicaltrials.gov/api/v2/studies`. Page size: 100.

### Pagination

`_paginated_search()` is the core loop shared by all trial-fetching methods. It pages via
`nextPageToken` until `max_results` is reached or no more pages are returned.

### Query building

`_build_search_params()` constructs the API query:

| Parameter | API field | Notes |
|---|---|---|
| `drug` | `query.intr` | Free-text intervention search |
| `indication` | `query.cond` | Free-text condition search |
| `date_before` | `query.term` | `AREA[StudyFirstPostDate]RANGE[MIN, date]` |
| `phase_filter` | `query.term` | `AREA[Phase](PHASE2 OR ...)` syntax |
| `status_filter` | `filter.overallStatus` | e.g. `TERMINATED` |
| `sort` | `sort` | e.g. `EnrollmentCount:desc` |

### Stop reason classification

`_classify_stop_reason(why_stopped)` maps free-text termination reasons to categories using
`STOP_KEYWORDS`. Checks for negation prefixes to avoid false positives (e.g. "no safety
concerns"). Returns: `safety`, `efficacy`, `business`, `enrollment`, `other`, or `unknown`.

### Phase normalisation

`_phase_rank()` maps phase strings to integers (0–8, higher = later stage) for sorting.
`_normalize_phase()` converts the v2 API's phase list (e.g. `["PHASE2", "PHASE3"]`) to
a human-readable string (`"Phase 2/Phase 3"`).

---

## Data Models

### `WhitespaceResult`

| Field | Type | Description |
|---|---|---|
| `is_whitespace` | `bool \| None` | True if no exact-match trials exist |
| `no_data` | `bool \| None` | True if drug AND indication both have zero trials anywhere |
| `exact_match_count` | `int \| None` | Number of trials with both drug and indication |
| `drug_only_trials` | `int \| None` | Total trials for this drug (any indication) |
| `indication_only_trials` | `int \| None` | Total trials for this indication (any drug) |
| `indication_drugs` | `list[IndicationDrug]` | Other drugs in the indication space (whitespace case only) |

### `Trial`

Core trial record. Key fields: `nct_id`, `title`, `phase`, `overall_status`, `why_stopped`,
`indications`, `interventions` (list of `Intervention`), `sponsor`, `enrollment`,
`start_date`, `completion_date`, `primary_outcomes`, `references` (PMIDs).

### `IndicationLandscape`

| Field | Type | Description |
|---|---|---|
| `total_trial_count` | `int \| None` | Total trials for this indication (from API count) |
| `competitors` | `list[CompetitorEntry]` | Top N by phase then enrollment |
| `phase_distribution` | `dict[str, int]` | Count of trials per phase |
| `recent_starts` | `list[RecentStart]` | Trials starting ≥ `CLINICAL_TRIALS_RECENT_START_YEAR` |

`CompetitorEntry` groups by sponsor + drug: `max_phase`, `trial_count`, `statuses`, `total_enrollment`.

### `TerminatedTrial`

Extends `Trial` with `drug_name`, `indication`, `stop_category`, and `termination_date`.
`stop_category` is one of: `safety`, `efficacy`, `business`, `enrollment`, `other`, `unknown`.

### `ClinicalTrialsOutput`

The agent's return type:

| Field | Type | Notes |
|---|---|---|
| `trials` | `list[Trial]` | From `search_trials`; empty if not called |
| `whitespace` | `WhitespaceResult \| None` | From `detect_whitespace`; always populated |
| `landscape` | `IndicationLandscape \| None` | From `get_landscape`; None if skipped |
| `terminated` | `list[TerminatedTrial]` | From `get_terminated`; empty if not called |
| `summary` | `str` | Final AI message — the natural language assessment |

Fields are `None`/empty when the agent chose not to call the corresponding tool — this is
expected, not an error.

---

## Result Parsing

`_parse_result()` walks the agent's message history after `ainvoke()` completes:

- Messages with a `.name` attribute are tool responses. Content is JSON-decoded if it
  arrives as a string, then deserialized into the appropriate Pydantic model by tool name.
- The last message without a `.name` is the final AI message. Handles both plain string
  content and list-of-blocks content formats.

---

## Known Limitations & Future Work

See [future.md](../future.md) for the full list. The two most relevant:

1. **Drug synonym expansion** — `query.intr` is a free-text field. A drug registered as
   "metformin hydrochloride" or a brand name won't match a query for "metformin", causing
   false whitespace signals.

2. **`is_whitespace` is binary** — it doesn't capture "early stage, unproven." For example,
   metformin + glioblastoma returns `is_whitespace=False` with 9 trials, but those are all
   Phase 1/2 with small enrollment. The maturity dimension lives only in the free-text
   summary, not in the structured data.
