# Clinical Trials Agent

Assesses the clinical trial landscape for a drug-disease pair. Returns structured data
(trial records, whitespace signal, competitive landscape, terminated trials) plus a
2–3 sentence natural language summary.

---

## Architecture

```
build_clinical_trials_agent(llm, date_before)  ← LangGraph create_react_agent
    └─ build_clinical_trials_tools(date_before)
         ├─ detect_whitespace(drug, indication)
         ├─ search_trials(drug, indication)
         ├─ get_landscape(indication)
         ├─ get_terminated(drug, indication)
         └─ finalize_analysis(summary)            ← terminates the loop

run_clinical_trials_agent(agent, drug_name, disease_name)
    └─ agent.ainvoke(...)
    └─ Walk message history → assemble ClinicalTrialsOutput
```

Each tool first resolves `indication → MeSH D-number` via
`services.disease_helper.resolve_mesh_id` and forwards it to the client as
`target_mesh_id` so the client can post-filter Essie's recall-first results.
If the resolver returns `None` the tool short-circuits to the empty-result
shape and logs a WARNING — no client call is made.

### Files

| File | Role |
|---|---|
| `agents/clinical_trials/clinical_trials_agent.py` | `build_clinical_trials_agent` + `run_clinical_trials_agent`, system prompt, message-history → output assembly |
| `agents/clinical_trials/clinical_trials_tools.py` | LangChain `@tool` wrappers around the client; calls `resolve_mesh_id` per tool |
| `agents/clinical_trials/clinical_trials_output.py` | `ClinicalTrialsOutput` — the agent's return type |
| `data_sources/clinical_trials.py` | ClinicalTrials.gov API v2 client (incl. `_filter_by_mesh`) |
| `models/model_clinical_trials.py` | Pydantic models between client and agents |
| `services/disease_helper.py` | `resolve_mesh_id` (NCBI E-utilities, cached) |

---

## Entry Point

```python
def build_clinical_trials_agent(llm, date_before: date | None = None)
async def run_clinical_trials_agent(agent, drug_name: str, disease_name: str) -> ClinicalTrialsOutput
```

`date_before` is captured via closure inside `build_clinical_trials_tools` and
applied to every client call so all queries in a session operate on a
consistent time window.

`run_clinical_trials_agent` invokes the LangGraph ReAct agent with a single
`HumanMessage("Analyze {drug_name} in {disease_name}")` and walks the
resulting message history, pulling typed artifacts off `ToolMessage`s and
mapping each tool name to its slot in `ClinicalTrialsOutput`:

| Tool name | ClinicalTrialsOutput field |
|---|---|
| `detect_whitespace` | `whitespace` |
| `search_trials` | `trials` |
| `get_landscape` | `landscape` |
| `get_terminated` | `terminated` |
| `finalize_analysis` | `summary` |

---

## ReAct Loop

The agent is built via `langgraph.prebuilt.create_react_agent`. The system
prompt (in `clinical_trials_agent.py`) instructs:

1. Typically start with `detect_whitespace`.
2. ALWAYS call `get_terminated` alongside `search_trials` and `get_landscape` — terminated trials are evidence whether or not active trials exist.
3. Batch independent tool calls into a single response when possible.
4. `finalize_analysis(summary)` MUST be the final tool call — it terminates the loop and supplies the 2–3 sentence plain-text assessment.

The summary covers: whether trials exist for the pair, competitive landscape
(crowded vs. open), red flags from terminated trials, overall opportunity
assessment. It must reference only information returned by the tools in this
run (no facts from the LLM's training).

---

## Tools

Tools are thin `@tool(response_format="content_and_artifact")` wrappers that
call `ClinicalTrialsClient`. Each one resolves the indication to a MeSH
D-number first; if the resolver returns `None`, the tool returns an empty
artifact with a "MeSH unresolved, skipped" content message and never
contacts CT.gov. All tools share the `date_before` cutoff via closure.

Sizing limits below come from settings (snake_case in `config.py`,
overridable via env vars in `.env.constants` — the values shown are the
example defaults).

### `detect_whitespace(drug, indication) → (str, WhitespaceResult)`

Detects whether the drug-indication pair has been explored. Runs three concurrent queries:

- **Exact match**: trials with both drug AND indication (cap: `clinical_trials_whitespace_exact_max = 50`)
- **Drug-only count**: total trials with this drug (no MeSH filter applied — there is no indication to filter against)
- **Indication-only count**: total trials with this indication (paginated + MeSH-filtered when `target_mesh_id` is set; otherwise short-circuits on the API's `totalCount`)

When no exact matches exist (whitespace), also fetches up to `clinical_trials_whitespace_indication_max = 200`
indication trials filtered to `CLINICAL_TRIALS_WHITESPACE_PHASE_FILTER = "(PHASE2 OR PHASE3 OR PHASE4)"`,
applies the MeSH filter, ranks by phase desc then active status, deduplicates by
drug name, and returns the top `clinical_trials_whitespace_top_drugs = 50` unique
drugs as `indication_drugs`.

Returns a `WhitespaceResult` artifact.

### `search_trials(drug, indication) → (str, list[Trial])`

Fetches trial records for the pair, capped at `clinical_trials_search_max = 50`,
sorted by `EnrollmentCount:desc`. Results are MeSH-post-filtered before return.

Returns a list of `Trial` artifacts.

### `get_landscape(indication) → (str, IndicationLandscape)`

Fetches up to `clinical_trials_landscape_max_trials = 50` trials for the
indication sorted by `StartDate:desc`, MeSH-post-filters them, then aggregates
into the top `10` competitors grouped by sponsor + drug. Drug/Biological
interventions only; vaccines (matched by `VACCINE_NAME_KEYWORDS`) are excluded.

Returns an `IndicationLandscape` artifact with `competitors`,
`phase_distribution`, and `recent_starts`.

### `get_terminated(drug, indication) → (str, TrialOutcomes)`

Runs four concurrent queries and returns a `TrialOutcomes` with four
scope-labelled lists:

- **drug_wide** (`list[TerminatedTrial]`): TERMINATED trials for this drug, ANY indication, page size `clinical_trials_terminated_drug_page_size = 50`. Filtered to `stop_category in {safety, efficacy}` only — business/enrollment/unknown are dropped as noise. Not MeSH-filtered (no indication to filter against).
- **indication_wide** (`list[TerminatedTrial]`): TERMINATED trials for ANY drug in this indication. MeSH-filtered, then capped at `clinical_trials_terminated_indication_max = 20`. All stop categories retained.
- **pair_specific** (`list[TerminatedTrial]`): TERMINATED trials for this drug AND this indication. MeSH-filtered. All stop categories retained — a safety/efficacy entry here means the exact hypothesis has been directly tested and stopped early.
- **pair_completed** (`list[Trial]`): COMPLETED trials for this drug AND this indication. MeSH-filtered. Catches Phase 3 trials that ran to protocol end but missed their primary endpoint (CT.gov marks those COMPLETED, not TERMINATED).

### `finalize_analysis(summary) → (str, str)`

Terminates the agent loop. Receives the agent's 2–3 sentence plain-text
summary and returns it as the artifact, which `run_clinical_trials_agent`
maps to `ClinicalTrialsOutput.summary`.

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
| `date_before` | `query.term` | `AREA[StartDate]RANGE[MIN, date]` |
| `phase_filter` | `query.term` | `AREA[Phase](PHASE2 OR ...)` syntax |
| `status_filter` | `filter.overallStatus` | e.g. `TERMINATED` |
| `sort` | `sort` | e.g. `EnrollmentCount:desc` |

### MeSH post-filter

`_filter_by_mesh(trials, target_mesh_id)` keeps trials whose
`mesh_conditions` or `mesh_ancestors` (both extracted from
`derivedSection.conditionBrowseModule`) contain the target D-number.
Trials with both lists empty are dropped (cannot be verified against the
MeSH tree). Comparison is on the D-number `id` field, not `term`.

When `target_mesh_id` is supplied, `_count_trials` also paginates and
applies the filter — it cannot use the API's `totalCount` because that
reflects the unfiltered Essie query.

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

Core trial record. Key fields: `nct_id`, `title`, `phase`, `overall_status`,
`why_stopped`, `indications` (list of condition strings),
`mesh_conditions` / `mesh_ancestors` (lists of `MeshTerm`, used by
`_filter_by_mesh`), `interventions` (list of `Intervention`), `sponsor`,
`enrollment`, `start_date`, `completion_date`, `primary_outcomes`, `references` (PMIDs).

### `IndicationLandscape`

| Field | Type | Description |
|---|---|---|
| `total_trial_count` | `int \| None` | Total trials for this indication (post-MeSH-filter when `target_mesh_id` is set; otherwise the API's `totalCount`) |
| `competitors` | `list[CompetitorEntry]` | Top N by `max_phase` desc, then `most_recent_start` desc |
| `phase_distribution` | `dict[str, int]` | Count of trials per phase |
| `recent_starts` | `list[RecentStart]` | Trials starting ≥ `CLINICAL_TRIALS_RECENT_START_YEAR` |

`CompetitorEntry` groups by sponsor + drug: `max_phase`, `trial_count`,
`statuses`, `total_enrollment`, `most_recent_start`.

### `TerminatedTrial`

A standalone Pydantic model (not a `Trial` subclass) — fields:
`nct_id`, `title`, `drug_name`, `indication`, `mesh_conditions`, `phase`,
`why_stopped`, `stop_category`, `enrollment`, `sponsor`, `start_date`,
`termination_date`. `stop_category` is one of: `safety`, `efficacy`,
`business`, `enrollment`, `other`, `unknown`.

### `TrialOutcomes`

Returned by `get_terminated`. Fields: `drug_wide: list[TerminatedTrial]`,
`indication_wide: list[TerminatedTrial]`, `pair_specific: list[TerminatedTrial]`,
`pair_completed: list[Trial]`. See the `get_terminated` tool description above
for the semantics of each scope.

### `ClinicalTrialsOutput`

The agent's return type:

| Field | Type | Notes |
|---|---|---|
| `whitespace` | `WhitespaceResult \| None` | From `detect_whitespace` |
| `landscape` | `IndicationLandscape \| None` | From `get_landscape`; None if skipped |
| `trials` | `list[Trial]` | From `search_trials`; empty if not called |
| `terminated` | `TrialOutcomes` | From `get_terminated`; empty `TrialOutcomes()` if not called |
| `summary` | `str` | From `finalize_analysis` — the 2–3 sentence natural language assessment |

Fields are `None`/empty when the agent chose not to call the corresponding tool — this is
expected, not an error.

---

## Result Assembly

`run_clinical_trials_agent` walks the agent's message history after
`ainvoke()` completes:

- Each `ToolMessage` carries a typed `.artifact` (set by the tool's
  `response_format="content_and_artifact"`). The tool name is mapped to a
  `ClinicalTrialsOutput` field via the `field_map` in
  `clinical_trials_agent.py` and the artifact assigned directly — no JSON
  parsing or model reconstruction is needed.
- `finalize_analysis`'s artifact is the summary string.

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
