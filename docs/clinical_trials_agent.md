# Clinical Trials Agent

Assesses the clinical trial landscape for a drug × indication pair. The agent loop
fetches trial records (all-status, completed, terminated), the competitive
landscape, and FDA-label approval, and classifies every shown completed/terminated
trial as RELEVANT or CONTAMINATED. The narrative summary, development-stage tier,
and live-vs-closed verdict are authored **post-loop** by isolated LLM judgments —
not by the agent inside the loop.

---

## Architecture

```
build_clinical_trials_agent(llm, date_before, assigned_indication)  ← gated ReAct loop
    └─ build_clinical_trials_tools(date_before, assigned_indication)
         ├─ check_fda_approval(drug, indication)
         ├─ search_trials(drug, indication)
         ├─ get_completed(drug, indication)
         ├─ get_terminated(drug, indication)
         ├─ get_landscape(indication)
         └─ finalize_analysis(verdicts, relevance_reasoning)   ← gates the loop closed

run_clinical_trials_agent(agent, drug_name, disease_name, first_approval)
    └─ agent.ainvoke(...)
    └─ Walk message history → assemble base ClinicalTrialsOutput (field_map)
    └─ derive_trial_signals(...)            → output.signals  (deterministic)
    └─ judge_dev_stage(...)                 → signals.dev_stage / active_programs (LLM)
    └─ judge_ct_summary(...)                → output.summary / closure / closure_reason (LLM)
```

Each trial-fetching tool resolves `indication → (mesh_id, mesh_term)` via
`services.disease_helper.resolve_mesh_id` and passes `mesh_term` to the client,
which filters **server-side** with `AREA[ConditionMeshTerm]"<mesh_term>"`. If the
resolver returns `None`, the tool short-circuits to the empty-result shape and
logs a WARNING — no client call is made.

### Files

| File | Role |
|---|---|
| `agents/clinical_trials/clinical_trials_agent.py` | `build_clinical_trials_agent` + `run_clinical_trials_agent`, message-history → output assembly, post-loop judgment orchestration |
| `agents/clinical_trials/clinical_trials_tools.py` | LangChain `@tool` wrappers around the client; per-tool `resolve_mesh_id`; date-cutoff scrubbing; finalize verdict validation |
| `agents/clinical_trials/clinical_trials_output.py` | `ClinicalTrialsOutput`, `TrialSignals`, `FinalizeClinicalTrialsArtifact` |
| `agents/_react_loop.py` | `build_gated_react_loop` — ReAct graph that ends the moment `finalize` succeeds |
| `agents/_trial_signals.py` | `derive_trial_signals` — deterministic phase/closure signals from relevant trials |
| `data_sources/clinical_trials.py` | ClinicalTrials.gov API v2 client (server-side MeSH filter, stop-reason classification) |
| `models/model_clinical_trials.py` | Pydantic models between client and agent |
| `services/disease_helper.py` | `resolve_mesh_id` → `(mesh_id, mesh_term)` (NCBI E-utilities, cached) |
| `services/dev_stage.py` | `judge_dev_stage` (LLM tier) + `dev_stage_phrase` (tier → display string) |
| `services/clinical_trials_summary.py` | `judge_ct_summary` (LLM prose + closure verdict) |
| `prompts/clinical_trials.txt` | the agent's system prompt |

---

## Entry Point

```python
def build_clinical_trials_agent(llm, date_before: date | None = None,
                                assigned_indication: str | None = None)
async def run_clinical_trials_agent(agent, drug_name: str, disease_name: str,
                                    first_approval: int | None = None) -> ClinicalTrialsOutput
```

`date_before` is captured via closure inside `build_clinical_trials_tools` and
applied to every client call, so all queries operate on a consistent time window
(used for holdout validation). `assigned_indication` is pinned to the tools — a
tool call for any *other* indication is soft-rejected with a nudge back to the
assigned one (the `finalize` validator also raises `DataSourceError` if tools were
reused across indications, treating it as a wiring bug, not a retryable error).

`first_approval` (year the drug was first approved anywhere, from ChEMBL) is **not**
used in the loop — it is fed only to `judge_ct_summary` so its closure judgment can
distinguish "old off-patent drug, no commercial NDA" from "efficacy failure."

---

## Gated ReAct Loop — `agents/_react_loop.py`

The agent is built via `build_gated_react_loop(llm, tools, SYSTEM_PROMPT,
_finalize_done)`. This is LangGraph's two-node ReAct graph (model + ToolNode)
except the tools→model edge is conditional: after the tools node runs,
`_finalize_done(messages)` is checked, and if `finalize_analysis` produced a
truthy artifact in the **trailing** block of `ToolMessage`s, the graph returns
`END` immediately (no extra model turn). A *rejected* finalize returns an
empty-string artifact, which the gate treats as falsy, so the loop continues.

Prompt caching: an `ephemeral` `cache_control` breakpoint is set on the system
message (caches the static system+tools prefix), and on the last message each turn
(caches the growing conversation).

### System prompt workflow

The prompt (`prompts/clinical_trials.txt`) instructs the agent to:

1. Call `check_fda_approval` **first**.
2. Call **all** of `search_trials`, `get_completed`, `get_terminated`,
   `get_landscape` — batched in parallel; do NOT skip any based on the approval
   result.
3. Classify **every** completed/terminated trial it was shown as `relevant` or
   `contaminated`, judging from the trial's interventions (is this the studied
   drug?), title, and brief summary (these separate, e.g., systemic from pulmonary
   disease, and catch other-drug trials swept in by recall-first search).
4. Call `finalize_analysis(verdicts, relevance_reasoning)` as the **last** action.
   Any plain-text prose after this is discarded — the trial-section summary is
   authored separately, post-loop.

---

## Tools

All tools are `@tool(response_format="content_and_artifact")` wrappers over
`ClinicalTrialsClient`. Trial-fetching tools resolve the indication to a MeSH
descriptor first; if `resolve_mesh_id` returns `None`, the tool returns an empty
artifact and never contacts CT.gov. All share the `date_before` cutoff via closure.

Sizing limits come from settings (snake_case in `config.py`, overridable via
`.env.constants`) and `constants.py` (e.g. `CLINICAL_TRIALS_FETCH_MAX = 50`).

### Date-cutoff scrubbing

`_scrub_post_cutoff_outcome(trial, cutoff) → (trial, was_scrubbed)` enforces the
holdout window: a trial whose terminal outcome (completion/termination) occurred
*after* the cutoff has its `overall_status` / `why_stopped` / `completion_date`
stripped (rendered UNKNOWN). `search_trials` keeps such trials with status UNKNOWN;
`get_completed` / `get_terminated` **drop** them and decrement `total_count`.

### `check_fda_approval(drug, indication) → (str, ApprovalCheck)`

- **Live path:** `resolve_drug_name(drug)` → ChEMBL id → `get_all_drug_names` →
  `FDAClient.get_all_label_indications` → `extract_approved_from_labels`.
- **Holdout path (`date_before` set):** looks the pair up in a hardcoded
  approval table `as_of=date_before`.

Returns `ApprovalCheck(is_approved, label_found, matched_indication,
drug_names_checked)`. `label_found=False` means the drug isn't in openFDA at all
(e.g. withdrawn) → approval status undetermined.

### `search_trials(drug, indication) → (str, SearchTrialsResult)`

All-status trial query for the pair. Returns `total_count` (exact, via CT.gov
`countTotal`), `by_status` (counts for RECRUITING / ACTIVE_NOT_RECRUITING /
WITHDRAWN / UNKNOWN — not COMPLETED/TERMINATED), and the top trials by enrollment
(`CLINICAL_TRIALS_FETCH_MAX = 50`). Post-cutoff trials are kept as UNKNOWN.

### `get_completed(drug, indication) → (str, CompletedTrialsResult)`

COMPLETED trials for the pair: exact `total_count` + top 50 by enrollment.
Post-cutoff completions are dropped (count decremented). Records the shown NCT ids
under the normalized indication key for `finalize_analysis` verification.

### `get_terminated(drug, indication) → (str, TerminatedTrialsResult)`

TERMINATED trials for the pair: exact `total_count` + top 50 by enrollment, each
carrying `why_stopped`. The content message counts safety/efficacy stops among the
shown set (`_classify_stop_reason(why_stopped) in {safety, efficacy}`). Post-cutoff
terminations are dropped (count decremented). Also records shown NCT ids for
finalize verification.

### `get_landscape(indication) → (str, IndicationLandscape)`

Competitive landscape for the indication: `total_trial_count`, top 10 competitors
(grouped by sponsor + drug, ranked by max phase then recency), `phase_distribution`,
and `recent_starts` (`CLINICAL_TRIALS_RECENT_START_YEAR`+). Drug/Biological
interventions only; vaccines excluded. **Skipped entirely when `date_before` is set**
(returns an empty `IndicationLandscape()`), because landscape aggregates would leak
post-cutoff trial outcomes.

### `finalize_analysis(verdicts, relevance_reasoning) → (str, FinalizeClinicalTrialsArtifact | "")`

Gates the loop closed. `verdicts` is a list of `{"nct": ..., "verdict":
"relevant" | "contaminated"}`. Validates that **every** NCT shown by `get_completed`
/ `get_terminated` has a verdict and that no verdict names an unseen NCT — on
mismatch it returns `("REJECTED: …", "")`, which fails the gate and loops back to
the model. On success it returns a `FinalizeClinicalTrialsArtifact(relevant_ncts,
contaminated_ncts, relevance_reasoning)`.

---

## Result Assembly & Post-Loop Judgments

`run_clinical_trials_agent` walks the message history after `ainvoke()`:

**1. Harvest artifacts.** Each `ToolMessage` carries a typed `.artifact`; the tool
name maps to a slot via `field_map`:

| Tool name | slot |
|---|---|
| `search_trials` | `search` |
| `get_completed` | `completed` |
| `get_terminated` | `terminated` |
| `get_landscape` | `landscape` |
| `check_fda_approval` | `approval` |
| `finalize_analysis` | `finalize` |

A WARNING is logged if `check_fda_approval` was never called. The `finalize`
artifact is unpacked defensively — if finalize never succeeded, a default
`FinalizeClinicalTrialsArtifact()` is used. The base `ClinicalTrialsOutput` is
built from these plus `relevant_nct_ids` / `contaminated_nct_ids` /
`relevance_reasoning` from the finalize artifact.

**2. `derive_trial_signals`** (deterministic, only when finalize succeeded) →
`output.signals` (`TrialSignals`). Computes the phase signals below from RELEVANT
trials only.

**3. `judge_dev_stage`** (LLM, when signals + relevant trials exist) → overwrites
`signals.dev_stage` and `signals.active_programs`. The relevant-trial set fed here
is completed + terminated filtered by `relevant_nct_ids`, **plus** all-status
search trials not in `contaminated_nct_ids` and not already seen (so an active
Phase 3 that lives only in the search set is not missed).

**4. `judge_ct_summary`** (LLM) → `output.summary` (prose), `output.closure`,
`output.closure_reason`. Fed `dev_stage_phrase(signals)` and `active_programs` so
its prose cannot contradict the resolved tier, plus `first_approval`.

> **LOAD-BEARING ORDER:** `judge_dev_stage` MUST run before `judge_ct_summary` —
> the summary judgment is fed the resolved stage phrase and must not re-author it.

Both LLM judgments are cached (`JUDGMENT_CACHE_TTL`) and return a safe floor /
`None` on parse failure — they never fabricate a tier or prose.

---

## Data Models — `models/model_clinical_trials.py`

### `Trial`

Core trial record. Key fields: `nct_id`, `title`, `brief_summary`, `phase`,
`overall_status`, `why_stopped`, `indications` (condition strings),
`mesh_conditions` / `mesh_ancestors` (`list[MeshTerm]`), `interventions`
(`list[Intervention]`), `sponsor`, `enrollment`, `start_date`, `completion_date`
(prefers `primaryCompletionDate`), `primary_outcomes` (`list[PrimaryOutcome]`),
`references` (PMIDs).

`MeshTerm`: `id` (descriptor, e.g. `D003924`), `term`.
`Intervention`: `intervention_type`, `intervention_name`, `description`.
`PrimaryOutcome`: `measure`, `time_frame`.

### `SearchTrialsResult`

| Field | Type | Description |
|---|---|---|
| `total_count` | `int` | Exact all-status count for the pair |
| `by_status` | `dict[str, int]` | RECRUITING / ACTIVE_NOT_RECRUITING / WITHDRAWN / UNKNOWN (excludes COMPLETED/TERMINATED) |
| `trials` | `list[Trial]` | Top 50 by enrollment |

### `CompletedTrialsResult` / `TerminatedTrialsResult`

Each: `total_count` (exact COMPLETED / TERMINATED count) + `trials` (top 50 by
enrollment). Terminated trials carry `why_stopped`.

### `IndicationLandscape`

| Field | Type | Description |
|---|---|---|
| `total_trial_count` | `int \| None` | Exact count for the indication |
| `competitors` | `list[CompetitorEntry]` | Top 10 by `max_phase` desc, then `most_recent_start` desc |
| `phase_distribution` | `dict[str, int]` | Count of trials per phase (post vaccine filter) |
| `recent_starts` | `list[RecentStart]` | Trials starting ≥ `CLINICAL_TRIALS_RECENT_START_YEAR` |

`CompetitorEntry` groups by sponsor + drug: `drug_type`, `max_phase`,
`trial_count`, `statuses`, `total_enrollment`, `most_recent_start`.
`RecentStart`: `nct_id`, `sponsor`, `drug`, `phase`.

### `ApprovalCheck`

`is_approved`, `label_found`, `matched_indication`, `drug_names_checked`.

### `TrialSignals` (in `clinical_trials_output.py`)

Deterministic, computed from RELEVANT trials by `derive_trial_signals`. `dev_stage`
and `active_programs` start at floor defaults and are overwritten by
`judge_dev_stage`.

| Field | Type | Notes |
|---|---|---|
| `highest_completed_phase` | `str \| None` | Highest completed phase incl. pure Phase 4. **Display fact only** — not a pivotal-evidence signal; do not make tier/closure decisions on it |
| `has_completed_phase3` | `bool` | A relevant completed trial in band {Phase 2/Phase 3, Phase 3, Phase 3/Phase 4} |
| `completed_phase3_nct_ids` | `list[str]` | their NCT ids |
| `has_active_phase3` | `bool` | An active Phase-3-band trial on the all-status search set (best-effort contamination drop) |
| `active_phase3_nct_ids` | `list[str]` | their NCT ids |
| `phase3_terminated_for_cause` | `bool` | A relevant Phase-3-band trial (excl. Phase 3/Phase 4) terminated for safety/efficacy |
| `terminated_phase3_nct_ids` | `list[str]` | their NCT ids |
| `dev_stage` | `str` | floor `"untested"`; set by `judge_dev_stage` |
| `active_programs` | `str` | floor `"None active"`; set by `judge_dev_stage` |

Phase bands: active/completed pivotal = Phase 2/Phase 3 … Phase 3/Phase 4 inclusive;
cause-termination pivotal excludes Phase 3/Phase 4 (ambiguous). Pure Phase 4 ranks
above Phase 3 but is NOT a pivotal band (post-approval / off-label).

### `ClinicalTrialsOutput`

The agent's return type.

| Field | Type | Notes |
|---|---|---|
| `search` | `SearchTrialsResult \| None` | from `search_trials` |
| `completed` | `CompletedTrialsResult \| None` | from `get_completed` |
| `terminated` | `TerminatedTrialsResult \| None` | from `get_terminated` |
| `landscape` | `IndicationLandscape \| None` | from `get_landscape`; empty under holdout |
| `approval` | `ApprovalCheck \| None` | from `check_fda_approval` |
| `summary` | `str` | post-loop prose from `judge_ct_summary`; `""` if synthesis returned None |
| `closure` | `Literal["live", "closed", "unknown"]` | typed verdict from `judge_ct_summary`; supervisor consumes directly |
| `closure_reason` | `str` | one-sentence justification |
| `relevant_nct_ids` | `list[str]` | from finalize |
| `contaminated_nct_ids` | `list[str]` | from finalize — excluded from signals |
| `relevance_reasoning` | `str` | finalize justification |
| `signals` | `TrialSignals \| None` | deterministic facts + resolved dev_stage |

`FinalizeClinicalTrialsArtifact`: `relevant_ncts`, `contaminated_ncts`,
`relevance_reasoning`. (Prose is no longer authored at finalize.)

---

## Client: `ClinicalTrialsClient` — `data_sources/clinical_trials.py`

Extends `BaseClient`. Base URL `https://clinicaltrials.gov/api/v2/studies`, page
size 100.

### Pagination & query building

`_paginated_search()` pages via `nextPageToken` until `max_results` / `max_pages`
is hit or pages exhaust (returns `(trials, saturated)`). `_count_trials_total()`
makes one cheap `pageSize=1, countTotal=true` call. `_build_search_params()` maps:

| Parameter | API field | Notes |
|---|---|---|
| `drug` | `query.intr` | Free-text intervention search |
| `mesh_term` | `query.cond` | **Server-side** MeSH filter via `AREA[ConditionMeshTerm]"<mesh_term>"` (`_mesh_cond`) — same filter for counts and fetch |
| `date_before` | `query.term` | `AREA[StartDate]RANGE[MIN, date]` |
| `phase_filter` | `query.term` | `AREA[Phase](PHASE2 OR …)` |
| `status_filter` | `filter.overallStatus` | e.g. `TERMINATED` |
| `sort` | `sort` | e.g. `EnrollmentCount:desc` |

> Note (`_mesh_cond`, line ~40): `AREA[ConditionMeshTerm]` is a documented KNOWN
> BUG — it can over-match on the MeSH *term* string. There is no longer a
> client-side `_filter_by_mesh` post-filter; filtering is entirely server-side.

### Stop-reason classification

`_classify_stop_reason(why_stopped)` maps free-text termination reasons to
`safety`, `efficacy`, `enrollment`, `business`, or `unknown` using `STOP_KEYWORDS`
(specific phrases checked before catch-alls, e.g. "enrollment futility" before
"futility"). A `NEGATION_PREFIXES` lookback (`no `, `not `, `unrelated to `,
`without `, `non-`) within ~20 chars suppresses false positives (e.g. "no safety
concerns"), unless a separator intervenes. Falls back to the raw text when nothing
matches.

### Phase handling

`_phase_rank()` maps phase strings to integers (Not Applicable=0 … Phase 4=8) for
sorting. `_normalize_phase()` converts the v2 API's phase list (e.g.
`["PHASE2", "PHASE3"]`) to a human-readable string (`"Phase 2/Phase 3"`).

---

## Known Limitations & Future Work

See [future.md](../future.md) for the full list. Most relevant:

1. **Drug synonym expansion** — `query.intr` is free-text; a trial registered under
   "metformin hydrochloride" or a brand name may miss a query for "metformin".
2. **`AREA[ConditionMeshTerm]` over-match** — the server-side MeSH condition filter
   matches on the term string and can pull in clinically distinct diseases sharing
   wording; this is exactly what the agent's relevance split
   (`relevant` vs `contaminated`) is there to clean up downstream.
