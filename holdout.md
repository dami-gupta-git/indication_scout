# Temporal Holdout Methodology

> See also: [README.md](README.md) (project overview and CLI usage).

## Purpose

The temporal holdout simulates running IndicationScout "as of" a past cutoff date so the system can
be evaluated on whether it would have surfaced repurposing opportunities that were later validated.
A holdout run must not see any evidence published or made public on or after the cutoff —
otherwise the evaluation leaks the answer.

The cutoff is a single `date_before: date | None` argument that flows from the CLI down through the
supervisor, sub-agents, data-source clients, and supporting services. When `date_before is None`,
the system runs in normal/production mode. When set, every layer that touches time-stamped
external data applies a filter or substitutes a frozen-in-time data source.

## Entry Point

CLI flag: `scout find -d <drug> --date-before YYYY-MM-DD`
([cli.py:124-134](src/indication_scout/cli/cli.py#L124-L134)).

The CLI converts the `datetime` to a `date` and forwards it to `_run_for_drug`, which forwards it
to `build_supervisor_agent(... date_before=cutoff)`
([cli.py:143-144](src/indication_scout/cli/cli.py#L143-L144),
[cli.py:69-72](src/indication_scout/cli/cli.py#L69-L72)).

Output reports for holdout runs are written under `snapshots/holdouts/`, with a
`_holdout_<ISO-DATE>` suffix in the filename and a `> **HOLDOUT** — date_before=...` banner
prepended to the markdown ([cli.py:81-89](src/indication_scout/cli/cli.py#L81-L89)).

## Per-Layer Behavior

### 1. Supervisor agent

[supervisor_agent.py:33-65](src/indication_scout/agents/supervisor/supervisor_agent.py#L33-L65)

- Loads a different system prompt when `date_before is not None`:
  `prompts/supervisor_holdout.txt` instead of `prompts/supervisor.txt`. The holdout prompt
  explicitly tells the LLM that the drug has zero FDA approvals at the cutoff unless the briefing's
  `approved_indications` says otherwise, and that the candidate that "feels obvious" is exactly the
  one being tested
  ([supervisor_holdout.txt:4-46](src/indication_scout/prompts/supervisor_holdout.txt#L4-L46)).
- Forwards `date_before` to the supervisor tools builder, which forwards it to the literature and
  clinical-trials sub-agents
  ([supervisor_tools.py:84](src/indication_scout/agents/supervisor/supervisor_tools.py#L84),
  [supervisor_tools.py:341](src/indication_scout/agents/supervisor/supervisor_tools.py#L341)).

### 2. Holdout-only investigation tool

[supervisor_tools.py:610-797](src/indication_scout/agents/supervisor/supervisor_tools.py#L610-L797)

In holdout mode the supervisor's tool list gets `investigate_top_candidates` inserted before
`finalize_supervisor`. This tool auto-runs `analyze_literature` and `analyze_clinical_trials` in
parallel for the top `SUPERVISOR_INVESTIGATION_CAP` entries (currently 3) of the merged competitor +
mechanism allowlist, removing the LLM's ability to skip the "obvious" candidate that the holdout is
specifically testing.

The auto-investigated artifacts are stashed in a closure (`auto_findings`) because they are
invoked outside the ReAct message loop. They are merged into `findings_by_disease` after the run
([supervisor_agent.py:161-180](src/indication_scout/agents/supervisor/supervisor_agent.py#L161-L180)),
with LLM-driven re-runs taking precedence.

### 3. PubMed (literature)

`date_before` is forwarded from the literature sub-agent into `RetrievalService.fetch_and_cache`
and the PubMed client
([literature_tools.py:25,75,105](src/indication_scout/agents/literature/literature_tools.py#L25)).
PubMed's E-utilities search restricts results to articles dated before the cutoff. For PMIDs that
return without a date in the search response, `_filter_pmids_by_date` reads `pub_date` from the
pgvector cache when available, then falls back to `esummary` for unknown PMIDs
([retrieval.py:392-438](src/indication_scout/services/retrieval.py#L392-L438)). Policy: missing or
unparseable dates are KEPT (matches the production filter). The `holdout_mode` flag is also passed
into downstream prompts so the LLM is told the literature window is constrained.

### 4. ClinicalTrials.gov

[clinical_trials.py:96-468](src/indication_scout/data_sources/clinical_trials.py#L96-L468),
[clinical_trials_tools.py:120-491](src/indication_scout/agents/clinical_trials/clinical_trials_tools.py#L120-L491)

Two-step filter:

1. **Inclusion filter (at fetch).** The CT.gov API request restricts to trials whose `start_date`
   is strictly before the cutoff
   ([clinical_trials.py:420-422](src/indication_scout/data_sources/clinical_trials.py#L420-L422)).
2. **Outcome scrubber (post-fetch).** A trial that started before the cutoff is correctly included
   in the holdout, but if it `completion_date` is on or after the cutoff its
   `overall_status`, `why_stopped`, and `completion_date` reflect the post-cutoff future. The
   scrubber `_scrub_post_cutoff_outcome` rewrites those fields to `UNKNOWN` / `None` so the trial
   appears as "still in progress at the cutoff"
   ([clinical_trials_tools.py:120-157](src/indication_scout/agents/clinical_trials/clinical_trials_tools.py#L120-L157)).
   Date comparisons use lexicographic ISO-prefix comparison so `YYYY-MM` and `YYYY-MM-DD`
   formats both work.

Per-tool behavior:

- `search_trials` keeps scrubbed trials in the result with `status=UNKNOWN`
  ([clinical_trials_tools.py:206-219](src/indication_scout/agents/clinical_trials/clinical_trials_tools.py#L206-L219)).
- `get_completed` and `get_terminated` DROP scrubbed trials entirely — a trial that hadn't yet
  completed/terminated at the cutoff doesn't belong in those scopes
  ([clinical_trials_tools.py:297-304](src/indication_scout/agents/clinical_trials/clinical_trials_tools.py#L297-L304),
  [clinical_trials_tools.py:388-395](src/indication_scout/agents/clinical_trials/clinical_trials_tools.py#L388-L395)).
- `get_landscape` is **disabled** entirely under holdout. The competitor-landscape aggregator rolls
  per-trial `overall_status` and `phase` across many sponsors; reconstructing that snapshot
  as-of-cutoff would require per-competitor scrubbing inside the aggregator, which isn't
  implemented. The tool returns an empty `IndicationLandscape` with an explanatory note
  ([clinical_trials_tools.py:453-467](src/indication_scout/agents/clinical_trials/clinical_trials_tools.py#L453-L467)).

### 5. FDA approvals

[approval_check.py:126-221](src/indication_scout/services/approval_check.py#L126-L221),
[clinical_trials_tools.py:493-532](src/indication_scout/agents/clinical_trials/clinical_trials_tools.py#L493-L532)

The live openFDA label path leaks today's approvals (e.g. a 2020 holdout would see semaglutide as
approved for MASH because the current label lists it). Under `date_before`:

- `check_fda_approval` swaps openFDA for a hardcoded JSON table at
  [data/drug_approvals.json](data/drug_approvals.json) consulted via
  `get_approved_indications(drug_name, candidate_diseases, as_of)`
  ([approval_check.py:157-221](src/indication_scout/services/approval_check.py#L157-L221)).
- An entry's approval is honored only if `entry.approved < as_of`.
- Matching is case-insensitive substring (a curated table entry "MASH" matches a candidate
  "non-alcoholic steatohepatitis (MASH)"). This is intentional — strict equality would require a
  full synonym map.
- Drugs not in the table return an empty set with a warning. Approval reasoning is then
  *silently disabled* for that holdout run rather than falling back to live FDA. This is by design:
  better to have no approval signal than a leaking one.
- `list_approved_indications_at` is the analogue used to seed the supervisor's drug briefing.

### 6. OpenTargets / mechanism

OpenTargets data has **no temporal filter** and is always current. This is documented in the CLI
help text ("Mechanism (OpenTargets) data has no date filter and is always current",
[cli.py:131-132](src/indication_scout/cli/cli.py#L131-L132)).

The OT competitor cache is keyed on `date_before` so cutoff and no-cutoff runs do not share cached
competitor lists ([retrieval.py:106-109](src/indication_scout/services/retrieval.py#L106-L109)),
and the OT client suppresses its current-state approved-indications strip under `date_before`.

## Cache Isolation

Any cache that survives across runs has `date_before` mixed into its key so a holdout run cannot
get a hit from a non-holdout run (or a different cutoff). Examples:

- OpenTargets competitors cache key includes `date_before.isoformat()`
  ([retrieval.py:106-109](src/indication_scout/services/retrieval.py#L106-L109)).
- ClinicalTrials.gov client request signatures include `date_before`
  ([clinical_trials.py:225,277](src/indication_scout/data_sources/clinical_trials.py#L225)).

## Summary Table

| Layer                        | Holdout behavior                                                                 |
|------------------------------|----------------------------------------------------------------------------------|
| Supervisor prompt            | Swapped to `supervisor_holdout.txt`                                              |
| Supervisor tool list         | `investigate_top_candidates` inserted, auto-investigates top `SUPERVISOR_INVESTIGATION_CAP` (currently 3) allowlist |
| PubMed search                | `date_before` filter on E-utilities; pgvector + esummary date check on results   |
| CT.gov fetch                 | Restrict to `start_date < cutoff`                                                |
| CT.gov outcome fields        | Trials completing on/after cutoff have status/why_stopped/completion_date scrubbed |
| `get_completed`/`get_terminated` | Scrubbed trials dropped                                                       |
| `get_landscape`              | Disabled (returns empty)                                                         |
| FDA approval                 | Hardcoded JSON table, gated on `approved < as_of`; missing drug → silent disable |
| OpenTargets / mechanism      | Not date-filtered; cache keyed on `date_before` to keep results separate         |
| Output                       | Written under `snapshots/holdouts/<drug>_holdout_<DATE>_<TS>.md` with banner     |

## Design Rule

The methodology follows the project's accuracy-vs-coverage stance: **error by omission is
acceptable; inaccurate output is not.** When a holdout-correct data source isn't available
(landscape aggregation, FDA approvals for an uncurated drug), the system disables that signal
rather than substituting a leaking one.
