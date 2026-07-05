# Agent Design

IndicationScout uses one supervisor agent and three specialist sub-agents. Every agent is a
LangGraph `create_react_agent` (ReAct loop) with `@tool(response_format="content_and_artifact")`
tools, so each tool returns both an LLM-visible content string and a typed Pydantic artifact.
After the loop ends, `run_<name>_agent` walks the message history and pulls each tool's
artifact off `ToolMessage.artifact` to assemble the agent's typed output. The legacy
`BaseAgent` ABC in `agents/base.py` is unused.

## Topology

```
                 ┌─────────────────────────────────────┐
                 │          Supervisor agent            │
                 │   (orchestrator + ranker + briefer)  │
                 └────┬─────────────┬──────────┬───────┘
                      │             │          │
       find_candidates│  analyze_mechanism      │ analyze_literature
                      │             │          │ analyze_clinical_trials
                      ▼             ▼          ▼
              Open Targets    Mechanism     Literature       Clinical-Trials
              competitors     sub-agent     sub-agent        sub-agent
```

The supervisor wraps each sub-agent as a single tool. Sub-agents are **not** invoked through
LangGraph subgraphs — they are awaited directly from the supervisor's tool functions.

## Supervisor (`agents/supervisor/`)

Orchestrates the run, manages a per-run **disease allowlist**, and writes the final summary.
Tools: `find_candidates`, `analyze_mechanism`, `analyze_literature`, `analyze_clinical_trials`,
`get_drug_briefing`, `finalize_supervisor` (+ `investigate_top_candidates` in holdout mode).

Two pieces of closure-scoped state coordinate the run:

- **Allowlist** (`allowed_diseases`, `allowed_efo_ids`): merged competitor + mechanism
  candidates, keyed lowercase → `(canonical_name, source∈{competitor, mechanism, both})`.
  `analyze_literature` / `analyze_clinical_trials` reject any disease not in the allowlist
  (prevents the LLM from inventing diseases or rewording them).
- **Seed gates** (`asyncio.Event` for `find_candidates_done` and `analyze_mechanism_done`):
  investigation tools `await` both before running, so parallel tool calls can't race a
  half-populated allowlist. Both events fire in `try/finally` so a sub-agent crash never
  deadlocks downstream tools.

A separate `drug_facts` store accumulates ChEMBL aliases, FDA-approved indications, mechanism
targets, and mechanism→disease associations as tools run. `get_drug_briefing` renders this
store as markdown so the supervisor can reason about subset/superset/sibling relationships
between candidates and approved indications before finalizing.

`run_supervisor_agent` then canonicalises disease names against the allowlist and assembles a
`SupervisorOutput` (drug, candidates, mechanism artifact, per-disease findings, summary).

## Mechanism sub-agent (`agents/mechanism/`)

Tools: `get_drug` → `get_target_associations` → `finalize_analysis`. The agent loop only fetches
data; the ranking is **deterministic, post-LLM**:

1. `get_drug` resolves the drug to a ChEMBL ID and returns its `MechanismOfAction` list.
2. `get_target_associations` (called per target) fetches Open Targets associations, filtered by
   `mechanism_signal_threshold` over `MECHANISM_SIGNAL_KEYS`.
3. After the loop, `_assemble_candidates` builds per-target rows, fetches `EvidenceRecord`s,
   and calls `mechanism_candidates.select_top_candidates`, which:
   - Aggregates `direction_on_target` × `direction_on_trait` by majority vote
     (`_MAJORITY_THRESHOLD = 0.8`) to label each (target, disease) pair as LoF-driven /
     GoF-driven / inconclusive.
   - Keeps only **POSITIVE** rows where the drug's action direction *opposes* the
     disease-driving direction (LoF drug ↔ GoF-driven, GoF drug ↔ LoF-driven).
   - Drops rows whose disease is FDA-approved (via `get_fda_approved_disease_mapping`) or in
     `BROADENING_BLOCKLIST`.
   - Sorts by `ranking_score` (OT's `overall_score` in production; a leak-free recomputed
     score that drops `clinical_precedence` in holdout mode), trims to
     `MECHANISM_TOP_CANDIDATES`.

Output: `MechanismOutput` with `drug_targets`, `mechanisms_of_action`, `candidates`
(`MechanismCandidate` rows), `summary`.

## Literature sub-agent (`agents/literature/`)

A RAG pipeline over PubMed driven by tool order:

`build_drug_profile` → `expand_search_terms` → `fetch_and_cache` → `semantic_search` →
`synthesize` → `finalize_analysis`.

Tools share intermediate results through a closure-scoped `store` dict (drug profile, queries,
PMIDs, abstracts) so the LLM never has to round-trip large payloads. `fetch_and_cache` writes
abstracts + BioLORD-2023 embeddings into Postgres/pgvector via `RetrievalService`;
`semantic_search` re-ranks against the drug+disease query embedding; `synthesize` produces an
`EvidenceSummary` with a self-judged strength label (`strong` / `moderate` / `weak` / `none`).
The `date_before` cutoff propagates through `fetch_and_cache` and switches `synthesize` into
holdout mode. Output: `LiteratureOutput`.

## Clinical-Trials sub-agent (`agents/clinical_trials/`)

Tools (all called in parallel, no inter-tool dependencies):
`check_fda_approval`, `search_trials`, `get_completed`, `get_terminated`, `get_landscape`,
`finalize_analysis`. Every indication-filtered call resolves the indication → MeSH D-number via
`disease_helper.resolve_mesh_id` and post-filters CT.gov results by `mesh_conditions` /
`mesh_ancestors` (works around CT.gov's recall-first Essie engine). Trial interventions are
also filtered by ChEMBL drug aliases (whole-word match) so observational studies that merely
mention the drug don't pollute the pair count. Stop categories (safety/efficacy/business/
enrollment) are derived at the tool layer from `why_stopped` text. Output: `ClinicalTrialsOutput`
(search / completed / terminated / landscape / approval / summary).

## Termination & holdout mode

Every sub-agent's prompt mandates `finalize_analysis` as the last call — plain-text AIMessages
after the loop are discarded. The supervisor mirrors this with `finalize_supervisor`.

When `supervisor_fanout` is on, the supervisor appends a fan-out directive to `supervisor.txt`
(the prompt file is the same with the flag off or on) and gains `investigate_top_candidates`,
which fans out `analyze_literature` + `analyze_clinical_trials` over the top
`supervisor_investigation_cap` allowlist entries in parallel — bypassing the LLM's tendency to
skip "obvious" candidates that holdout evaluations are designed to recover. Those tool calls
happen outside the ReAct loop, so their artifacts are stashed in a closure (`auto_findings`)
and merged into `SupervisorOutput` by `run_supervisor_agent`. `date_before` is a separate axis:
it forwards the holdout cutoff to the literature and clinical-trials sub-agents.
