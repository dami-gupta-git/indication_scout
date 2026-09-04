# Architecture Overview

## What the system does

IndicationScout is an agentic drug repurposing system. A drug name goes in; a coordinated set of
LLM agents queries biomedical data sources (Open Targets, ClinicalTrials.gov, PubMed, Europe PMC,
ChEMBL, openFDA) and produces a report ranking candidate new indications for that drug, each backed
by mechanism, literature, and clinical-trial evidence.

A supervisor agent owns the run. It surfaces candidate diseases from two seed sources, delegates
per-disease investigation to three specialist sub-agents (mechanism, literature, clinical trials),
audits its own draft ranking, and emits a structured output that is rendered to a markdown report.
Every claim in the report traces back to an upstream data source; the guiding rule is accuracy over
coverage — missing a real candidate is acceptable, surfacing an ungrounded one is not.

## How a run flows

```
 CLI / API
    │
    ▼
 analysis runner  (normalizes drug name, owns DB session, threads holdout cutoff)
    │
    ▼
 Supervisor agent ──────────────────────────────────────────────┐
    │  1. find candidates (competitor seeds + mechanism seeds)  │
    │  2. label each candidate against FDA approvals            │
    │  3. investigate top candidates (fan-out)                  │
    │       ├── literature sub-agent                            │
    │       └── clinical-trials sub-agent                       │
    │  4. critique the draft ranking (mandatory)                │
    │  5. finalize → structured output                          │
    ▼                                                           │
 Services  (retrieval/RAG, disease normalization, approval      │
            checks, dev-stage judging, LLM wrappers)            │
    │                                                           │
    ▼                                                           ▼
 Data source clients (async, cached)          Pydantic models carry all data
    │                                         across module boundaries; agents
    ▼                                         never see raw API responses
 External APIs
 (Open Targets · ClinicalTrials.gov · PubMed · Europe PMC · ChEMBL · openFDA · Anthropic)
```

The final output is a single structured object: the candidate disease list, per-disease findings
(mechanism, literature, trials), a ranked top-5 with per-candidate blurbs, and a narrative summary.
A report module renders it to markdown; the API can also return it as JSON.

## Layers

| Layer | Directory | Responsibility |
|-------|-----------|----------------|
| Data sources | `data_sources/` | One async client per external API, all sharing a base client with retry/backoff and error wrapping |
| Models | `models/` | Pydantic contracts between data sources and everything downstream |
| Agents | `agents/` | The supervisor and the three specialist sub-agents, each with its own tools and typed output |
| Services | `services/` | Business logic: retrieval/RAG, disease normalization, approval checking, LLM access, embeddings, the shared run entry point |
| API | `api/` | FastAPI app with an async job model; serves the built React frontend in production |
| CLI | `cli/` | The `scout` command: run the pipeline, re-render a saved report, diff two reports |
| Report | `report/` | Structured output → markdown |
| Config | `config.py`, `constants.py` | Settings from two env files (secrets vs. tunable limits); all magic numbers live in constants |

## Data sources

| Source | Protocol | What it contributes |
|--------|----------|---------------------|
| Open Targets | GraphQL | Drug record (targets, mechanisms, indications, warnings, adverse events) and per-target disease associations with evidence scores |
| ClinicalTrials.gov | REST | Trial counts and exemplars per drug × disease pair, plus a competitive landscape per indication |
| PubMed | REST + XML | Literature search and abstract retrieval |
| Europe PMC | REST | Citation counts (for ranking safety papers) and drug-scoped literature search |
| ChEMBL | REST | Drug-name resolution to a ChEMBL ID (the sole drug identifier) and all known names/synonyms |
| openFDA | REST | Drug labels, used for approval extraction |
| Anthropic | REST | All LLM calls: a main model for agent loops, a small model for high-volume classification |

Every client is an async context manager built on a shared base that retries transient failures
with exponential backoff and raises a single error type carrying the source name and context.
All calls go through a shared disk cache (JSON files keyed by a hash of the request, per-namespace
directories, config-driven TTL), so repeated runs on the same drug are mostly cache hits.

## Agents

All four agents are ReAct-style: the LLM calls tools in a loop until it calls its finalize tool.
Two loop implementations are in use. The supervisor and clinical-trials agents run on a custom
gated loop whose finalize tool can reject (for example, when the ranking critique has not run, or
when not every shown trial was classified) and send the model back to retry; the loop ends the
moment finalize succeeds. The literature and mechanism agents have finalize tools that are pure
termination signals, so they use LangGraph's prebuilt ReAct agent.

Tools follow one convention throughout: each returns a short human-readable string the LLM reads,
plus a typed Pydantic object the LLM never sees. After the loop ends, plain code walks the message
history and assembles the typed objects into the agent's output. Exact identifiers (NCT IDs, PMIDs,
association scores) therefore always come from tool return values, never from the model restating
them from memory.

### Supervisor

The supervisor wraps each sub-agent as a tool. Its run has a fixed skeleton: surface candidates,
run the mechanism analysis, investigate the top candidates, critique the draft ranking (finalize is
rejected until the critique has run), then finalize with a narrative summary and top-5 blurbs. In
fan-out mode the per-candidate literature and trials tools are replaced by a single parallel
investigation tool, so the top candidates are investigated concurrently. After the loop, assembly
code canonicalises disease names against the merged candidate allowlist — a disease the supervisor
names that is not in the allowlist is rejected and logged, never "tried anyway".

### Mechanism

Pulls the drug record and per-target disease associations from Open Targets, then scores and trims
candidates in deterministic post-LLM code: associations are thresholded, filtered against the
drug's already-approved diseases, and capped to a configured count. This agent is also one of the
two candidate seed sources for the supervisor.

### Literature

Runs a RAG pipeline over PubMed: build a drug profile, expand search terms, fetch and store
abstracts (Postgres with pgvector embeddings, BioLORD biomedical embedding model), semantic-search
the stored abstracts, then synthesize an evidence summary for the drug × disease pair. The summary
grades evidence strength and direction independently, buckets every PMID (supporting,
contradicting, neutral, contaminated) in code from a per-paper verdict map, and records the
evidence basis — whether the signal is drug-specific, drug-class-level only, or confined to an
already-approved sub-indication. A separate safety search adds a two-tier safety signal: a
drug-level summary (regulatory warnings plus citation-ranked literature) and a per-indication harm
flag.

### Clinical trials

Queries ClinicalTrials.gov per drug × disease pair: overall counts by status, completed trials,
terminated trials (with stop reasons), and the competitive landscape for the indication. The
disease side of every query is resolved to a MeSH descriptor first and filtered server-side, which
keeps free-text noise out. An FDA approval check tool tells the agent whether the pair is already
approved. The agent must classify every trial it was shown as relevant or contamination before its
finalize is accepted.

## Keep the output grounded

Several mechanisms enforce the accuracy-over-coverage rule:

- **Allowlist at assembly.** The candidate list built during surfacing is the only set of diseases
  that can appear in findings. Anything else the model proposes is dropped and logged.
- **Approval labeling upstream.** Each candidate's relationship to the drug's existing FDA
  approvals is decided once, from the label (curated short-circuit list first, then a
  label-grounded LLM call), and threaded down read-only: already-approved candidates are dropped,
  combination-only approvals are demoted, and "contaminated" candidates — real targets whose trial
  counts are polluted by an approved narrower subset — are kept but have their trial tables
  suppressed in the report. The model never authors this label.
- **Approved sub-indication exclusion.** Because the trial registry's MeSH filter matches
  ancestors, a broad candidate query can pull in evidence for an approved child indication. The
  drug's approved-indication list is threaded into both the per-trial relevance gate and the
  literature judge so that evidence cannot count toward the broader candidate. Siblings of an
  approved indication and genuinely broader populations still count.
- **Floor/cap pattern.** Where an LLM judgment coexists with deterministic facts, code clamps the
  judgment: the development-stage tier is floored by what the trial record proves, and literature
  strength is forced down when the evidence basis is not drug-specific. The model proposes; code
  never lets it contradict the data.
- **No fabricated absence.** When a signal is missing (no label found, no safety papers), fields
  come back empty — never a synthesized "no concerns found".

## Holdout mode

The pipeline can run as-of a past date to validate predictions against what happened later. A
cutoff date is threaded to the literature and trials agents, which filter their queries by it. Open
Targets has no date filtering, so two compensations apply: the association ranking is recomputed
locally with the clinical-precedence channel removed (that channel encodes current trials and
approvals and would leak the future), and undateable regulatory safety data is omitted in favour of
date-filtered literature. A validation script runs the pipeline over a runbook of known
drug → later-approved-indication pairs and scores whether each shows up.

## Entry points

The CLI runs a full analysis (`scout find`), a single fixed drug–disease investigation
(`scout investigate`), a re-render of a saved structured output (`scout render`), and a diff of two
saved outputs via the regression harness (`scout diff-report`). Reports are written as markdown
snapshots plus a structured JSON dump.

The API exposes the same pipeline as async jobs: a POST starts an analysis, polling GETs return
progress (the pipeline emits named phase events) and the finished report. An in-memory job store
backs this. In production the app also serves the built React frontend and can serve committed seed
reports for example drugs instead of running the pipeline.

## Testing

Unit tests (no network) and integration tests (real APIs) mirror the source tree; a snapshot
regression harness compares structured outputs and backs the report diff command. Async tests run
under pytest with automatic asyncio mode.

## Not yet wired

Europe PMC candidate sourcing — extracting treated conditions from drug-scoped abstracts and
grouping them into candidate indications — is implemented and tested but not yet called by any
agent. See `design_europe_pmc.md`.
