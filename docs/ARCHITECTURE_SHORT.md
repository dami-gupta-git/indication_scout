# IndicationScout Architecture (Short)

## Overview

IndicationScout is an agentic drug repurposing system. A drug name goes in; coordinated AI agents query six biomedical data sources (Open Targets, ClinicalTrials.gov, PubMed, Europe PMC, ChEMBL, openFDA) and produce a ranked report of candidate diseases the drug might treat beyond its approved uses. Each candidate comes with its mechanism of action, clinical trial history, literature evidence, and safety signals, and the supervisor writes a narrative summary and short structured write-ups for the top candidates.

The system runs as a command-line tool and as a FastAPI web service with a React frontend. It is written in Python with async I/O throughout.

## Layers

```
CLI / API ──> Supervisor agent ──> Literature, Clinical Trials, Mechanism sub-agents
                    │                              │
                    └──────────── Services ────────┤
                                                   ▼
                                      Async data source clients
                                      (Open Targets, ClinicalTrials.gov,
                                       PubMed, Europe PMC, ChEMBL, openFDA)
                                                   │
                                                   ▼
                                   Pydantic models at every module boundary
```

The codebase is organised into these areas:

| Area | Contents |
| --- | --- |
| Data sources | One async client per external API, all built on a shared base client |
| Models | Pydantic data contracts per source (targets, drugs, trials, abstracts, articles, molecules, drug profiles, evidence summaries) |
| Services | Business logic: LLM calls, embeddings, disease normalisation, PubMed query building, retrieval and synthesis, drug safety, FDA approval checks, development-stage judgement, the shared analysis runner, job tracking, and progress reporting |
| Agents | The supervisor and three sub-agents, plus shared trial-formatting and trial-signal helpers |
| Prompts | LLM prompt templates stored as text files |
| Report | Renders the supervisor's structured output as markdown |
| API | FastAPI app with health, analysis, drill-down and example-report routes |
| CLI | The `scout` command |
| Regression | Snapshot harness that diffs two saved reports |
| Database | SQLAlchemy models, including a PubMed abstracts table with pgvector embeddings |

Agents never see raw API responses. Every client parses its response into typed Pydantic models, and every model that ingests external data converts null fields to empty values so downstream code needs no null guards.

## Analysis Flow

1. **Drug intake.** The drug name is normalised and resolved to its ChEMBL ID, synonyms, first approval year, and FDA-approved indications. This runs once per analysis as a shared task, and both candidate-discovery paths wait on the same result.
2. **Candidate discovery.** Candidates come from two sources: diseases pursued by competitor drugs that share the drug's targets, and diseases linked to those targets through the mechanism agent. The lists are merged, duplicate disease names are collapsed, and diseases the drug is already approved for are removed. The result is an allowlist of diseases the supervisor may report on.
3. **Mechanism analysis.** The mechanism agent scores target–disease associations and keeps diseases where the drug's effect opposes the disease driver.
4. **Investigation.** For the top candidates, the supervisor runs the literature and clinical trials agents. In fan-out mode it investigates the top candidates in parallel.
5. **Ranking.** The supervisor drafts a ranking, runs a mandatory critique of the ranking, and finalises with a summary and blurbs for the top candidates.
6. **Assembly.** Plain Python walks the agent's message history, matches disease names against the allowlist, and builds the final structured output. The report renderer turns it into markdown.

## Agents

All agents follow a reason-and-act pattern: the model calls tools, reads short summaries of their results, and keeps going until it calls its finalise tool.

### Supervisor

The supervisor wraps each sub-agent as a tool and orchestrates the run.

| Tool | Purpose |
| --- | --- |
| Find candidates | Surface competitor and mechanism candidates and merge them |
| Analyse mechanism | Run the mechanism sub-agent |
| Analyse literature | Run the literature sub-agent for one disease |
| Analyse clinical trials | Run the clinical trials sub-agent for one disease |
| Investigate top candidates | Run literature and trials over the top candidates in parallel (holdout or fan-out mode only) |
| Get drug briefing | Read the drug-level facts gathered so far |
| Critique ranking | Audit the draft ranking order; required before finalising |
| Finalise | Return the narrative summary and top-candidate blurbs |

In fan-out mode the per-disease literature and trials tools are removed, so the supervisor must use the parallel path.

### Sub-agents

| Agent | Tools | Output |
| --- | --- | --- |
| Literature | Build drug profile, expand search terms, fetch and cache abstracts, semantic search, safety search, synthesise, finalise | Evidence summary with supporting and contradicting papers, evidence strength and direction, and safety findings |
| Clinical trials | Check FDA approval, search trials, get completed trials, get terminated trials, get landscape, finalise | Trial record, phase distribution, terminations, competitors, and approval status for the disease |
| Mechanism | Get drug, get target associations, finalise | Drug targets, mechanisms of action, and ranked candidate diseases |

The literature agent embeds PubMed abstracts with BioLORD-2023, a biomedical sentence-embedding model, stores them in Postgres with pgvector, and retrieves the most relevant ones by semantic similarity. Each retrieved paper is graded on its own for direction (supports or contradicts) and study design before the evidence is synthesised. Reports of the same trial are merged, and the synthesis set is capped.

The clinical trials agent classifies every trial it was shown before it can finalise. Deterministic helpers compute trial facts such as the highest completed phase and whether a Phase 3 trial was terminated for cause.

### Mechanism Scoring

The mechanism agent labels each target–disease pair as loss-of-function driven, gain-of-function driven, or inconclusive by majority vote over the direction annotations in Open Targets evidence, with an 80 percent threshold. It keeps only pairs where the drug's action opposes the driver: an inhibitor against a gain-of-function disease, or an activator against a loss-of-function disease. Diseases that are already FDA-approved for the drug, or on a blocklist of overly broad terms, are dropped. The rest are sorted by score and trimmed to a configured count.

The agent also reproduces Open Targets' overall association score locally. The published score is a weighted harmonic sum over per-datasource scores, and the local version matches it to about 0.01 mean absolute error. This allows one datasource to be removed for holdout runs (see Holdout Mode).

### Agent Loops

Two loop types are in use. The supervisor and clinical trials agents run on a custom gated loop: their finalise tool can fail validation, for example when the critique has not run or a trial is unclassified, and the loop sends the model back to fix it. The literature and mechanism agents have no reject path, so they use LangGraph's prebuilt agent with the finalise tool returning directly. Both loops end the moment finalise succeeds, which avoids an extra model turn.

## Grounding

Each tool returns two things: a short text summary the model reads, and a typed data object the model never sees. After the loop ends, plain Python reads the typed objects off the tool messages and assembles the output. This was chosen over having the model produce structured output at the end of the run, which would require it to retype NCT IDs, PMIDs and scores from memory, where a transposed digit becomes a wrong citation. It also avoids the extra model turn that structured output needs.

Deterministic code bounds the model's judgements. The development-stage judgement has a floor: if the trial record proves a higher stage than the model proposed, such as an active Phase 3 trial when the model said early phase, the code raises it. It never lowers a stage. In literature synthesis, if the evidence is not specific to the drug, the code forces evidence strength and direction down to none.

Each candidate's relationship to the drug's FDA label is decided once upstream and carried through every layer. The values are approved, contaminated (overlapping an approved use), combination only, or none.

## Safety

The literature agent's safety search produces two tiers of signal.

**Drug-wide.** The query uses PubMed's major-topic adverse-effects tag for the drug, which finds papers primarily about its harms without needing to know the event names. It falls back to a title and abstract search when the tags are sparse. Results are ranked by Europe PMC citation count, because PubMed's relevance ranking buries landmark safety papers: the APPROVe trial on rofecoxib ranks outside PubMed's top 300 for a toxicity query despite over 1,600 citations. The assessment keeps FDA boxed warnings (from the latest label version), Open Targets safety rows, adverse-event reports, and literature as separate sources, and treats adverse-event report counts as associations, not causes.

**Disease-specific.** The query combines the drug, adverse-event vocabulary, and the candidate disease, which recovers indication-specific papers the drug-wide pool misses. Each paper is classified as confirmed harm, safety assessed only, irrelevant, or unclear. Confirmed harm requires a named adverse outcome and an exact quote found in the title or abstract. Missing evidence, incomplete model output, or an unverified quote produces an unavailable result, and a reviewed negative is not reported as evidence of safety.

The drug-wide assessment appears once in its own report section. A confirmed disease-specific harm appears as a warning on that candidate and is passed to the ranking critique, which can move it down. A historical drug-wide withdrawal is not passed to the critique, because the model misread it as ruling out every candidate disease.

## Holdout Mode

Holdout mode runs the pipeline as of a past cutoff date so its predictions can be compared with what happened later. The cutoff is passed to the literature and clinical trials agents. PubMed's own date filter relies on an unreliable date field, so results are checked again against each paper's sort date. Current labels, Open Targets safety rows, and current pharmacovigilance data are excluded from safety analysis.

Open Targets has no date-filtered API, so mechanism analysis always uses current data. Its clinical-precedence datasource encodes current trials and approvals and would leak post-cutoff information. Holdout runs rank by the locally recomputed score with that datasource removed; production runs rank by the published score.

## Data Clients

All clients extend a shared async base client that provides GraphQL, JSON REST, and XML REST requests. It retries rate-limit and server errors (429, 500, 502, 503, 504) with exponential backoff, up to three retries capped at 30 seconds. Clients raise a dedicated data-source error that names the source and context, and a request that still fails is appended to a failure log.

| Client | Interface | Provides |
| --- | --- | --- |
| Open Targets | GraphQL | Drug data, targets, disease associations, evidence, competitor drugs |
| ClinicalTrials.gov | REST v2 | Trial search, completed and terminated trials, indication landscape |
| PubMed | E-utilities (REST and XML) | Search, abstracts, MeSH subject tags |
| Europe PMC | REST | Citation counts, drug-scoped literature search |
| ChEMBL | REST | Drug name resolution, synonyms, ATC drug classes |
| openFDA | REST | Drug labels, approved indications, boxed warnings |

## Caching

API and LLM responses are cached on disk as JSON, one folder per namespace, keyed by a SHA-256 hash of the namespace and request parameters. Each entry stores its data, timestamp, and expiry; the default expiry is configured (currently 60 days), and expired or corrupt entries are deleted when read. Namespaces cover each data source and the LLM-backed services such as disease normalisation, synthesis, and approval checks. Open Targets also keeps per-target evidence files, and ChEMBL keeps per-drug name files.

## API and CLI

The FastAPI app exposes a health check, asynchronous analysis jobs (start, poll, fetch the markdown report, delete), drill-down views, and example reports. Jobs are held in an in-memory store and report progress by phase: candidates, mechanism, trials, literature, summary. In production the app also serves the built React frontend and logs visitors and bots.

The `scout` CLI has four commands:

| Command | Purpose |
| --- | --- |
| find | Run the full pipeline for a drug, optionally as of a cutoff date |
| investigate | Run the pipeline on a fixed drug and disease pair without candidate discovery |
| render | Re-render a saved result to markdown without rerunning |
| diff-report | Compare two saved results with the regression harness |

The CLI and API share one analysis runner, which normalises the drug name, manages the database session, and passes the cutoff date through.

## Configuration and Observability

Settings load through pydantic-settings from two files: one for secrets, database credentials, and model names, and one for tunable numeric limits. Environment variables override both. The limits have no built-in defaults and must be set; they cover timeouts and retries, PubMed batch sizes, concurrency for LLM, PubMed and per-disease work, candidate and investigation caps, and mechanism thresholds. Settings are frozen once loaded.

The system uses Anthropic Claude models in large and small tiers. Optional OpenTelemetry tracing sends agent and model calls to Langfuse.

## Testing

Unit tests run without network access and mirror the source tree. Integration tests hit the live APIs. A regression harness stores report snapshots and diffs new runs against them to catch changes in results. Code is formatted with black, linted with ruff, and type-checked with mypy.

## Current State

All six data clients, the data models, the services, the four agents, the API, and the CLI are complete. A literature-based candidate source is built and tested but not yet wired in: it extracts the conditions each Europe PMC abstract says the drug treats and groups them into candidate indications.

## Design Principles

Missing scientific or clinical values stay empty and are never filled with defaults. Missing a real candidate is acceptable; reporting an unsupported one is not, so allowlist checks and reject paths are never loosened to rescue a candidate that was filtered out upstream.
