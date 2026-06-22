# scripts/

Standalone developer scripts: testers, probes, seeders, benchmarks, and
tracing/eval utilities. None are part of the installed package; run them directly
(most expect the project venv and `.env` / `.env.constants`).

## Pipeline / agent testers

- **agent_tester.py** — Scratch harness for driving the supervisor agent for a
  single drug end-to-end (mostly commented-out experiments).
- **pipeline_tester.py** — Ad-hoc retrieval/term-expansion pipeline experiments
  (PubMed fetch + RAG caching) for specific drug/disease pairs.
- **inspect_trastuzumab_profile.py** — Prints a built `DrugProfile` (name, type,
  ATC, targets, MoA, synonyms) for a drug, to eyeball profile construction.
- **drug_trial_sweep.py** — Exploratory drug-only CT.gov sweep (`query.intr=<drug>`,
  no indication filter); prints an indication histogram. Not allowlist-filtered.

## Data source / API probes

- **fda_api_tester.py** — Hits the openFDA FAERS API and inspects raw responses.
- **fetch_competitors_for_tests.py** — One-off: fetches `get_drug_competitors`
  output for known drugs to pin exact integration-test assertions.
- **fetch_pmids_for_tests.py** — One-off: fetches PMIDs for overlapping queries to
  seed `fetch_and_cache` integration tests.

## Database / seeding

- **seed_db.py** — Seeds the database with initial data.
- **truncate_db.py** — Truncates all user tables in the public schema (excludes
  `alembic_version`), CASCADE.
- **seed_examples_from_reports.py** — Populates `seed_examples/` from the latest
  saved `test_reports/{drug}_{ts}.json` payloads, validating each as
  `SupervisorOutput` and recording capture times.
- **prefetch_embedding_model.py** — Downloads and caches the BioLORD-2023 embedding
  model into the HF cache so the app's first request is instant.

## Tracing / observability (Langfuse + OTel)

- **otel_hello.py** — Throwaway "hello span" proving the raw OTel → OTLP → Langfuse
  pipeline works (no LangChain/OpenInference).
- **otel_hello_llm.py** — Step 2: auto-instruments one real LangChain LLM call and
  watches its span (model, token counts, latency) land in Langfuse.
- **trace_summary.py** — Collapses one Langfuse trace into a per-agent LLM
  cost/token/call table. Reads Langfuse creds from `.env`.

## Evaluation / benchmarks

- **run_evaluation.py** — Runs model evaluation.
- **spike_astream.py** — Throwaway spike confirming `astream` reproduces `ainvoke`'s
  messages for the supervisor agent (read-only).
- **trailing_turn_bench.py** — Before/after benchmark for the trailing-turn removal:
  warm-up run + one timed run, capturing total time and per-agent turn counts.

## Misc

- **session.py** — Session-file manager for IndicationScout (see its docstring for
  usage and rules).

## validation/

Holdout-validation harnesses for checking the pipeline against known approvals;
read against `runbook.txt`. Outputs go to `results/holdout_validation/`.

- **gen_seed_candidate_recall.py** — Holdout recall harness. Runs only the seed
  phase (mechanism + competitor surfacing + merge) per runbook row and emits a
  binary presence score (1 = known indication in the merged candidate list,
  0 = absent) by LLM-matching against the leak-free merged allowlist. Writes
  `results/holdout_validation/validation_results_N.md`. (Replaced the former
  `run_validation.py`, which drove the full `scout find` pipeline.)
- **probe_rank.py** — Probes whether the CT.gov candidate sweep catches each
  approval and at what rank, across several ranking strategies. Writes
  `results/holdout_validation/probe/probe_rank_results.md`.
- **probe_candidates.py** — Probes the competitor-candidate stage standalone (no
  LLM agent) to debug a missed disease; reports whether/where a target survived
  the OT ranking + merge.
- **probe_mechanism.py** — Runs only the mechanism agent for a few drugs (cheap A/B
  for the target fan-out); prints fetched-target counts and final candidates.
- **langfuse_runs.py** — Reports the most recent supervisor pipeline runs from
  Langfuse (timestamp, drug, HEAD commit, latency, cost, cache-hit ratio).

### validation/ data files

- **runbook.txt** — CSV (`drug,indication,date`) of known approvals; input for
  `gen_seed_candidate_recall.py` and `probe_rank.py`.
- **drug_approvals.json** — Per-drug `{disease, approved}` approval records.
