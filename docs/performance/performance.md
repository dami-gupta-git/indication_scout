# Performance

Measurements and changes from the June 2026 speed work, plus later optimizations. Numbers come from Railway `[TIMING]`
logs and local measured runs unless noted. Run-to-run LLM latency noise is about ±15s even at temperature 0, so
differences under ~15s are not measurable end-to-end.

## Where the time goes

### Full-run totals (`[TIMING] run_analysis ... total`, 2026-06-10/11)

| Drug | Total | Cache | External API | Source |
|---|---|---|---|---|
| lithium | 360.6s | cold | 167s (46%) | Railway |
| bupropion (run 1) | 174.7s | cold | 119s (68%) | Railway |
| bupropion (run 2) | 122.6s | warm | 3.7s (3%) | Railway |
| lisinopril (run 1) | 263.0s | cold | 42.7s (16%) | Railway |
| lisinopril (run 2) | 107.7s | warm | 6.3s (6%) | Railway |
| gabapentin | 236.2s | partial | — | Railway |
| metformin (run 1) | ~7 min | cold (volume unmounted) | 1483s OT / 1438 calls | Railway |
| metformin (warm) | 124.7s | warm | 10.6s (9%) | Railway |
| metformin (local, fat-tool) | 123.9 / 140.6s | warm | — | local |
| metformin (local, baseline) | 131.3s | warm | — | local |

Warm floor ≈ 110–140s, cold ≈ 175–360s. The spread is driven by cache state, not the drug. Warm, external API
drops to 3–10% of the run and the cost is LLM generation.

### Phase breakdown (warm metformin, job 81337a10)

| Phase | Time | Notes |
|---|---|---|
| analyze_mechanism | 27–30s warm (132s cold) | agent loop; `_assemble_candidates` 0.6s warm / ~190s cold (51 targets) |
| investigate_top_candidates (3 diseases ∥) | 38–49s | clinical_trials or literature, drug-dependent |
| critique_ranking | 1–10s | one supervisor LLM call |
| supervisor glue + finalize | remainder | supervisor's own ReAct turns |

### Sub-agent isolation (local, warm, pre-fix code)

| Component | Warm time | Composition |
|---|---|---|
| literature pure pipeline (no agent) | 1.5s | embeddings ~1.3s of it |
| literature agent (7-turn ReAct) | ~24s (up to 87s for some drugs) | LLM round-trips, not data |
| clinical_trials data (5 tools, parallel) | 1.0s | |
| clinical_trials agent | 48.7s | see per-turn split |
| mechanism agent loop | 27–68s | high LLM-latency variance |

clinical_trials per turn (warm, metformin × PCOS, 50.5s). Per-turn times are estimated by splitting wall-time by
output-token share, not instrumented per LLM call.

| Turn | Tool(s) | in_tok | out_tok | est. time |
|---|---|---|---|---|
| 1 | all 5 data tools (batched) | 2,666 | 315 | ~5s (data ~1s) |
| 2 | finalize_analysis (summary) | 7,588 | 1,169 | ~23s |
| 3 | (final) trailing — discarded | 8,771 | 1,011 | ~22s |

The discarded trailing turn existed on every agent (clinical_trials 1,011 tokens / ~22s, supervisor 894 / ~18s,
literature 362–818 / ~8–16s, mechanism 204–431 / ~5–9s); summed over a run it was plausibly 60–90s of waste.

### Embedding cost (local CPU, BioLORD-2023)

| Workload | Time |
|---|---|
| 100 abstracts (1 batch) | 1.3s |
| 3×100 sequential | 3.9s |
| 3×100 "parallel" (embed_async) | 3.8s |

Embeddings serialize across diseases (CPU/GIL-bound; `asyncio.to_thread` gives no real parallelism) but the
absolute cost is small. Not a bottleneck; the hypothesis was tested and rejected.

### Controlled warm benchmark (2026-06-11, `scripts/controlled_timing_bench.py`)

Same baseline code (literature not fat-tool), caches pre-warmed, local, one run per drug. literature and
clinical_trials columns show the slowest single disease (they run in parallel).

| Drug | Total | mechanism | literature (slowest) | clinical_trials (slowest) | critique |
|---|---|---|---|---|---|
| metformin | 132.6s | 28.7s | 31.2s | 45.9s | 1.4s |
| lisinopril | 159.3s | 25.2s | 86.7s | 45.0s | 2.0s |

clinical_trials is a stable ~45s floor; literature varies 31s–87s by drug, so the warm bottleneck is
drug-dependent. Total ≈ mechanism + max(lit, CT) + critique + supervisor glue.

## Findings

1. Cache state dominates everything. On Railway the cache was writing to ephemeral image disk because no volume was
   mounted at `/data/cache`; fixed by mounting the volume at `/cache` and pointing `SCOUT_CACHE_DIR`/`HF_HOME` at it.
2. Warm, the cost is LLM generation, not data or embeddings.
3. The discarded trailing `(final)` turn was the largest behavior-preserving lever.
4. clinical_trials cost is generation, not turn count — its five data tools were already batched into one turn, so a
   fat-tool refactor would not help it.
5. The literature fat-tool (collapsing 7 turns to 2) is byte-identical on output and cuts the critical path on
   literature-heavy drugs (lisinopril 87s) as well as saving ~15 LLM calls per run.

## Changes made

### Supervisor fan-out (biggest win)

`cd83231` / `4065f7c`. Same agents, run concurrently: `analyze_literature` and `analyze_clinical_trials` are unchanged;
a single `investigate_top_candidates` tool launches them together instead of the LLM awaiting each in turn. Gated
behind `SUPERVISOR_FANOUT`. Measured: sildenafil 260s → 118s (55%), imatinib 206s → 129s (37%), identical results.

- Cooperative async on one event loop. Each agent call spends nearly all its wall-clock awaiting I/O (LLM, PubMed,
  CT.gov); `asyncio.gather` overlaps those waits.
- Two nested gathers in `supervisor/supervisor_tools.py`: outer over the top-N diseases
  (`SUPERVISOR_INVESTIGATION_CAP`), inner per disease over literature + trials — up to ~6 legs in flight.
- When fan-out is on, `build_supervisor_tools` inserts `investigate_top_candidates` before `finalize` and removes the
  per-candidate `analyze_*` tools; the LLM ignored prompt-level "do it in parallel" directives.
- The tool awaits `find_candidates_done` and `analyze_mechanism_done` events so the allowlist is populated first.
- Legs are invoked directly via `.ainvoke()` with a ToolCall-shaped dict, outside the ReAct loop; artifacts are stashed
  in the `auto_findings` closure and merged into `SupervisorOutput`. The LLM sees a one-line-per-disease summary.
- What made overlap safe: per-leg DB session from a shared pool; closure-scoped `shown_by_pair` dedup set;
  out-of-band collection so shared message history is never mutated concurrently; semaphore/lock rebind across event
  loops (`7341c80`, `c4f0d6e`); embedding moved off the event loop (`09419e7`).

### Concurrency / event loop

- `5d94f5e` — eliminate the discarded trailing turn. literature and mechanism use the prebuilt's `return_direct=True`;
  clinical_trials and supervisor use the gated ReAct loop in `agents/_react_loop.py` (they can reject/retry). Measured
  A/B via `scripts/trailing_turn_bench.py`, warm, metformin: 133.4s → 95.4s (~28%); every agent dropped exactly one
  turn (CT 3→2, mechanism 4→3, sub-agent per disease 7→6, supervisor 6→5).
- `09419e7` — offload embedding to a thread. `model.encode` ran on the event loop, stalling health/polling for the
  full encode. Moved to `asyncio.to_thread` inside the existing lock.
- `c89a85b` — concurrent PubMed efetch (semaphore-bounded gather, order preserved); removed the 90s 429 floor in
  `base_client` and the 90s MeSH retry sleep; tight NCBI request timeout instead of aiohttp's 5-min default.
- `76757d9` — embedding lock fairness: `embed_async` embeds in 64-item chunks and releases the model lock between
  chunks so a concurrent query-embed can interleave. Identical output.
- `7341c80` / `c4f0d6e` — semaphores and the model lock rebind lazily when the event loop changes; a second
  `asyncio.run()` used to crash.

### Caching

- `0e28b7b` — cache `search_trials` / `get_landscape`, the last two uncached CT.gov calls (namespaces `ct_search`,
  `ct_landscape`). Warm metformin after this: 0 CT.gov calls, external API ~1s; a prior warm run had hit 105.7s
  external / 97.2s clinical_trials when these were uncached and slow. It did not lower the warm floor (87.3s vs
  111.4s on identical code and caches is LLM variance); the remaining warm cost is the clinical_trials finalize
  summary (~26–35s/disease) plus supervisor glue (~37–59s).
- `167272c` — cache openFDA misses (7-day TTL) so unresolved aliases aren't re-fetched: ~77s/112 calls → ~0 warm.
- `9663d42` — warm the `pubmed_pubtypes` cache from efetch XML so `semantic_search` skips a second esummary
  round-trip per article.

### CPU / resource

- `447fb39` — pin torch threads to the cgroup CPU quota (reads `/sys/fs/cgroup/cpu.max`; only ever reduces).
- `9d5606c` — CPU-only PyTorch wheel in Docker; image ~6GB → ~2GB.
- `99b2a4b` — cap mechanism agent to top-3 targets; investigation cap 6 → 3. Behavior-neutral on tested drugs.

### Startup / cold path

- `ffbc536` / `198b968` / `f5b5df2` / `e50589f` — embedding-model loading moved from startup preload to lazy load on
  first analysis; seeded drugs skip the load entirely.

### Backoff

- `658ecdc` — backoff starts at 2s (2s/4s/8s); per-PMID logging trimmed out of the retrieval hot path.

## Rejected

- Fat-tool for clinical_trials: already one data turn plus the finalize summary.
- Haiku for summaries: changes scientific output wording.
- Capping mechanism targets below the current cap: dropped real candidates (metformin counterexample).
- Cross-candidate embed batching: implemented, measured 1.02x, reverted (see `plans/done/PLAN_option_b_batching.md`).
- Chasing sub-10s wins: below the noise floor.
- Pre-warming caches for arbitrary drugs on deploy: not done; example chips serve seeded reports instead.
