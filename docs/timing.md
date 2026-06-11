# IndicationScout — Timing Analysis

> Captured 2026-06-10/11 from Railway `[TIMING]` deploy logs and local measured runs.
> Numbers mix cold/warm cache, Railway vs local, and code versions — see caveats at the
> bottom. This is an opportunistic session snapshot, NOT a controlled benchmark, except the
> "Controlled warm benchmark" section which IS apples-to-apples.

## Full-run totals (`[TIMING] run_analysis ... total`)

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

**Warm floor ≈ 110–140s. Cold ≈ 175–360s.** The spread is driven by cache state, not the drug.

## Phase breakdown (warm metformin — best-traced run, job 81337a10)

| Phase | Time | Notes |
|---|---|---|
| analyze_mechanism | 27–30s warm (132s cold) | agent loop; `_assemble_candidates` 0.6s warm / ~190s cold (51 targets) |
| investigate_top_candidates (3 diseases ∥) | 38–49s | clinical_trials dominates the parallel phase |
| critique_ranking | 1–10s | one supervisor LLM call |
| supervisor glue + finalize | remainder | |

## Sub-agent isolation (local, warm, measured directly)

| Component | Warm time | Composition |
|---|---|---|
| literature pure pipeline (no agent) | 1.5s | embeddings ~1.3s of it |
| literature **agent** (7-turn ReAct) | ~24s | ~22s is LLM round-trips, NOT data |
| clinical_trials data (5 tools, parallel) | 1.0s | |
| clinical_trials **agent** | 48.7s | see per-turn split below |
| mechanism agent loop | 27–68s | high run-to-run LLM-latency variance |

### clinical_trials agent per-turn (warm, metformin × PCOS, 50.5s)

| Turn | Tool(s) | in_tok | out_tok | est. time |
|---|---|---|---|---|
| 1 | all 5 data tools (batched) | 2,666 | 315 | ~5s (data ~1s) |
| 2 | finalize_analysis (summary) | 7,588 | 1,169 | ~23s |
| 3 | (final) trailing — **discarded** | 8,771 | 1,011 | ~22s |

→ ~45s of the 50s is LLM generation of two ~1k-token blocks; turn 3 (~22s) is thrown away.

## Trailing `(final)` turn waste across agents (output tokens, discarded)

| Agent | Trailing tokens | Est. wasted time |
|---|---|---|
| clinical_trials | 1,011 | ~22s |
| supervisor | 894 | ~18s |
| literature | 362–818 | ~8–16s |
| mechanism | 204–431 | ~5–9s |

Every agent emits a large discarded trailing summary after `finalize_analysis`. Summed across a
full run (4 agents, sub-agents invoked ~3× each) this is plausibly **60–90s** of pure waste.

## Embedding cost (local CPU, BioLORD-2023)

| Workload | Time |
|---|---|
| 100 abstracts (1 batch) | 1.3s |
| 3×100 sequential | 3.9s |
| 3×100 "parallel" (embed_async) | 3.8s |

Embeddings serialize across diseases (CPU/GIL-bound; `asyncio.to_thread` gives no real
parallelism), but absolute cost is small (~1.3s/disease). NOT a bottleneck — earlier hypothesis
ruled out.

## Cache impact (same drug, cold → warm)

- bupropion: 174.7s → 122.6s; external API 119s → 3.7s
- lisinopril: 263.0s → 107.7s
- metformin: ~7min → 124.7s; open_targets 1483s/1438 calls → 14s/58 calls
- **Cache state dominates everything.** Volume mount fix (SCOUT_CACHE_DIR → mounted /cache) was
  the single biggest latency win; before it, every deploy ran fully cold.

## Controlled warm benchmark (2026-06-11)

Same baseline code (literature NOT fat-tool), caches pre-warmed, local, single measured run per
drug. literature/clinical_trials columns show the SLOWEST single disease (they run in parallel).
`scripts/controlled_timing_bench.py`.

| Drug | Total | mechanism | literature (slowest disease) | clinical_trials (slowest) | critique |
|---|---|---|---|---|---|
| metformin | 132.6s | 28.7s | 31.2s | 45.9s | 1.4s |
| lisinopril | 159.3s | 25.2s | 86.7s | 45.0s | 2.0s |

Observations:
- **clinical_trials is consistently ~45s** across both drugs — the stable warm floor for the
  investigate phase.
- **literature varies wildly (31s vs 87s)** — lisinopril's literature was the bottleneck, not
  clinical_trials. The literature ReAct loop's cost scales with how much the LLM generates, which
  varies by drug/disease. This is exactly what the literature fat-tool removes.
- **mechanism is stable (~25–29s)**, critique negligible (~1–2s).
- The two phases (mechanism, then investigate) are serial, so Total ≈ mechanism + max(lit, CT)
  + critique + supervisor glue. metformin: 28.7 + 45.9 + 1.4 + glue ≈ 133s. lisinopril:
  25.2 + 86.7 + 2.0 + glue ≈ 159s.

Note: the `investigate (parallel wall)` column read 0.0s in the raw output — the
`investigate_top_candidates` `[TIMING]` line is emitted from a logger namespace the bench's
capture handler didn't catch. Dropped from the table; the per-disease columns are accurate.

## Caveats

- Earlier sections mix cold/warm, Railway/local, and pre/post fat-tool code — NOT apples-to-apples.
  The "Controlled warm benchmark" section IS apples-to-apples (same code, both warm, local).
- Per-turn times are estimated by splitting agent wall-time across turns by output-token share,
  not directly instrumented per LLM call.
- n=1 per drug in the controlled run; with ±15s run-to-run noise, treat differences under ~15s as
  noise. The literature 31s-vs-87s gap is well outside noise and is a real per-drug effect.
