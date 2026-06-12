# Making IndicationScout Faster — Analysis, Findings, Recommendations

> Session analysis, 2026-06-10/11. Companion data in `timing.md`. All numbers measured
> (Railway `[TIMING]` logs + local isolation runs), not estimated unless noted.

## TL;DR

1. **Cache state dominates everything.** Same drug swings 124s (warm) ↔ 360s (cold). The volume
   mount fix (the cache was writing to ephemeral image disk, wiped every deploy) was the single
   biggest latency win this session. Cold is now a one-time cost per drug, then warm forever.
2. **Warm, the cost is LLM generation — not data.** External API drops to 3–10% of a warm run.
   Embeddings (~1.3s/disease) are NOT a bottleneck (hypothesis ruled out by measurement).
3. **The biggest behavior-preserving lever is the discarded trailing `(final)` turn.** Every agent
   generates a large summary AFTER `finalize_analysis` that is thrown away — ~22s on
   clinical_trials alone, plausibly 60–90s across a full run.
4. **The warm bottleneck is drug-dependent** (controlled benchmark): clinical_trials is a stable
   ~45s floor, but literature swings 31s–87s by drug. For metformin, CT dominates; for lisinopril,
   literature (87s) dominates. Both are LLM-generation bound.
5. **Run-to-run noise is ±15s** (LLM latency varies even at temperature=0). Sub-10s "improvements"
   are not reliably measurable end-to-end.

## How the time decomposes (warm metformin)

Full warm run ≈ 110–140s. Phases:

| Phase | Time | Composition |
|---|---|---|
| analyze_mechanism | 27–30s | agent loop (LLM); `_assemble_candidates` 0.6s warm |
| investigate_top_candidates (3 diseases, parallel) | 38–49s | clinical_trials or literature, drug-dependent |
| critique_ranking | 1–10s | one supervisor LLM call |
| supervisor glue + finalize | remainder | supervisor's own ReAct turns |

Sub-agents, isolated and measured warm:

| Component | Warm | What it actually is |
|---|---|---|
| literature pure pipeline (no agent) | 1.5s | embeddings ~1.3s |
| literature **agent** (7 ReAct turns) | ~24s (up to 87s for some drugs) | LLM round-trips + generation, not data |
| clinical_trials data tools (5, parallel) | 1.0s | |
| clinical_trials **agent** | 48.7s | turn 2 summary ~23s + turn 3 trailing ~22s |
| mechanism agent loop | 27–68s | high LLM-latency variance |

### clinical_trials agent per-turn (warm, 50.5s) — the clearest example

| Turn | Tool(s) | out_tok | ~time |
|---|---|---|---|
| 1 | all 5 data tools (batched in one turn) | 315 | ~5s (data ~1s) |
| 2 | finalize_analysis (the summary the supervisor uses) | 1,169 | ~23s |
| 3 | (final) trailing — **discarded** | 1,011 | ~22s |

→ ~45s of 50s is LLM generation of two ~1k-token blocks. Turn 3 is pure waste.

## Findings, in priority order

### F1. The trailing `(final)` turn is the largest waste — and it's discarded
After `finalize_analysis`, the ReAct loop feeds the tool result back to the model, which generates
one more message. That message is never read (assembly uses the `finalize_analysis` artifact; the
clinical_trials prompt literally says "Plain-text after this is discarded"). It is large:

| Agent | Trailing tokens | Est. wasted time |
|---|---|---|
| clinical_trials | 1,011 | ~22s |
| supervisor | 894 | ~18s |
| literature | 362–818 | ~8–16s |
| mechanism | 204–431 | ~5–9s |

Across a full run (4 agents, sub-agents invoked ~3×) this is plausibly **60–90s of pure waste**,
much of it on the critical path.

### F2. clinical_trials cost is LLM generation, NOT turn count
Its 5 data tools are already batched into one turn (1s of work). Its cost is two large LLM
generations (summary + discarded trailing). So a fat-tool refactor (which helped literature by
collapsing 5 turns) would NOT help clinical_trials — it's already minimal on turns. The lever here
is F1 (kill the trailing turn).

### F3. The literature fat-tool's latency value is drug-dependent
Collapsing literature's 7-turn ReAct loop to 2 turns is correct and proven byte-identical
(candidate diseases, evidence strength, pmids all unchanged; the `top_diseases` wobble is
pre-existing ranking nondeterminism, not the change).

The controlled warm benchmark (timing.md) shows literature time is **highly drug-dependent**:
- metformin: literature 31s, clinical_trials 46s → CT is the bottleneck, literature savings hidden.
- lisinopril: **literature 87s**, clinical_trials 45s → **literature IS the bottleneck**.

So the fat-tool is NOT just a cost win. On literature-heavy drugs it cuts the critical path
materially (the 87s is the ReAct loop generating a lot — exactly what collapsing to 2 turns
removes). Net: worth it for BOTH cost (−15 Anthropic calls/run) AND latency on literature-heavy
drugs. (Currently stashed at `stash@{0}` on main, awaiting review.)

### F4. Caching is correct but was silently broken on Railway
The file cache logic is sound and persists within a deploy. It was writing to `/data/cache` on
ephemeral image disk because no volume was mounted there (two-volume conflict + 24h soft-delete
made this hard to fix). Fixed by mounting the volume at `/cache` and pointing
`SCOUT_CACHE_DIR`/`HF_HOME` at it. Verified via a temporary `/debug/cache` endpoint
(`on_separate_mount: true`, OT namespaces populating on the volume).

### F5. Embeddings are not a bottleneck (hypothesis tested and rejected)
BioLORD embedding is ~1.3s per 100-abstract batch and DOES serialize across diseases
(CPU/GIL-bound; `asyncio.to_thread` gives no real parallelism). But the absolute cost is small —
the literature pure pipeline is 1.5s total warm. The 24s literature cost was LLM round-trips, not
embeddings.

## Recommendations, ranked by (impact × safety)

### R1. Eliminate the trailing `(final)` turn across all agents — DO THIS FIRST
- **Impact:** highest. ~22s on clinical_trials (critical path), 60–90s aggregate.
- **Safety:** high. The trailing message is discarded; removing it is behavior-preserving.
- **How:** terminate the LangGraph loop when `finalize_analysis` is called (custom node returning
  `Command(goto=END)`, or a finalize tool that ends the graph) so the model never gets the tool
  result back to react to. Verify per-agent that the trailing AIMessage is truly unread first.
- **Caveat:** the sub-agents are `create_react_agent` (prebuilt) — this needs a small custom graph
  or a post-finalize short-circuit. Test before/after per agent (turns drop by 1, output identical).

### R2. Keep the literature fat-tool — cost win AND latency win on literature-heavy drugs
- Proven safe, −15 LLM calls/run, and cuts the critical path materially when literature dominates
  (e.g. lisinopril 87s). Commit it.
- Fix its integration tests (still reference the removed 5 tools) before merging.

### R3. Cache `search_trials` + `get_landscape` — cold-run win only
- These two CT.gov calls are uncached (unlike `ct_completed`/`ct_terminated`). Warm, all 5 tools
  are 1s, so this barely helps warm — but it removes repeated network calls cold and across runs.
- Mirror the existing `ct_completed` cache pattern (namespaces `ct_search`, `ct_landscape`,
  `CLINICAL_TRIALS_CACHE_TTL`). Low risk, modest reward.

### R4. Pre-warm caches for common drugs — UX win
- Example chips already serve a cached `SupervisorOutput`. For arbitrary drugs, the first run is
  cold. If cold-first-run latency matters, pre-warm common drugs' OT/PubMed caches on deploy.

### Explicitly NOT recommended
- **Fat-tool for clinical_trials** — won't help; it's already 1 turn of data + LLM summary.
- **Haiku for summaries** — user ruled out (changes scientific output wording).
- **Capping mechanism targets** — proven to drop real candidates (metformin counterexample);
  violates the accuracy-over-coverage rule.
- **Chasing sub-10s wins** — below the ±15s run-to-run noise floor; not measurable.

## What would make the next analysis trustworthy
The numbers here mix cold/warm, Railway/local, and code versions. A controlled benchmark (same
code, all warm, several drugs, phase splits) is in `timing.md`. Per-turn times are estimated by
splitting agent wall-time by output-token share, not directly instrumented — adding per-LLM-call
timing would make R1's before/after exact.
