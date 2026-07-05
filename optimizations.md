# Performance Optimizations

A log of the changes over the last ~2 months that significantly improved runtime/throughput.
Commits with measured before/after numbers are the highest-impact.

## Deep dive: the supervisor fan-out (biggest win)

**Same agents, run concurrently.** The fan-out did NOT rewrite any agent. `analyze_literature` and
`analyze_clinical_trials` are the identical code paths as before. What changed is *who drives them
and when*: instead of the LLM `await`-ing each tool one at a time inside the ReAct loop, a single
tool launches them all together.

**How concurrency is achieved.** Cooperative async on a single event loop — not threads, not
processes. Each agent call is an `async` coroutine that spends almost all its wall-clock `await`-ing
I/O (LLM API calls, PubMed/CT.gov HTTP). `asyncio.gather` schedules many at once; whenever one hits
an `await`, the loop runs another. So while candidate A's literature leg waits on Anthropic,
candidate B's trials leg issues its HTTP request. Nothing runs on two CPUs at once — the I/O waits
just overlap, and that's where ~all the time goes.

**Two nested gather levels** (`investigate_top_candidates` in `supervisor/supervisor_tools.py`):
- Outer `gather` over the top-N candidate diseases (`SUPERVISOR_INVESTIGATION_CAP`, currently 3).
- Inner `gather` per disease over `analyze_literature` + `analyze_clinical_trials`.
- → up to ~6 sub-agent legs in flight simultaneously.

**Forcing the LLM to use it.** When `SUPERVISOR_FANOUT` is on, `build_supervisor_tools` inserts
`investigate_top_candidates` before `finalize` AND removes the per-candidate `analyze_*` tools, so
the LLM has no serial fallback — it ignored prompt-level "do it in parallel" directives, so the
tooling forces a single fan-out call.

**Sequencing.** The tool first awaits two `asyncio.Event`s (`find_candidates_done`,
`analyze_mechanism_done`) so the allowlist is fully populated before fanning out.

**Out-of-band result collection.** The legs are invoked directly via `.ainvoke()` with a
ToolCall-shaped dict (so the return carries a populated `.artifact`), NOT through the ReAct loop —
so the agent never sees their messages. Artifacts are stashed in the `auto_findings` closure dict
keyed by disease and merged into `SupervisorOutput` afterward. The tool returns only a compact
one-line-per-disease summary for the LLM to rank against.

**What made overlap safe** (the real engineering, not just `gather`):
- Per-leg DB session from a shared pool — a SQLAlchemy `Session` is not safe for concurrent use.
- Closure-scoped dedup set (`shown_by_pair`) so concurrent legs don't clobber each other's
  trial-relevance state.
- Out-of-band collection avoids concurrent mutation of shared ReAct message history.
- Prerequisites from earlier commits: semaphore/lock rebind across event loops (`7341c80`,
  `c4f0d6e`) or the concurrent path crashed; embedding moved off the event loop (`09419e7`) or one
  CPU-bound `encode` froze the whole loop.

## Concurrency / event-loop (largest latency wins)

- **`cd83231` / `4065f7c` — supervisor fan-out.** Collapses per-candidate serial ReAct
  investigation into one parallel `investigate_top_candidates` fan-out. Gated behind
  `SUPERVISOR_FANOUT`. *Measured: sildenafil 260s → 118s (55%), imatinib 206s → 129s (37%),
  identical results.* Single biggest win.
- **`5d94f5e` — eliminate discarded trailing turn after finalize.** Every agent ran one wasted
  LLM turn after `finalize` whose output was discarded. Removed via `return_direct` (literature,
  mechanism) and a gated ReAct loop `_react_loop.py` (clinical_trials, supervisor, which can
  reject/retry). *Measured: metformin 133.4s → 95.4s.*
- **`09419e7` — offload embedding to a thread.** `model.encode` (CPU-bound) ran directly on the
  event loop, stalling health/polling/concurrent runs for the full encode. Moved to
  `asyncio.to_thread` inside the existing lock (encodes still serialize, loop stays free).
- **`c89a85b` — concurrent PubMed efetch + drop rate-limit floors.** efetch batches now run via
  `asyncio.gather` (semaphore-bounded) instead of a serial loop; order preserved. Removed the 90s
  429 floor in `base_client` and the 90s MeSH retry sleep in `disease_helper` (per-second rate
  limits clear in ~1s). Added a tight NCBI request timeout so stalled connections fail fast instead
  of hanging on aiohttp's 5-min default.
- **`76757d9` — embedding lock fairness.** `embed_async` embeds in 64-item chunks and releases the
  shared model lock between chunks so a concurrent query-embed can interleave (head-of-line fix).
  Same compute, identical output.
- **`7341c80` / `c4f0d6e` — semaphore/lock rebind across event loops.** PubMed/MeSH semaphores and
  the embedding model lock bound to the loop they were created on; a second `asyncio.run()` crashed.
  Lazy rebind on loop change. (Reliability of the concurrent path rather than raw latency.)

## Caching (eliminate repeated network)

- **`0e28b7b` — cache `search_trials` / `get_landscape`.** The last two uncached CT.gov calls.
  Slow CT.gov responses added up to ~90s/run. *Measured: external API 13.7s → 105.7s for the same
  drug across two runs; now warm-cached.*
- **`167272c` — cache openFDA misses + force parallel fan-out.** Caches 404/empty label lookups
  (7-day TTL) so unresolved aliases aren't re-fetched. *Measured: ~77s/112 calls → ~0 when warm
  (was the dominant openFDA cost).* Also dropped per-candidate `analyze_*` tools when fan-out is on
  so the LLM can't fall back to serial investigation, plus `[TIMING]` accounting.
- **`9663d42` — warm pubtypes cache from efetch XML.** Populates the `pubmed_pubtypes` cache during
  efetch parsing so `semantic_search`'s later `fetch_pubtypes()` hits the cache instead of a second
  esummary round-trip per article.

## CPU / resource

- **`447fb39` — pin torch threads to cgroup quota.** In a container torch defaulted to host core
  count (e.g. 48 vs the 24-vCPU quota), causing scheduler thrash. Reads `/sys/fs/cgroup/cpu.max`
  and caps `torch.set_num_threads` to the real quota. Only ever reduces; no-op when unknown.
- **`9d5606c` — CPU-only PyTorch in Docker.** Default torch dragged in the multi-GB CUDA/NVIDIA
  stack, useless on a CPU host. Install CPU-only wheel first. *Image ~6GB → ~2GB* (build/deploy).
- **`99b2a4b` — cap mechanism agent to top-3 targets; investigation cap 6 → 3.** Fewer downstream
  LLM/data calls. Behavior-neutral on tested drugs.

## Startup / cold path

- **`ffbc536` / `198b968` / `f5b5df2` / `e50589f`** — iterations on embedding-model loading
  (startup preload → lazy load on first analysis). Startup no longer blocks on the BioLORD cache;
  seeded drugs skip the load entirely.

## Backoff tuning

- **`658ecdc`** — backoff starts at 2s (2s/4s/8s); trimmed verbose per-PMID logging out of the
  retrieval hot path.
