# Anthropic Prompt Caching

How IndicationScout uses Anthropic's ephemeral prompt caching to cut per-turn latency and cost
in the ReAct agent loops. Implementation lives in `src/indication_scout/agents/_react_loop.py`.

## Why

Every agent (`supervisor`, `clinical_trials`, `literature`, `mechanism`) runs a multi-turn ReAct
loop: model call -> tool calls -> tool results appended -> model call again. Without caching,
each turn reprocesses the entire accumulated history (system prompt, tool definitions, every
prior tool result) at full input-token cost. Caching lets turns 2+ read the static prefix and
prior conversation from cache instead of reprocessing them.

Anthropic's caching is a **prefix match**: any byte change anywhere in the cached prefix
invalidates everything after it. Render order is `tools` -> `system` -> `messages`.

## Two breakpoints per turn

### 1. System prompt breakpoint (static prefix)

`cached_system_message(prompt)` wraps the system prompt as:

```python
SystemMessage(content=[
    {"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}
])
```

Because tools render before the system block, one breakpoint here caches **tools + system
prompt together**. This is built once per agent and never changes across turns, so from turn 2
onward it's a pure cache read.

### 2. Growing-history breakpoint (conversation so far)

`_with_history_breakpoint(messages)` stamps a `cache_control` marker onto the **last** message's
final content block:

- string content -> wrapped as `[{"type": "text", "text": content, "cache_control": {...}}]`
- list content -> marker merged into a copy of the last dict block
- empty content (e.g. an AIMessage with only tool_calls) -> left alone; the system breakpoint is
  the only cache point that turn

Critically, this operates on a **shallow copy** of the last message only — stored state
messages are never mutated. The breakpoint must move to the new tail every turn; if it were left
on an interior message it would (a) permanently consume one of Anthropic's 4 per-request
breakpoint slots and (b) drift the cached prefix out of sync with the actual conversation.

Net effect: turn N reads the entire prior conversation (including all tool results) from cache
and only pays full price for the new tail appended since turn N-1.

## Two integration points

The codebase has two ReAct loop shapes, so caching is wired in twice:

- **`build_gated_react_loop`** (used by `supervisor` and `clinical_trials` — both have a
  non-terminal reject path in their finalize tool) applies both breakpoints inline inside
  `call_model`.
- **`history_cache_pre_model_hook`** (used by `literature` and `mechanism` — both use
  LangGraph's prebuilt `create_react_agent`) is the equivalent seam for the prebuilt: it returns
  `{"llm_input_messages": _with_history_breakpoint(state["messages"])}`. Returning it under
  `llm_input_messages` (not `messages`) means the breakpoint is used as model input for that
  turn only and never gets persisted into checkpointed state.

## Observability

`supervisor_agent.py` logs per-turn cache stats pulled from LangChain's
`usage_metadata.input_token_details`:

```python
_cache_read = _details.get("cache_read", 0)
_cache_write = (
    _details.get("ephemeral_5m_input_tokens", 0)
    + _details.get("ephemeral_1h_input_tokens", 0)
) or _details.get("cache_creation", 0)
```

Logged per turn as `[LLMTURN] supervisor turn N/M: in=... out=... cache_read=... cache_write=... -> <tool(s)>`.

**Diagnostic signal:** `cache_read == 0` across turns 2+ means something is silently
invalidating the cache — most commonly a prefix below the model's minimum cacheable size, or a
byte-level change upstream of a breakpoint (non-deterministic serialization, a timestamp, a
varying tool set).

## Benchmarking

`scripts/prompt_cache_bench.py` runs the full pipeline (`run_analysis`) twice back-to-back for
one drug and reports wall-clock, cache hit rate, USD cost, and token breakdown for each run. It
makes no changes to agent code or behavior — it only taps `ChatAnthropic._agenerate` (read-only)
to read `usage_metadata` off every LLM call from every agent.

```
python scripts/prompt_cache_bench.py <drug>
```

To exercise a wider concurrent candidate fan-out than the production default (3), set
`SUPERVISOR_INVESTIGATION_CAP` in the environment before running — the script does this via a
process-local `os.environ` override, not by editing `.env.constants` (the shipped production
value).

### The cache is shared across drugs, not per-drug

The system-prompt breakpoint (`cached_system_message`) caches text that's **identical regardless
of which drug or disease is being analyzed** — the system prompt and tool schemas don't mention
the drug. So once any agent runs once, for any drug, that breakpoint stays warm for the TTL
window and gets reused by the *next* run regardless of drug. Only the growing-history breakpoint
is drug-specific (it includes the actual tool results for that pair).

Practical consequence: getting a genuinely cold baseline requires the API key to have made zero
calls to Anthropic in the preceding TTL window — not just avoiding the same drug. In practice,
"run 1" of a benchmark still shows both `cache_read > 0` and `cache_write > 0` (a mix) unless
nothing has hit the API in a while. Treat single-run "cold" numbers as a lower-bound estimate of
the real cold-start gain, not the true ceiling.

### TTL

The `cache_control` blocks in `_react_loop.py` use plain `{"type": "ephemeral"}` with no explicit
`"ttl"` key, so they default to Anthropic's **5-minute** tier (not the 1-hour tier, which requires
an explicit `"ttl": "1h"` field — not set anywhere in this codebase). The 5-minute window resets
on every read, so as long as *some* call touches the shared system-prompt prefix at least every 5
minutes, it never fully expires.

### Measured results (2026-07-14)

Full-pipeline runs (`run_analysis`), comparing a first run for a drug against an immediate
second run for the same drug. Given the cross-drug sharing above, "cold" here means "first touch
in this session," not "guaranteed empty cache" — see caveat above.

| Drug | Fan-out | Cold hit rate | Warm hit rate | Latency Δ | Cost Δ |
|---|---|---|---|---|---|
| baricitinib | 3 | 48% | 49% | 13.8% faster | 11.2% cheaper |
| adalimumab | 3 | 47% | 49% | 21.8% faster | 13.7% cheaper |
| bupropion | 3 | 42% | 49% | -0.8% (noise) | 17.1% cheaper |
| bupropion | 3 | 40% | 49% | -7.0% (noise) | 20.0% cheaper |
| aspirin | 3 | 46% | 48% | 18.4% faster | 18.5% cheaper |
| bupropion | 3 | 40% | 48% | 14.5% faster | 18.7% cheaper |
| gefitinib | 5 | 44% | 48% | 35.9% faster | 10.5% cheaper |

**Takeaways:**

- **Hit rate converges to ~44-49%**, stable across drugs, run count, and fan-out width (3 vs 5
  concurrent candidates).
- **Cost savings are the reliable number: 10-20% cheaper on the warm run, every single trial, no
  exceptions.** Mechanically grounded in the cache-read rate (~$0.30/MTok) vs. base input rate
  (~$3/MTok for the Sonnet tier) — roughly half of all input tokens get the ~10x discount.
- **Latency savings are real but noisier**: positive in most runs (14-36% faster), flat-to-
  slightly-negative in a couple (API queueing/network variance can dominate at this scale).
  Wider fan-out (5 vs 3) showed the largest latency win observed (35.9%), consistent with more
  concurrent turns competing to read the same shared cache rather than reprocess it.
