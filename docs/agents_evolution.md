# Agent Architecture Evolution

## Overview

Both the clinical trials and literature agents went through similar evolutionary arcs,
converging on the same architectural patterns as understanding deepened.

---

## Evolution of Understanding

### Stage 1 — `BaseAgent` + `create_agent()` (react)

The first agents used LangChain's `create_agent()` with a flat list of tools and a system prompt.
The agent was a class (`BaseAgent` subclass) with a `run()` method.

Key characteristics:
- Tools were module-level globals or built via closure (v2)
- Tools returned `model_dump()` dicts — Pydantic objects were serialized immediately
- After `agent.ainvoke()`, a `_parse_result()` method walked the message history,
  JSON-parsed each `ToolMessage`, and reconstructed Pydantic models from scratch
- The summary came from the final `AIMessage` in the history
- No explicit state management — everything lived in the message list

Pain points:
- `_parse_result()` was fragile — depended on JSON round-trips and `ToolMessage.name` matching
- Pydantic objects were destroyed and reconstructed unnecessarily
- No typed state — hard to know what had been collected at any point

---

### Stage 2 — LangGraph `StateGraph` (langgraph)

Replaced `create_agent` with an explicit `StateGraph` with three nodes:
`agent_node → tools_node → assemble_node` (loop back via `tools_condition`).

Key characteristics:
- Typed `State` (Pydantic `BaseModel`) holds structured results alongside messages
- `tools_node` has a dispatch table mapping tool names to state fields + parsers
- Still used `json.loads(msg.content)` + `model_validate()` in the dispatch table
- `assemble_node` extracts the final `AIMessage` text as the summary
- Tools still returned `model_dump()` dicts

Improvement over v2:
- Typed state makes it clear what has been collected
- Cleaner separation between tool execution and result storage
- No more `_parse_result()` — state is built incrementally

Remaining friction:
- JSON round-trip still happening: tool returns Pydantic → `ToolNode` serializes to JSON
  string → `tools_node` parses back to dict → `model_validate()` reconstructs Pydantic
- Drug/indication passed as LLM args — LLM could theoretically pass wrong values

---

### Stage 3 — `response_format="content_and_artifact"` (artifact)

Discovered LangChain's `response_format="content_and_artifact"` decorator option.
Tools return `tuple[str, <PydanticObject>]` — the string goes into `ToolMessage.content`
(what the LLM reads), and the raw Python object is stored in `ToolMessage.artifact`.

Key characteristics:
- `tools_node` reads `msg.artifact` directly — no `json.loads`, no `model_validate`
- Pydantic objects survive the tool call intact
- `tool_handlers` dispatch table simplified to a plain `artifact_fields` dict
- Drug/injection still LLM-provided (not yet injected via closure)

For clinical trials this is the right architecture because the LLM still needs to make
decisions (whitespace branch vs trials branch).

---

### Stage 4 — Linear Pipeline (current literature agent)

Realized that the literature agent's 4-step sequence is fully deterministic —
no branching, no LLM decision-making needed between steps.
Removed the LLM from orchestration entirely.

Key characteristics:
- No `agent_node`, no `ToolNode`, no `tools_condition`
- Each step is a plain async node that calls a service method directly
- State flows linearly: `expand → fetch → search → synthesize → assemble`
- `llm` parameter removed from `build_literature_graph()`
- Summary comes from `EvidenceSummary.summary` (already produced by `svc.synthesize`
  which calls the LLM internally via `query_llm`)
- No tool wrappers needed for the pipeline nodes

Key insight: "agent" doesn't always mean LLM-driven. When the flow is deterministic,
a pipeline is simpler, more reliable, and cheaper.

---

### Stage 5 — LLM-Driven Agentic with Closure Store (current literature agent)

Reverted from the deterministic pipeline back to an LLM-orchestrated agent, but kept
all the artifact and closure improvements. The key insight: the LLM adds value in
deciding *when* to call `build_drug_profile` vs skip it, handling edge cases (no results,
weak evidence), and producing a coherent final narrative summary.

Key characteristics:
- Three-node graph: `agent_node → tools_node → assemble_node` (loop via `tools_condition`)
- Five tools: `build_drug_profile`, `expand_search_terms`, `fetch_and_cache`,
  `semantic_search`, `synthesize`
- **Closure-based store** (`store: dict = {}`) shared across all tools via closure.
  `tools_node` writes `msg.artifact` to both state fields and the store after each round.
  Downstream tools read prior results from the store — no `InjectedState`, no magic.
- All tools use `response_format="content_and_artifact"`: `ToolMessage.content` is a
  human-readable string for the LLM; `ToolMessage.artifact` is the typed Python object
  for the code. No JSON round-trip.
- `build_drug_profile` is an explicit tool the LLM can call; `expand_search_terms`
  also auto-fetches the profile if missing — graceful degradation.
- `assemble_node` extracts the final `AIMessage` text after the last `ToolMessage`
  as the plain-text summary.
- System prompt instructs the LLM to batch independent tool calls and to write
  only 3-4 plain text sentences as its final message.

Why LLM orchestration again (vs Stage 4 pipeline):
- The LLM can skip `build_drug_profile` if not needed, saving an API call.
- The LLM can handle "no results" gracefully and still call `synthesize` on weak evidence.
- The final narrative summary benefits from LLM judgment rather than being read
  directly from `EvidenceSummary.summary`.
- The tool dependency graph (profile → queries → pmids → abstracts → synthesis)
  is implicit — the LLM infers it from tool descriptions, which is more flexible
  than hardcoding the sequence.

Remaining caveats:
- The closure store is a side-channel outside LangGraph's formal state. It works
  cleanly because tool execution is sequential within one agent instance, but is
  not safe for concurrent agent runs sharing the same store.
- `synthesize` does not guard against an empty `semantic_search_results` list —
  unlike `fetch_and_cache` and `semantic_search` which return early. If the LLM
  calls synthesize before semantic_search, `svc.synthesize` receives `[]`.
- `assemble_node` summary extraction is index-based (`i > last_tool_idx`) and could
  miss the summary if the LLM emits a tool call after its narrative message.

---

### Stage 6 — `create_react_agent` + Closure Store + Post-Hoc Assembly (current)

Replaced the hand-rolled three-node StateGraph with LangGraph's prebuilt `create_react_agent`.
Removed all LangGraph state machinery from the literature agent entirely — no `StateGraph`,
no `LiteratureState`, no `tools_node`, no `assemble_node`, no `InjectedState`.

Key characteristics:
- `build_literature_agent` calls `create_react_agent(model=llm, tools=tools, prompt=SYSTEM_PROMPT)`
  and returns the compiled agent directly. No graph wiring.
- `build_literature_tools` returns a plain `list` of tools. The closure-scoped `store: dict = {}`
  is internal — callers never see it. Tools write to and read from the store as a side effect.
- `run_literature_agent` calls `agent.ainvoke()` and then walks the returned message history:
  - `ToolMessage.artifact` is read for each of the four field-producing tools via a `field_map` dict
  - The last `AIMessage` with no `tool_calls` is extracted as the narrative summary
  - Results are assembled into a `LiteratureOutput` Pydantic model
- No LangGraph `State`, no typed graph state, no node functions. The only LangGraph surface
  is `create_react_agent` itself.
- `build_drug_profile` is intentionally absent from `field_map` — its artifact (`DrugProfile`)
  is consumed by subsequent tools via the closure store, not surfaced in `LiteratureOutput`.

Why `create_react_agent` over a hand-rolled StateGraph:
- The literature agent has no branching — no `tools_condition`, no conditional edges.
  `create_react_agent` handles the agent loop automatically without any graph wiring.
- Removing `StateGraph` eliminates `LiteratureState`, `tools_node`, `assemble_node`, and all
  the plumbing that connected them. Less code, fewer failure points.
- Post-hoc assembly (walking message history after the run) is simpler than accumulating
  state incrementally via a `tools_node` dispatch table.
- `create_react_agent` is the right default when the agent loop is standard — only reach for
  `StateGraph` when you need custom branching or multi-agent coordination.

Trade-offs vs Stage 5:
- No typed intermediate state — you can't inspect what's been collected mid-run.
- Post-hoc assembly assumes the LLM calls each tool at most once; if a tool is called
  twice, the second `ToolMessage.artifact` silently overwrites the first in the assembly loop.
- The closure store is still a side-channel: not visible to LangGraph, not serializable,
  not safe for concurrent runs sharing the same tool set.

---

## Key Lessons

1. **Avoid unnecessary JSON round-trips.** Tools that return Pydantic objects should use
   `response_format="content_and_artifact"` so objects survive the tool call intact.

2. **Inject context via closure.** Drug name, indication, db session, drug profile —
   anything already known at graph-build time should be captured in the closure,
   not passed as LLM tool arguments. The LLM should only decide *which* tools to call.

3. **Not everything needs an LLM orchestrator.** If the sequence is fixed and deterministic,
   a linear pipeline is the right choice. Use an LLM agent only when genuine branching
   or decision-making is required (e.g. clinical trials whitespace branching).

4. **Typed state is essential.** LangGraph's `StateGraph` with a Pydantic state model
   makes it clear what has been collected at each step, and avoids the fragile
   message-history-walking approach of `_parse_result()`.

5. **The LLM's view vs the code's view.** `ToolMessage.content` is what the LLM sees
   (needs to be human-readable and informative). `ToolMessage.artifact` is what the
   code uses (the raw Python object). Keep these concerns separate.

6. **Closure-based store beats `InjectedState` for inter-tool dependencies.** When tools
   need to read results from earlier tools (e.g. semantic_search needs PMIDs from
   fetch_and_cache), a shared mutable dict closed over by all tools is simpler to write,
   simpler to test (just set `store["key"] = value` before invoking), and avoids
   LangGraph's `InjectedState` boilerplate entirely.

7. **LLM orchestration vs pipeline is a judgment call, not a rule.** A deterministic
   pipeline is right when the sequence is fixed and all inputs are known. An LLM
   orchestrator adds value when there is optional branching, graceful degradation,
   or a final narrative step that benefits from reasoning. The literature agent
   went pipeline → LLM-driven as the agent's responsibilities grew.

---

## Session Summary — 2026-04-06 (continued)

### Literature Agent — Stage 6 rewrite + tests

- Rewrote literature agent as `create_react_agent` + post-hoc assembly (Stage 6 above)
  - Removed `StateGraph`, `LiteratureState`, `tools_node`, `assemble_node`
  - `build_literature_agent` returns a compiled `create_react_agent` directly
  - `run_literature_agent` walks message history, reads `msg.artifact` via `field_map`,
    extracts last no-tool-call `AIMessage` as narrative summary
  - `build_literature_tools` returns a plain list; closure store is fully internal
- Wrote 9 unit tests for `build_literature_tools` (`test_literature_tools.py`)
  - Each tool tested end-to-end through the closure store (chain populated via prior tool calls)
  - Covers: artifact contents, store read/write, early-return guards, `date_before` forwarding,
    fallback `build_drug_profile` call when store is empty
- Wrote 7 unit tests for `run_literature_agent` (`test_literature_agent.py`)
  - Agent mocked via `agent.ainvoke` returning fixed message histories
  - Covers: full happy path, last-AIMessage selection, each of 4 tools missing (parametrized),
    `build_drug_profile` ToolMessage correctly ignored
- Wrote 1 integration test (`test_literature_agent.py`) — Semaglutide + NASH, cutoff 2025-01-01
  - PMIDs, top-5 semantic results, evidence strength, and supporting PMIDs verified by live run

---

## Session Summary — 2026-04-06

### Literature Agent

- Added `semantic_search` tool and wired it into `tools_node` and `LiteratureOutput`
- Fixed `semantic_search` handler in `tools_node` (was incorrectly using `lambda d: d["pmids"]`)
- Added `synthesize` tool from the bak
- Evolved tools to use `response_format="content_and_artifact"` — all four tools
  (`expand_search_terms`, `fetch_and_cache`, `semantic_search`, `synthesize`) now return
  `tuple[str, artifact]`
- `tools_node` simplified to read `msg.artifact` — no JSON parsing
- Rewrote agent as a **linear pipeline** — removed LLM orchestration entirely:
  - `expand_node`, `fetch_node`, `search_node`, `synthesize_node`, `assemble_node`
  - `llm` parameter removed from `build_literature_graph()`
  - `summarize_node` removed — summary read directly from `EvidenceSummary.summary`
- Fixed `LiteratureState.semantic_search_results`: `list[str]` → `list[dict]`
- Added `evidence_summary: EvidenceSummary | None = None` to state and output
- Updated unit tests to match pipeline structure (one test per node, shared `_make_svc()`)
- Updated integration test to assert on `semantic_search_results`, `evidence_summary`, and `summary`
- Fixed `agent_tester.py` — removed `llm`, removed old `run_literature_agent_artifact`

### Clinical Trials Agent

- Wrote `response_format="content_and_artifact"` rewrite plan to
  `for_me/clinical_trials_artifact_rewrite.md`
- Updated `clinical_trials_tools.py` — drug/indication injected via closure,
  all tools use `response_format="content_and_artifact"` (work in progress, not merged to agent yet)
