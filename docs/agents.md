# Agent Architecture

This document describes the agent layer of IndicationScout: how agents are structured, how they call tools, and how to add new ones.

For the overall system architecture see [ARCHITECTURE.md](../ARCHITECTURE.md).

## BaseAgent

**File**: `src/indication_scout/agents/base.py`

All agents extend `BaseAgent`, which defines a single async interface:

```python
class BaseAgent(ABC):
    @abstractmethod
    async def run(self, input_data: dict[str, Any]) -> dict[str, Any]:
        ...
```

- Input and output are both plain dicts. The output dict contains a single key whose value is a typed Pydantic model.
- Agents are stateless -- instantiate and call `run()`.

## File Layout

Each agent is split across three files:

| File | Purpose |
|------|---------|
| `agents/<name>.py` | Agent class, system prompt, `run()` method, `_parse_result()` |
| `agents/<name>_tools.py` | `@tool` wrappers around data source client methods |
| `agents/<name>_model.py` | Agent-specific output Pydantic model |

Data source models (e.g. `Trial`, `WhitespaceResult`, `IndicationLandscape`) live in `models/` and are referenced by the agent output model -- never duplicated.

## ClinicalTrialsAgent

### Agent class

**File**: `src/indication_scout/agents/clinical_trials.py`

```python
class ClinicalTrialsAgent(BaseAgent):
    async def run(self, input_data: dict[str, Any]) -> dict[str, Any]:
```

**Input** (`input_data` dict):

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `drug_name` | `str` | Yes | Drug to investigate |
| `disease_name` | `str` | Yes | Target indication |
| `date_before` | `date` | No | Temporal holdout cutoff |

**Output**: `{"clinical_trials_output": ClinicalTrialsOutput}`

**LLM**: `settings.big_llm_model` via `ChatAnthropic` from `langchain_anthropic`. Temperature 0, max_tokens 4096.

**Agent creation**:

```python
agent = create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)
```

Uses `langchain.agents.create_agent` to build a ReAct agent. The agent is invoked with:

```python
result = await agent.ainvoke(
    {"messages": [{"role": "user", "content": user_message}]},
    config={"recursion_limit": MAX_TOOL_ROUNDS},  # MAX_TOOL_ROUNDS = 10
)
```

**System prompt branching strategy** -- the LLM decides tool order:

1. Always start with `detect_whitespace`
2. If trials exist: `search_trials` then `get_landscape`
3. If whitespace (no trials): `get_terminated` then `get_landscape`
4. If terminated trials show safety/efficacy failures: may skip `get_landscape`

The LLM ends with a 2-3 sentence natural language assessment.

**Result parsing and collation**: The static method `_parse_result()` walks the full message history from `agent.ainvoke()` and collates multi-tool results into a single typed output:

- Iterates all messages, identifying `ToolMessage` objects by checking for a `.name` attribute
- Each tool's response is matched by name to a dedicated slot on `ClinicalTrialsOutput`: `detect_whitespace` → `whitespace`, `search_trials` → `trials`, `get_landscape` → `landscape`, `get_terminated` → `terminated`
- There is no merging across tools — each tool populates its own field. If a tool wasn't called, its field stays at the default (empty list or `None`)
- If the same tool is called twice, the second response overwrites the first (the loop doesn't accumulate). This is acceptable because the system prompt guides the LLM to call each tool once
- The last `AIMessage` (no `.name` attribute) is extracted as the summary. Handles both string content and list-of-blocks content formats
- Returns a `ClinicalTrialsOutput` instance with all pieces assembled

### Tools

**File**: `src/indication_scout/agents/clinical_trials_tools.py`

Entry point: `build_clinical_trials_tools(date_before: date | None = None) -> list`

Returns four LangChain `@tool` functions. The `date_before` parameter is captured via closure so it flows to every client call without being exposed as a tool parameter to the LLM.

| Tool | Arguments | Client Method | Returns |
|------|-----------|--------------|---------|
| `detect_whitespace` | `drug: str, indication: str` | `ClinicalTrialsClient.detect_whitespace()` | `WhitespaceResult.model_dump()` |
| `search_trials` | `drug: str, indication: str` | `ClinicalTrialsClient.search_trials()` | `[Trial.model_dump(), ...]` |
| `get_landscape` | `indication: str` | `ClinicalTrialsClient.get_landscape(top_n=10)` | `IndicationLandscape.model_dump()` |
| `get_terminated` | `query: str` | `ClinicalTrialsClient.get_terminated()` | `[TerminatedTrial.model_dump(), ...]` |

Design rules:

- Tools accept primitive types (strings) so the LLM can provide arguments directly.
- Tools return dicts via `model_dump()` because LangChain serializes tool return values into the message history as JSON for the LLM to read — Pydantic objects aren't directly serializable in that context. The `_parse_result()` method then reconstructs typed Pydantic models from those dicts on the other side: `Client returns Pydantic → tool calls .model_dump() → LLM sees dict → _parse_result reconstructs Pydantic`.
- Each tool creates its own client session: `async with ClinicalTrialsClient() as client:`.
- `get_landscape` passes `top_n=10` to keep the response within manageable LLM context.

### Output model

**File**: `src/indication_scout/agents/clinical_trials_model.py`

```python
class ClinicalTrialsOutput(BaseModel):
    trials: list[Trial] = []
    whitespace: WhitespaceResult | None = None
    landscape: IndicationLandscape | None = None
    terminated: list[TerminatedTrial] = []
    summary: str = ""
```

- References models from `models/model_clinical_trials.py`.
- Has the `coerce_nones` model validator.
- Fields are None/empty when the agent chose not to call the corresponding tool. This is expected behavior, not an error.

### Data source models

**File**: `src/indication_scout/models/model_clinical_trials.py`

These are the contracts between `ClinicalTrialsClient` and the agents. All have the `coerce_nones` validator.

**Trial-level models**:

| Model | Key fields |
|-------|------------|
| `Intervention` | `intervention_type`, `intervention_name`, `description` |
| `PrimaryOutcome` | `measure`, `time_frame` |
| `Trial` | `nct_id`, `title`, `brief_summary`, `phase`, `overall_status`, `why_stopped`, `indications`, `interventions` (list[Intervention]), `sponsor`, `enrollment`, `start_date`, `completion_date`, `primary_outcomes` (list[PrimaryOutcome]), `references` |

**Whitespace models**:

| Model | Key fields |
|-------|------------|
| `IndicationDrug` | `nct_id`, `drug_name`, `indication`, `phase`, `status`. Has `from_trial()` classmethod. |
| `WhitespaceResult` | `is_whitespace`, `no_data`, `exact_match_count`, `drug_only_trials`, `indication_only_trials`, `indication_drugs` (list[IndicationDrug]) |

**Landscape models**:

| Model | Key fields |
|-------|------------|
| `CompetitorEntry` | `sponsor`, `drug_name`, `drug_type`, `max_phase`, `trial_count` (int), `statuses` (set[str]), `total_enrollment` (int) |
| `RecentStart` | `nct_id`, `sponsor`, `drug`, `phase` |
| `IndicationLandscape` | `total_trial_count`, `competitors` (list[CompetitorEntry]), `phase_distribution` (dict[str, int]), `recent_starts` (list[RecentStart]) |

**Terminated trial models**:

| Model | Key fields |
|-------|------------|
| `TerminatedTrial` | `nct_id`, `title`, `drug_name`, `indication`, `phase`, `why_stopped`, `stop_category` (safety/efficacy/business/enrollment/other/unknown), `enrollment`, `sponsor`, `start_date`, `termination_date`, `references` |

## Data Flow

```
ClinicalTrialsAgent.run(input_data)
    |
    +-- build_clinical_trials_tools(date_before)   # creates 4 @tool functions
    +-- ChatAnthropic(model=big_llm_model)           # LLM for tool-calling decisions
    +-- create_agent(model, tools, system_prompt)   # LangChain ReAct agent
    |
    +-- agent.ainvoke(messages)                     # ReAct loop begins
    |       |
    |       +-- LLM decides: call detect_whitespace
    |       |       +-- ClinicalTrialsClient.detect_whitespace()
    |       |       +-- returns WhitespaceResult.model_dump()
    |       |
    |       +-- LLM decides: call search_trials or get_terminated
    |       |       +-- ClinicalTrialsClient method
    |       |       +-- returns list of model_dump() dicts
    |       |
    |       +-- LLM decides: call get_landscape (or skip)
    |       |       +-- ClinicalTrialsClient.get_landscape(top_n=10)
    |       |       +-- returns IndicationLandscape.model_dump()
    |       |
    |       +-- LLM produces final text summary
    |       +-- stop_reason: end_turn
    |
    +-- _parse_result(result)
    |       +-- walks message history
    |       +-- ToolMessages -> reconstructs Pydantic models from dicts
    |       +-- last AIMessage -> summary text
    |       +-- returns ClinicalTrialsOutput
    |
    +-- returns {"clinical_trials_output": ClinicalTrialsOutput}
```

## How to Call

```python
from indication_scout.agents.clinical_trials import ClinicalTrialsAgent
from datetime import date

agent = ClinicalTrialsAgent()
result = await agent.run({
    "drug_name": "semaglutide",
    "disease_name": "NASH",
    "date_before": date(2023, 1, 1),  # optional
})
output = result["clinical_trials_output"]  # ClinicalTrialsOutput

# Access structured data
output.trials          # list[Trial]
output.whitespace      # WhitespaceResult | None
output.landscape       # IndicationLandscape | None
output.terminated      # list[TerminatedTrial]
output.summary         # str -- natural language assessment
```

## Dependencies

```toml
"langchain-core>=1.2.23"
"langchain>=1.2.13"
"langchain-anthropic>=0.3.0"
```

## Test Layout

```
tests/
+-- unit/agents/
|   +-- test_clinical_trials_tools.py    # mocked client, verifies each tool returns correct dicts
|   +-- test_clinical_trials_agent.py    # tests _parse_result with fake message histories
+-- integration/agents/
    +-- test_clinical_trials_tools.py    # hits real ClinicalTrials.gov API
    +-- test_clinical_trials_agent.py    # hits real ClinicalTrials.gov + Anthropic APIs
```

**Unit tests** mock `ClinicalTrialsClient` and verify:

- Each tool returns `model_dump()` dicts with correct structure
- `date_before` flows through the closure to client calls
- `_parse_result` correctly reconstructs Pydantic models from message history

**Integration tests** verify end-to-end:

- Tools return correct data from real API
- Agent produces correct structured output for known drug-disease pairs
- Agent handles nonexistent drugs/diseases gracefully

## Adding a New Agent

To add a new agent (e.g. `LiteratureAgent`):

1. **Create `agents/literature_model.py`** -- output Pydantic model referencing models from `models/`. Include `coerce_nones` validator.
2. **Create `agents/literature_tools.py`** -- `build_literature_tools()` returning a list of `@tool` functions wrapping data source client methods. Tools accept primitives, return dicts via `model_dump()`. Each tool manages its own client session.
3. **Create `agents/literature.py`** -- agent class extending `BaseAgent`. Includes system prompt, `run()` method using `create_agent` and `ainvoke`, and a `_parse_result` static method that walks message history to reconstruct the typed output model.
4. **Add unit tests** in `tests/unit/agents/test_literature_*.py`.
5. **Add integration tests** in `tests/integration/agents/test_literature_*.py`.

Key patterns to follow:

- Tools accept primitive types, return dicts via `model_dump()`
- Tools create their own client sessions (`async with Client() as c:`)
- Use closures to capture config that the LLM should not see (e.g. `date_before`)
- `_parse_result` walks message history to reconstruct typed output from `ToolMessage` objects
- Output model fields default to None/empty for tools that were not called
- All Pydantic models that ingest external data include the `coerce_nones` validator

## Agent Catalogue

| Agent | File | Status | LLM | Data Source |
|-------|------|--------|-----|-------------|
| ClinicalTrialsAgent | `agents/clinical_trials.py` | Implemented | big_llm_model | ClinicalTrialsClient |
| LiteratureAgent | `agents/literature.py` | Planned | TBD | RetrievalService (PubMed + pgvector) |
| MechanismAgent | `agents/mechanism.py` | Planned | TBD | OpenTargetsClient |
| SafetyAgent | `agents/safety.py` | Planned | TBD | FDAClient |
| Orchestrator | `agents/orchestrator.py` | Planned | Sonnet | Coordinates all agents |
