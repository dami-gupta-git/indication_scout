# ClinicalTrialsAgent — Design

## What it does

Takes a drug name and disease name. Uses ClinicalTrials.gov to assess whether this pair is being explored, how crowded the space is, and whether prior attempts failed. Returns a structured assessment.

This is the simplest agent in the system because it has no dependencies on other agents' outputs and its data source (`ClinicalTrialsClient`) is already fully implemented and tested.

## Inputs from Pipeline State

| Field | Type | Notes |
|-------|------|-------|
| `drug_name` | `str` | Set before graph invocation |
| `disease_name` | `str` | Set before graph invocation |
| `date_before` | `date \| None` | Optional temporal holdout cutoff |

Does not need `drug_profile` or `rich_drug_data` — fully self-contained. All four tools operate directly on the drug and disease name strings via the ClinicalTrials.gov API.

## Tools

Each tool wraps an existing `ClinicalTrialsClient` method. The client methods are already implemented and tested — the tools are thin adapters that make them available to the LLM via LangGraph's tool-calling interface.

### `detect_whitespace(drug, condition) → dict`

Checks if the drug-condition pair has been explored. Runs three concurrent API queries internally (exact match, drug-only count, condition-only count). When whitespace exists, also populates a list of other drugs being tested for the same condition.

This is almost always the first tool the agent should call — it determines the entire reasoning path.

### `search_trials(drug, condition) → list[dict]`

Fetches trial records for the drug-condition pair. Returns structured data including phase, status, enrollment, sponsor, interventions, and outcomes. Useful when `detect_whitespace` shows trials exist and the agent needs details.

### `get_landscape(condition) → dict`

Competitive landscape — all drug/biologic trials for this condition grouped by sponsor + drug, ranked by phase then enrollment. Tells the agent how crowded the space is and who the major players are.

The full response can include up to 50 competitors with nested data. The tool should summarize to top 10 competitors to keep the LLM context manageable. The full data is still captured in the output model for downstream use.

### `get_terminated(query) → list[dict]`

Searches for terminated, withdrawn, or suspended trials. Each result includes a keyword-based `stop_category` (safety, efficacy, enrollment, business, other, unknown). The `query` parameter is free text — the agent can search by drug name, disease name, or both.

This is the key adversarial signal: if prior attempts at this drug-disease combination were stopped for safety or efficacy, that's a strong negative indicator.

### Tool implementation notes

- Tools accept primitive types (strings) since the LLM needs to provide the arguments
- Tools return dicts via `model_dump()` so the LLM can read the results
- Each tool creates its own `async with ClinicalTrialsClient()` for session management
- `date_before` for temporal holdout support needs to be passed through — either as a tool parameter or captured via closure when building the tools

## Agent Behavior

ReAct agent using Haiku with the four tools above. The LLM decides which tools to call and in what order based on intermediate observations.

### Reasoning patterns

**Trials exist (active space):**
```
detect_whitespace → sees matches → search_trials for details → get_landscape for competitive picture → return
```

**Whitespace (no trials):**
```
detect_whitespace → no matches → get_terminated to check for prior failures → get_landscape to see what else is in the space → return
```

**Prior failure found:**
```
detect_whitespace → no matches → get_terminated → finds trials stopped for efficacy/safety → return early (strong negative signal, landscape less relevant)
```

The agent isn't locked into these patterns — they're the expected common cases. The LLM may skip tools if earlier results provide enough signal (e.g., `detect_whitespace` already populates `condition_drugs` when whitespace exists, so `get_landscape` may be redundant).

### System prompt

The system prompt should:
- Tell the agent its role (clinical trial landscape analyst)
- Explain the drug repurposing context (assessing whether a known drug has potential in a new indication)
- Suggest starting with `detect_whitespace` and branching from there
- Ask for a 2-3 sentence summary assessment at the end
- Instruct it to return structured JSON matching `ClinicalTrialsOutput`

### LLM choice

Haiku. This agent runs once per drug-disease pair — with 15 candidate diseases per drug, that's 15 invocations. Haiku keeps cost and latency manageable. The tool-calling decisions here are straightforward and don't require Sonnet-level reasoning.

## Output

```
python
class ClinicalTrialsOutput(BaseModel):
    trials: list[Trial] = []
    whitespace: WhitespaceResult | None = None
    landscape: ConditionLandscape | None = None
    terminated: list[TerminatedTrial] = []
    summary: str = ""  # 2-3 sentence natural language assessment from the agent
```

The output model stores the raw data returned by each tool alongside the agent's natural language summary. Downstream consumers (SafetyAgent, final synthesizer) can use either the structured data or the summary depending on their needs.

Fields are `None` or empty when the agent chose not to call the corresponding tool — this is expected behavior, not an error.

## Pipeline State

```
python
class PipelineState(TypedDict, total=False):
    drug_name: str
    disease_name: str
    date_before: date | None
    drug_profile: DrugProfile | None
    rich_drug_data: RichDrugData | None
    clinical_trials_output: ClinicalTrialsOutput | None
    # other agent outputs added as they are built
    errors: Annotated[list[str], operator.add]
```

The state is a LangGraph `TypedDict`. Each node receives the full state and returns a partial dict with just the fields it wants to update. LangGraph handles merging the partial update back into the full state.

`errors` uses `Annotated[list[str], operator.add]` so errors from multiple nodes are appended rather than overwritten.

## Node Adapter

The node function is the bridge between LangGraph's state-passing convention and the agent's typed `run()` method:

```
python
async def clinical_trials_node(state: PipelineState) -> dict:
    agent = ClinicalTrialsAgent()
    output = await agent.run(
        drug=state["drug_name"],
        disease=state["disease_name"],
        date_before=state.get("date_before"),
    )
    return {"clinical_trials_output": output}
```

The agent class holds the logic; the node function is pure plumbing. This separation means the agent can be tested independently without setting up a LangGraph graph.

## Files to create/modify

| File | What |
|------|------|
| `agents/tools/__init__.py` | Package marker |
| `agents/tools/clinical_trials_tools.py` | Four `@tool` functions wrapping `ClinicalTrialsClient` |
| `agents/models.py` | `ClinicalTrialsOutput` (and later, other agent output models) |
| `agents/state.py` | `PipelineState` TypedDict |
| `agents/clinical_trials.py` | `ClinicalTrialsAgent` class + `clinical_trials_node` function |

Existing files unchanged: `data_sources/clinical_trials.py`, `models/model_clinical_trials.py`, `services/*`.

## Dependencies to add

```
toml
"langgraph>=0.2.0",
"langchain-anthropic>=0.3.0",
"langchain-core>=0.3.0",
```

## Graph Position

The agent is independent — it reads only `drug_name`, `disease_name`, and `date_before` from state, none of which are written by other agents. This means it can run in parallel with MechanismAgent and LiteratureAgent if the orchestrator graph supports it.

The graph is invoked once per drug-disease pair. The outer loop that iterates over candidate diseases lives outside the graph in plain Python:

```
python
candidates = await svc.get_drug_competitors(drug_name)
drug_profile = await svc.build_drug_profile(drug_name)

for disease in candidates:
    result = await pipeline.ainvoke({
        "drug_name": drug_name,
        "disease_name": disease,
        "drug_profile": drug_profile,
    })
```

`get_drug_competitors` and `build_drug_profile`   
are per-drug operations that run once outside the graph. Only `disease_name` changes between invocations.