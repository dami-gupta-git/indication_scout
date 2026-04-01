# LiteratureAgent

## Purpose

`LiteratureAgent` finds and synthesizes PubMed evidence for a drug-disease pair. Given a drug name and a candidate indication, it returns an `EvidenceSummary` describing the strength, volume, and nature of published evidence.

It is a **tool-using agent**: rather than executing a fixed retrieval sequence, it runs a tool-use loop against the Anthropic API and decides which `RetrievalService` methods to call, in what order, and whether to retry or broaden the search based on intermediate results.

**File:** [src/indication_scout/agents/literature.py](../src/indication_scout/agents/literature.py)

---

## Inputs and Outputs

**Input (`input_data`):**

| Key | Type | Description |
|-----|------|-------------|
| `drug_name` | `str` | Common drug name (e.g. `"metformin"`) |
| `disease_name` | `str` | Candidate indication (e.g. `"colorectal cancer"`) |
| `drug_profile` | `DrugProfile` | Pre-built drug profile (synonyms, targets, MOA, ATC codes) |

**Output:**

| Key | Type | Description |
|-----|------|-------------|
| `evidence_summary` | `EvidenceSummary` | Synthesized evidence (strength, study count, key findings, PMIDs) |
| `pmids_retrieved` | `list[str]` | All PMIDs fetched and cached during the run |

---

## Tools

The agent exposes four tools, each wrapping a `RetrievalService` method:

### `expand_search_terms`

Generates diverse PubMed keyword queries for the drug-disease pair using the drug profile (synonyms, gene targets, MOA, ATC codes, organ term).

```json
{
  "name": "expand_search_terms",
  "input_schema": {
    "drug_name": "string",
    "disease_name": "string"
  }
}
```

Returns: `list[str]` — 5–10 PubMed queries.

---

### `fetch_and_cache`

Runs PubMed searches for the given queries, embeds abstracts with BioLORD-2023, and stores them in pgvector. Returns the deduplicated union of all PMIDs found.

```json
{
  "name": "fetch_and_cache",
  "input_schema": {
    "queries": ["string"]
  }
}
```

Returns: `{ "pmid_count": int, "pmids": ["string"] }`

The agent uses `pmid_count` to decide whether to broaden search terms and retry.

---

### `semantic_search`

Embeds a query string from the drug-disease pair and retrieves the top-k most similar abstracts from pgvector, restricted to the supplied PMIDs.

```json
{
  "name": "semantic_search",
  "input_schema": {
    "pmids": ["string"],
    "top_k": "integer (default 5)"
  }
}
```

Returns: `list[AbstractResult]` — ranked by cosine similarity, each with `pmid`, `title`, `abstract`, `similarity`.

The agent uses similarity scores to decide whether the retrieved abstracts are relevant enough to synthesize, or whether different queries are needed.

---

### `synthesize`

Passes the retrieved abstracts to an LLM prompt and returns a structured `EvidenceSummary`.

```json
{
  "name": "synthesize",
  "input_schema": {
    "abstracts": [{ "pmid": "string", "title": "string", "abstract": "string", "similarity": "float" }]
  }
}
```

Returns: `EvidenceSummary` — `summary`, `study_count`, `study_types`, `strength` (`"strong"` / `"moderate"` / `"weak"` / `"none"`), `has_adverse_effects`, `key_findings`, `supporting_pmids`.

---

## Decision Logic

The system prompt encodes the branching strategy. The LLM owns these decisions:

**Low hit count:** If `fetch_and_cache` returns fewer than ~20 PMIDs, call `expand_search_terms` again with a broader disease term (e.g. generalize `"non-alcoholic steatohepatitis"` → `"liver disease"`) before proceeding to semantic search.

**Low similarity:** If `semantic_search` returns abstracts with all similarity scores below ~0.6, the evidence pool may be off-target. Try alternative queries (e.g. drug class + organ term instead of drug name + disease name) before calling `synthesize`.

**Sufficient evidence:** Once the agent has high-similarity abstracts in hand, call `synthesize` and return. Do not continue fetching.

**No evidence found:** If multiple query strategies return empty or very low similarity, call `synthesize` with whatever is available. The prompt instructs the LLM to return `strength: "none"` rather than fabricating signal.

---

## Tool-Use Loop

```
User message: "Find literature evidence for {drug} in {disease}."
    │
    ▼
LLM call (with tools)
    │
    ├── tool_use: expand_search_terms → queries
    │
    ▼
LLM call (tool_result appended)
    │
    ├── tool_use: fetch_and_cache(queries) → pmid_count
    │       if pmid_count < threshold:
    │           tool_use: expand_search_terms (broader) → retry
    │
    ▼
LLM call (tool_result appended)
    │
    ├── tool_use: semantic_search(pmids) → abstracts + similarity scores
    │       if all scores < threshold:
    │           tool_use: fetch_and_cache (different queries) → retry semantic_search
    │
    ▼
LLM call (tool_result appended)
    │
    ├── tool_use: synthesize(abstracts) → EvidenceSummary
    │
    ▼
LLM call (tool_result appended)
    │
    stop_reason: "end_turn"
    │
    ▼
Return EvidenceSummary
```

The loop runs until `stop_reason == "end_turn"`. A max-iterations guard prevents runaway loops.

---

## Dependencies

| Component | Role |
|-----------|------|
| `RetrievalService` | Executes all four tool operations; unchanged by this agent |
| `DrugProfile` | Provides synonyms, targets, MOA, ATC codes for query expansion |
| `EvidenceSummary` | Output model (`models/model_evidence_summary.py`) |
| `services/llm.py` | Needs a `query_llm_with_tools()` function that supports the Anthropic `tools` parameter |
| `AsyncAnthropic` client | Tool-use loop is driven directly against the Anthropic messages API |

---

## Comparison to the Current RAG Runner

The current `run_rag()` in [runners/rag_runner.py](../src/indication_scout/runners/rag_runner.py) calls `RetrievalService` methods in a fixed sequence for every disease, regardless of hit counts or similarity scores.

| Behaviour | `run_rag()` | `LiteratureAgent` |
|-----------|-------------|-------------------|
| Query strategy | Fixed 5–10 LLM-generated queries, always | LLM decides; can retry with broader/different terms |
| Low PMID count | Proceeds anyway | LLM can broaden disease term and re-fetch |
| Low similarity | Returns whatever top-5 are | LLM can try different queries before synthesizing |
| Termination | Processes all 15 diseases identically | LLM calls `synthesize` when it judges evidence sufficient |
| Control flow | Hardcoded in Python | Encoded in system prompt |

`RetrievalService` is unchanged — it becomes the tool execution backend for the agent.

---

## What Needs to Be Built

- [ ] `query_llm_with_tools()` in `services/llm.py` — wraps `client.messages.create` with `tools` parameter
- [ ] Tool definitions (JSON schema) for the four tools above
- [ ] `LiteratureAgent.run()` — tool-use loop, tool dispatch, result extraction
- [ ] System prompt for the agent
- [ ] Unit tests: mock tool dispatch, verify branching on low pmid count and low similarity
