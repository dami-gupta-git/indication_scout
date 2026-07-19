# Literature Agent

Assesses published PubMed evidence for a drug-disease pair. Returns a structured
`EvidenceSummary` (strength, study count, key findings, PMIDs) and the full list of
PMIDs retrieved during the run.

---

## Architecture

```
LiteratureAgent.run()
    +-- build_literature_tools()  <-- wraps RetrievalService methods
    +-- LangChain ReAct agent (ChatAnthropic + tools)
         +-- build_drug_profile(drug_name)
         +-- expand_search_terms(drug_name, disease_name)
         +-- fetch_and_cache(queries)
         +-- semantic_search(drug_name, disease_name, pmids)
         +-- safety_search(drug_name, disease_name)   <-- drug-level + disease-specific safety
         +-- synthesize(drug_name, disease_name, abstracts)
    +-- _parse_result()  <-- extracts EvidenceSummary + PMIDs from message history
```

### Files

| File | Role |
|---|---|
| `agents/literature_agent.py` | Agent orchestration, system prompt, result parsing |
| `agents/literature_tools.py` | LangChain `@tool` wrappers around `RetrievalService` |
| `services/retrieval.py` | `RetrievalService` -- executes all four tool operations |
| `models/model_evidence_summary.py` | `EvidenceSummary` -- the structured output model |
| `models/model_drug_profile.py` | `DrugProfile` -- input to query expansion |

---

## Entry Point

```python
class LiteratureAgent(BaseAgent):
    async def run(self, input_data: dict[str, Any]) -> dict[str, Any]
```

**Input:**

| Key | Type | Required |
|---|---|---|
| `drug_name` | `str` | Yes |
| `disease_name` | `str` | Yes |
| `drug_profile` | `DrugProfile` | Yes -- pre-built drug profile (synonyms, targets, MOA, ATC codes) |
| `db` | `Session` | Yes -- active SQLAlchemy session connected to pgvector DB |
| `date_before` | `date \| None` | No -- temporal holdout cutoff |

Both names are lowercased before use. `drug_profile` and `db` are captured via closure
at tool build time, so the LLM never sees them as tool parameters. `date_before` is
similarly captured and applied to `fetch_and_cache`.

**Output:** `{"evidence_summary": EvidenceSummary | None, "pmids_retrieved": list[str]}`

---

## ReAct Loop

The agent uses `langchain.agents.create_agent` with `temperature=0`, `max_tokens=4096`,
and a `RECURSION_LIMIT` of 10. The LLM model is `DEFAULT_LLM_MODEL` (`claude-sonnet-4-6`).

The system prompt instructs a fixed call sequence with no branching:

1. `build_drug_profile` -- fetch drug/target/ATC + OpenTargets safety data
2. `expand_search_terms` -- generate PubMed queries
3. `fetch_and_cache` -- run queries, embed abstracts, store in pgvector
4. `semantic_search` -- retrieve top-k abstracts by similarity
5. `safety_search` -- drug-level + disease-specific safety (REQUIRED; see "Drug Safety" in ARCHITECTURE.md)
6. `synthesize` -- produce structured EvidenceSummary (merges in the safety fields)

If no evidence is found, `synthesize` is still called and returns `strength: "none"`.
The agent ends with a plain-text confirmation that synthesis is complete.

---

## Tools

Tools are thin async wrappers around `RetrievalService` methods, defined in
`agents/literature_tools.py`. Built via `build_literature_tools(drug_profile, db, date_before)`.

The `drug_profile`, `db`, and `date_before` parameters are captured via closure so they
flow to every service call without being exposed as tool parameters to the LLM. A single
`RetrievalService(DEFAULT_CACHE_DIR)` instance is shared across all tools in a run.

### `expand_search_terms(drug_name, disease_name) -> list[str]`

Generates 5-10 diverse PubMed keyword queries using the drug profile (synonyms, targets,
MOA, ATC codes). Queries span 5 axes: drug name, drug class+organ, mechanism+organ,
target gene, synonym.

Calls `RetrievalService.expand_search_terms()`.

### `fetch_and_cache(queries) -> dict`

Runs PubMed searches for each query, embeds abstracts with BioLORD-2023, and stores
them in pgvector. Returns `{"pmid_count": int, "pmids": list[str]}`.

Calls `RetrievalService.fetch_and_cache()`. The `date_before` cutoff is applied here
via closure.

### `semantic_search(drug_name, disease_name, pmids, top_k=5) -> list[dict]`

Retrieves the top-k abstracts from pgvector most similar to the drug-disease query,
restricted to the supplied PMIDs.

Calls `RetrievalService.semantic_search()`.

### `safety_search(drug_name, disease_name) -> EvidenceSummary`

REQUIRED step. Produces a **two-tier** safety signal (full design in ARCHITECTURE.md → "Drug
Safety"): a DRUG-LEVEL blurb (`RetrievalService.safety_search` + `summarize_safety`, OT-anchored in
production / date-filtered literature in holdout, with `safety_severity`) and a DISEASE-SPECIFIC
`indication_harm` classification (`classify_indication_harm`). Independent of the efficacy PMID pool
— it runs its own citation-ranked adverse-event PubMed queries (`agents/literature/pubmed_ae.py`).
The result populates the `safety_*` / `indication_harm*` fields, which `synthesize` merges into the
final `EvidenceSummary`. Empty when there is no signal (never a fabricated "safe" verdict).

### `synthesize(drug_name, disease_name, abstracts) -> dict`

Passes retrieved abstracts to the LLM and returns a structured `EvidenceSummary` via
`model_dump()`. Fields: `strength`, `study_count`, `study_types`, `key_findings`,
`has_adverse_effects`, `supporting_pmids`, `summary`.

Calls `RetrievalService.synthesize()`.

---

## Result Parsing

`_parse_result()` walks the agent's message history after `ainvoke()` completes:

- Messages with a `.name` attribute are tool responses. Content is JSON-decoded if it
  arrives as a string (LangChain may stringify tool return values).
- `synthesize` tool response is deserialized into `EvidenceSummary`.
- `fetch_and_cache` tool response provides the PMID list. PMIDs are deduplicated with
  `dict.fromkeys()` for order preservation.
- Returns `{"evidence_summary": EvidenceSummary | None, "pmids_retrieved": list[str]}`.

If `synthesize` was never called (e.g., recursion limit hit), `evidence_summary` is `None`.

---

## Data Models

### `EvidenceSummary`

**File:** `models/model_evidence_summary.py`

The authoritative field list (with the PMID buckets and the safety fields) is the tree in
ARCHITECTURE.md → "EvidenceSummary". Efficacy fields: `summary`, `study_count`, `strength`,
`direction`, `evidence_basis`, `is_observational`, `is_animal_only`, `key_findings`, and the PMID
buckets. Safety fields (populated by `safety_search`, merged by `synthesize`):

| Field | Type | Description |
|---|---|---|
| `safety_summary` | `str` | Drug-level safety blurb (drug-wide) |
| `safety_pmids` | `list[str]` | PMIDs cited in `safety_summary` |
| `safety_severity` | `Literal["withdrawn","black_box","serious","moderate","none"]` | Drug-level severity |
| `indication_harm` | `bool` | A harm reported for this drug in THIS indication |
| `indication_harm_summary` | `str` | One-line disease-specific harm summary |
| `indication_harm_pmids` | `list[str]` | PMIDs cited for the indication harm |

Has the `coerce_nones` model validator. Also has a `coerce_pmids_to_str` field validator
that converts any non-string PMID values to strings (covers all PMID lists incl. `safety_pmids`,
`indication_harm_pmids`).

### `DrugProfile`

**File:** `models/model_drug_profile.py`

Flat LLM-facing projection of `RichDrugData`. Key fields: `name`, `synonyms`,
`target_gene_symbols`, `mechanisms_of_action`, `atc_codes`, `atc_descriptions`, `drug_type`.
Also carries the OpenTargets safety signal — `drug_warnings` (`list[DrugWarning]`, black-box /
withdrawn) and `adverse_events` (`list[AdverseEvent]`, FAERS with `log_likelihood_ratio`) — used by
`summarize_safety` as the authoritative drug-level signal in production. Built upstream via
`DrugProfile.from_rich_drug_data()` or `RetrievalService.build_drug_profile()`.

---

## Dependencies

| Component | Role |
|-----------|------|
| `RetrievalService` | Executes all tool operations (query expansion, fetch, search, safety, synthesis) |
| `DrugProfile` | Provides drug context for query expansion + the OT safety signal |
| `EvidenceSummary` | Output model |
| `SQLAlchemy Session` | pgvector DB access for abstract storage and retrieval |
| `pubmed_ae.search_adverse_events` | Citation-ranked adverse-event PubMed retrieval (drug-level + disease-scoped) |
| `EuropePMCClient` | Citation counts for ranking safety literature |
| `BioLORD-2023` | Embedding model used by `fetch_and_cache` and `semantic_search` |

---

## Differences from ClinicalTrialsAgent

| Aspect | ClinicalTrialsAgent | LiteratureAgent |
|--------|-------------------|-----------------|
| Data source | `ClinicalTrialsClient` (REST API) | `RetrievalService` (PubMed + pgvector RAG) |
| Tool branching | LLM decides tool order based on whitespace | Fixed sequence, no branching |
| Output model | `ClinicalTrialsOutput` (5 fields, multi-tool) | Flat dict with `EvidenceSummary` + PMID list |
| Output model file | Separate `agents/clinical_trials_model.py` | No separate model file -- uses `EvidenceSummary` from `models/` |
| Additional inputs | `date_before` only | `drug_profile`, `db`, `date_before` |
| LLM | `settings.big_llm_model` | `DEFAULT_LLM_MODEL` (`claude-sonnet-4-6`) |
| Recursion limit | 15 | 10 |

---

## Test Layout

```
tests/
+-- unit/agents/
|   +-- test_literature_agent.py    # tests _parse_result with fake message histories
|   +-- test_literature_tools.py    # mocked RetrievalService, verifies tool return shapes
```
