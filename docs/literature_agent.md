# Literature Agent

Assesses published PubMed evidence for a drug-disease pair. Returns a `LiteratureOutput`
carrying the structured `EvidenceSummary` (strength, direction, study count, key findings,
PMID buckets, safety fields) plus the intermediate artifacts from the run (queries, PMIDs,
ranked abstracts).

---

## Architecture

```
build_literature_agent()
    +-- build_literature_tools()  <-- closure-scoped wrappers around RetrievalService
    +-- build_gated_react_loop()  (agents/_react_loop.py)
         model node: ChatAnthropic + tools, with Anthropic prompt-caching breakpoints
         tools node: build_drug_profile -> expand_search_terms -> fetch_and_cache
                      -> semantic_search -> safety_search -> synthesize -> finalize_analysis
         loop ends the turn finalize_analysis succeeds (no discarded trailing turn)
run_literature_agent()
    +-- ainvoke() the compiled graph
    +-- walk result["messages"], pull each tool's typed artifact off msg.artifact
    +-- assemble LiteratureOutput
```

### Files

| File | Role |
|---|---|
| `agents/literature/literature_agent.py` | `build_literature_agent()` / `run_literature_agent()` |
| `agents/literature/literature_tools.py` | `@tool`-decorated wrappers around `RetrievalService`, closure-scoped store |
| `agents/literature/literature_output.py` | `LiteratureOutput` — the structured return value |
| `agents/literature/pubmed_ae.py` | Citation-ranked adverse-event PubMed search used by `safety_search` |
| `agents/_react_loop.py` | Shared gated ReAct loop + Anthropic prompt-caching helpers (also used by `clinical_trials` and `supervisor`) |
| `services/retrieval.py` | `RetrievalService` — executes every tool operation |
| `models/model_evidence_summary.py` | `EvidenceSummary` — the structured evidence output |
| `models/model_drug_profile.py` | `DrugProfile` — input to query expansion and safety search |
| `prompts/literature.txt` | System prompt |

---

## Entry Point

```python
def build_literature_agent(llm, svc, db, date_before=None, approved_indications=None)
async def run_literature_agent(agent, drug_name: str, disease_name: str) -> LiteratureOutput
```

**`build_literature_agent` inputs:**

| Arg | Type | Required |
|---|---|---|
| `llm` | LangChain chat model | Yes |
| `svc` | `RetrievalService` | Yes — shared across tool calls in the run |
| `db` | SQLAlchemy `Session` | Yes — connected to the pgvector DB, one per call (not shared across concurrent candidates) |
| `date_before` | `date \| None` | No — temporal holdout cutoff |
| `approved_indications` | `list[str] \| None` | No — drug's FDA-approved indications, forwarded to `synthesize` so the strength judge excludes papers about an already-approved sub-indication |

`svc`, `db`, `date_before`, and `approved_indications` are captured via closure at tool-build
time, so the LLM never sees them as tool parameters.

**Output:** `LiteratureOutput` — see Data Models below.

Called from `agents/supervisor/supervisor_tools.py` (`analyze_literature` tool, one fresh
agent + DB session per drug-disease call) and `services/analysis_runner.py`.

---

## ReAct Loop

The agent runs on `build_gated_react_loop()` (`agents/_react_loop.py`), the same construct
used by the `clinical_trials` and `supervisor` agents: a minimal two-node LangGraph
(model + tools) that mirrors `create_react_agent`, except the loop ends as soon as
`finalize_analysis` succeeds instead of feeding its result back to the model for a discarded
trailing turn. `literature`'s `finalize_analysis` tool is `return_direct=True` with no reject
path, so termination is unconditional on that tool succeeding.

Two Anthropic prompt-caching breakpoints keep repeat turns cheap: one on the static system
prompt + tool definitions (`cached_system_message`), one on the tail of the growing message
history (`_with_history_breakpoint`), so each turn reprocesses only what changed.

The system prompt (`prompts/literature.txt`) instructs a fixed call sequence with no
branching, and requires `finalize_analysis` as the last tool call in every run — including
every empty-result case:

1. `build_drug_profile` — fetch drug/target/ATC pharmacology
2. `expand_search_terms` — generate PubMed queries
3. `fetch_and_cache` — run queries, embed abstracts, store in pgvector
4. `semantic_search` — retrieve top-k abstracts by similarity
5. `safety_search` — drug-level + disease-specific safety (REQUIRED; see "Drug Safety" in ARCHITECTURE.md)
6. `synthesize` — produce structured `EvidenceSummary` (merges in the safety fields)
7. `finalize_analysis` — termination signal; carries the narrative summary as its artifact

If no evidence is found, `synthesize` and `finalize_analysis` are still called.

---

## Tools

Tools are thin async wrappers around `RetrievalService` methods, defined in
`agents/literature_tools.py` via `build_literature_tools(svc, db, date_before, approved_indications)`.
All use `@tool(response_format="content_and_artifact")` so the typed return value survives on
`ToolMessage.artifact` (the string content is only the LLM-facing summary). Tools share
inter-call data through a closure-scoped `store` dict — the LLM never passes PMIDs, queries,
or abstracts between calls.

### `build_drug_profile(drug_name) -> DrugProfile`

Resolves the ChEMBL ID and fetches the pharmacological profile (gene targets, mechanisms,
ATC codes). Stores the profile for reuse by later tools.

Calls `RetrievalService.build_drug_profile()`.

### `expand_search_terms(drug_name, disease_name) -> list[str]`

Generates diverse PubMed keyword queries using the drug profile (building one on the fly if
`build_drug_profile` wasn't called first).

Calls `RetrievalService.expand_search_terms()`.

### `fetch_and_cache(drug_name) -> list[str]`

Runs PubMed searches for the queries in the store, embeds abstracts with BioLORD-2023, and
caches them in pgvector. Returns deduplicated PMIDs. No-ops with a warning if
`expand_search_terms` hasn't run.

Calls `RetrievalService.fetch_and_cache()`. The `date_before` cutoff is applied here via
closure.

### `semantic_search(drug_name, disease_name) -> list[AbstractResult]`

Re-ranks the cached abstracts (PMIDs from the store) by similarity to the drug-disease query.
No-ops with a warning if `fetch_and_cache` hasn't run.

Calls `RetrievalService.semantic_search()`.

### `safety_search(drug_name, disease_name) -> EvidenceSummary`

REQUIRED step. Produces a **two-tier** safety signal (full design in ARCHITECTURE.md → "Drug
Safety"): a DRUG-LEVEL blurb (`RetrievalService.safety_search` + `summarize_safety`,
OT-anchored in production / date-filtered literature in holdout, with `safety_severity`) and a
DISEASE-SPECIFIC `indication_harm` classification (`classify_indication_harm`). Independent of
the efficacy PMID pool — it runs its own citation-ranked adverse-event PubMed queries
(`agents/literature/pubmed_ae.py::search_adverse_events`). Stores all six safety/harm fields so
`synthesize` can merge them in. Empty when there is no signal (never a fabricated "safe"
verdict).

### `synthesize(drug_name, disease_name) -> EvidenceSummary`

Passes the ranked abstracts (from the store) and `approved_indications` to the LLM and
returns a structured `EvidenceSummary`. If `safety_search` already populated the store, its
six safety/harm fields are merged onto the result — order-independent even if the LLM calls
`synthesize` before `safety_search` in a given run.

Calls `RetrievalService.synthesize()`.

### `finalize_analysis(summary: str) -> str`

`return_direct=True`. Signals the run is complete; the `summary` argument becomes the
artifact used to populate `LiteratureOutput.summary`. Ends the gated loop unconditionally —
literature has no reject path.

---

## Result Assembly

`run_literature_agent()` walks `result["messages"]` after `ainvoke()` completes. Each
`ToolMessage` with a name in a fixed `field_map` (`expand_search_terms` → queries,
`fetch_and_cache` → pmids, `semantic_search` → abstracts, `synthesize` → evidence,
`finalize_analysis` → summary) has its `.artifact` read directly into the corresponding
`LiteratureOutput` field — no message-content parsing or JSON decoding. Missing queries,
PMIDs, or evidence are each logged as a warning rather than raising.

It also logs a per-turn token/cache accounting pass over the `AIMessage`s in the run (input
tokens, output tokens, cache-read/cache-write hits) to isolate whether loop overhead comes
from turn count, output size, or prompt-caching misses — currently commented out at the
`logger.info` call sites, kept for ad hoc debugging.

---

## Data Models

### `LiteratureOutput`

**File:** `agents/literature/literature_output.py`

The full return value of a run:

| Field | Type | Source |
|---|---|---|
| `search_results` | `list[str]` | `expand_search_terms` |
| `pmids` | `list[str]` | `fetch_and_cache` |
| `semantic_search_results` | `list[AbstractResult]` | `semantic_search` |
| `evidence_summary` | `EvidenceSummary \| None` | `synthesize` (merged with `safety_search`) |
| `summary` | `str` | `finalize_analysis` |

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

Flat LLM-facing projection of `RichDrugData`. Key fields: `chembl_id`, `target_gene_symbols`,
`mechanisms_of_action`, `atc_codes`, `atc_descriptions`, `drug_type`. Also carries the
OpenTargets safety signal — `drug_warnings` (`list[DrugWarning]`, black-box / withdrawn) and
`adverse_events` (`list[AdverseEvent]`, FAERS with `log_likelihood_ratio`) — used by
`safety_search` to build targeted PubMed provenance queries. Built via
`DrugProfile.from_rich_drug_data()` or `RetrievalService.build_drug_profile()`.

---

## Dependencies

| Component | Role |
|-----------|------|
| `RetrievalService` | Executes all tool operations (query expansion, fetch, search, safety, synthesis) |
| `build_gated_react_loop` (`agents/_react_loop.py`) | Shared ReAct loop + prompt-caching, also used by `clinical_trials` and `supervisor` |
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
| Finalize reject path | Yes — empty-summary / critique-not-run loops back to the model | None — `finalize_analysis` is `return_direct=True`, terminates unconditionally |
| Output model | `ClinicalTrialsOutput` (multi-field, built from `FinalizeClinicalTrialsArtifact`) | `LiteratureOutput` (flat, wraps `EvidenceSummary` + intermediate artifacts) |
| Output model file | `agents/clinical_trials/clinical_trials_output.py` | `agents/literature/literature_output.py` |
| Additional inputs | `date_before`, `assigned_indication` | `svc`, `db`, `date_before`, `approved_indications` |
| Loop construct | `build_gated_react_loop` | `build_gated_react_loop` (same shared helper) |

---

## Test Layout

```
tests/
+-- unit/agents/
|   +-- test_literature_agent.py    # tests run_literature_agent's artifact assembly with fake message histories
|   +-- test_literature_tools.py    # mocked RetrievalService, verifies tool return shapes
+-- integration/agents/literature/
|   +-- test_literature_agent.py
|   +-- test_literature_tools.py
+-- integration/services/
|   +-- test_literature_strength.py
+-- trial_risk/
|   +-- test_literature.py          # unrelated: trial_risk ML model's literature-signal features
```
