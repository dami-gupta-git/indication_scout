# LLM Invocations

All LLM calls go through two wrapper functions in `src/indication_scout/services/llm.py`. There are no direct Anthropic SDK calls elsewhere in the codebase.

## Core LLM Service

**File:** `src/indication_scout/services/llm.py`

| Line | Function | Model |
|------|----------|-------|
| 113 | `query_llm()` | `claude-sonnet-4-6` (main) |
| 128 | `query_small_llm()` | `claude-haiku-4-5-20251001` (small) |

Both use `AsyncAnthropic`, `temperature=0`. Token limits come from settings:
`llm_max_tokens` for the main model and `small_llm_max_tokens` for the small
model (configured via `.env.constants`). System prompts are optional; an empty
string is passed as `NOT_GIVEN`. Empty content responses raise `DataSourceError`.

This module also exposes JSON-parsing helpers used by callers:
`strip_markdown_fences()`, `parse_llm_response()`, `parse_last_json_array()`,
`parse_last_json_object()`.

## Callers

### `src/indication_scout/services/disease_helper.py` — 3 calls (all small model)

| Line | Function | Purpose |
|------|----------|---------|
| 70 | `llm_normalize_disease()` | Normalizes a raw disease term into a PubMed search term |
| 107 | `llm_normalize_disease_batch()` | Normalizes multiple disease terms in one batched LLM call |
| 159 | `merge_duplicate_diseases()` | Merges duplicate disease names and removes overly broad terms |

### `src/indication_scout/services/retrieval.py` — 3 calls

| Line | Function | Model | Purpose |
|------|----------|-------|---------|
| 632 | `synthesize()` | Main | Summarizes top PubMed abstracts into a structured `EvidenceSummary` |
| 662 | `extract_organ_term()` | Small | Extracts the primary organ/tissue for a disease |
| 743 | `expand_search_terms()` | Small | Generates diverse PubMed search queries from drug × disease |

### `src/indication_scout/services/approval_check.py` — 3 calls

| Line | Function | Model | Purpose |
|------|----------|-------|---------|
| 303 | `list_approved_indications_from_labels()` | Main | Extracts approved indications from raw FDA label text |
| 369 | `extract_approved_from_labels()` | Small | Identifies which candidate diseases appear as approved indications in label text |
| 528 | `get_fda_approved_disease_mapping()` | Main | Per-candidate approved/not-approved verdict for missing candidates |

## Summary

- **9 total LLM call sites** across 3 caller files
- 6 of 9 use the small model (Haiku); 3 use the main model (Sonnet) — `synthesize()`, `list_approved_indications_from_labels()`, and `get_fda_approved_disease_mapping()`
- All calls are async and use the centralized wrappers in `llm.py`

## Configuration

Models are configured in `src/indication_scout/config.py` (lines 40–41):

```python
llm_model: str = "claude-sonnet-4-6"
small_llm_model: str = "claude-haiku-4-5-20251001"
```

Token limits are required (no defaults) and loaded from `.env.constants`:

```python
llm_max_tokens: int
small_llm_max_tokens: int
```
