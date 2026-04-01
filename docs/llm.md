# LLM Invocations

All LLM calls go through two wrapper functions in `src/indication_scout/services/llm.py`. There are no direct Anthropic SDK calls elsewhere in the codebase.

## Core LLM Service

**File:** `src/indication_scout/services/llm.py`

| Line | Function | Model |
|------|----------|-------|
| 31 | `query_llm()` | `claude-sonnet-4-6` (main) |
| 42 | `query_small_llm()` | `claude-haiku-4-5-20251001` (small) |

Both use `AsyncAnthropic`, `temperature=0`, `max_tokens=1024`.

## Callers

### `src/indication_scout/services/disease_helper.py` — 3 calls (all small model)

| Line | Function | Purpose |
|------|----------|---------|
| 58 | `llm_normalize_disease()` | Normalizes raw disease terms into PubMed search terms |
| 73 | `merge_duplicate_diseases()` | Merges duplicate disease names, removes overly broad terms |
| 154 | `llm_normalize_disease()` | Generalizes to a broader disease category when initial results are sparse |

### `src/indication_scout/services/retrieval.py` — 5 calls

| Line | Function | Model | Purpose |
|------|----------|-------|---------|
| 67 | `_normalize_disease_groups()` | Small (indirect) | Normalizes disease names via `llm_normalize_disease` before merging competitor groups |
| 437 | `synthesize()` | Main | Summarizes PubMed abstracts into structured evidence summaries |
| 465 | `extract_organ_term()` | Small | Extracts primary organ/tissue for a disease |
| 541 | `expand_search_terms()` | Small | Generates diverse PubMed search queries |
| 579 | `get_disease_synonyms()` | Small | Generates disease synonyms |

## Summary

- **8 total LLM call sites** across 3 files
- 7 out of 8 use the small model (Haiku); only `synthesize()` uses the main model (Sonnet)
- All calls are async and use the centralized wrappers in `llm.py`

## Configuration

Models are configured in `src/indication_scout/config.py` (lines 28–29):

```python
llm_model: str = "claude-sonnet-4-6"
small_llm_model: str = "claude-haiku-4-5-20251001"
```
