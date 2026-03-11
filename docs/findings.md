# Findings

Important discoveries, decisions, and patterns encountered during development.
Each entry is dated and categorized.

---

## Project Rules

### Never remove TODO comments
- TODO comments must be left in place — do not remove them during code review, cleanup, or refactoring

---

## API & Data Sources

### PubMed model naming (2026-03-01)
- Model is `PubmedAbstract` (not `PubMedArticle`); method is `fetch_abstracts` (not `fetch_articles`)
- File: `models/model_pubmed_abstract.py`

### Disk cache is shared (2026-03-01)
- `utils/cache.py` is used by ALL clients and services, not just OpenTargets
- Any client or service can call `cache_get` / `cache_set` directly

### `.env.example` had wrong embedding model (2026-03-01)
- Was `text-embedding-3-small` (OpenAI); correct value is `FremyCompany/BioLORD-2023`
- Fixed in `.env.example`

---

## Models & Data Contracts

### `expand_search_terms` input type (2026-03-01)
- Takes `DrugProfile`, not `RichDrugData`
- `DrugProfile` is built from `RichDrugData` upstream via `DrugProfile.from_rich_drug_data()`

### `DrugData.atc_classifications` frequently omitted (2026-03-01)
- Field exists and is populated; easy to miss when documenting `DrugData`

### Pydantic `coerce_nones` validator (2026-03-01)
- Every model ingesting external data has a `model_validator(mode="before")` that coerces `None` → default
- Does **not** inherit to nested models — must be added to every model in the hierarchy

---

## Architecture & Patterns

### `markers.py` location (2026-03-01)
- Lives at `src/indication_scout/markers.py` (package root), not in a subdirectory
- Easy to omit from project structure listings

### CLAUDE.md discrepancy (2026-03-01)
- CLAUDE.md references `model_pubmed.py` / `PubMedArticle` — both are wrong
- Actual file/class: `model_pubmed_abstract.py` / `PubmedAbstract`
- CLAUDE.md is read-only; this cannot be fixed there

---

## Prompt Engineering

### `expand_search_terms` axis balance (2026-03-01)
- With 3 target gene queries (PRKAA1, PRKAA2, STK11) on a 9-query cap, one axis dominates
- Mitigation: prompt could cap target gene queries at 2 — or let dedup in `fetch_and_cache` handle overlap
- Decision pending; see `docs/notes.md`