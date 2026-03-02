# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Session Startup

At the start of every session, read `PROJECT_STATE.md`, the most recent `session_*.md` file in the project root, and `README.md`.

## Session File Workflow

- Session files are named `session_{datetime}.md` (e.g. `session_2026-02-28_14-31.md`) and live in the project root.
- Append to the current session file throughout the session.

## Build & Development Commands

```bash
# Install (editable)
pip install -e .            # runtime only
pip install -e ".[dev]"     # with dev tools (pytest, black, ruff, mypy)

# Run tests
pytest                              # all tests (unit + integration)
pytest tests/unit/                  # unit tests only
pytest tests/unit/test_pubmed.py    # single file
pytest tests/unit/test_pubmed.py::test_parse_article  # single test

# Lint & format
black src/ tests/
ruff check src/ tests/              # lint
ruff check --fix src/ tests/        # lint + autofix
mypy src/                           # type checking

# Run the API
uvicorn indication_scout.api.main:app --reload

# CLI
scout find -d "metformin"
```

## Architecture

IndicationScout is an agentic drug repurposing system. A drug name goes in; coordinated AI agents query multiple biomedical data sources and produce a repurposing report.

### Layered structure (`src/indication_scout/`)

- **data_sources/** — Async API clients for external biomedical databases. Each extends `BaseClient` (async context manager with retry/backoff). Current clients: `OpenTargetsClient` (GraphQL), `ClinicalTrialsClient` (REST), `PubMedClient` (REST+XML), `ChEMBLClient`, `DrugBankClient`. Errors surface as `DataSourceError`.
- **models/** — Pydantic `BaseModel` contracts between data sources and agents. Organized per source: `model_open_targets.py` (TargetData, DrugData and their nested models), `model_clinical_trials.py` (Trial, WhitespaceResult, ConditionLandscape, TerminatedTrial), `model_pubmed.py` (PubMedArticle). Agents never see raw API responses.
- **agents/** — AI agents that each own a slice of analysis. All extend `BaseAgent` (single `async run()` method). Agents: `orchestrator`, `literature`, `clinical_trials`, `mechanism`, `safety`. The `Orchestrator` coordinates the specialist agents.
- **services/** — Business logic layer (LLM calls, embeddings, scoring). Currently scaffolded.
- **api/** — FastAPI app (`api/main.py`). Routes in `api/routes/`, request/response schemas in `api/schemas/`.
- **config.py** — `pydantic_settings.BaseSettings` loaded from `.env`. Access via `get_settings()`.
- **constants.py** — All magic numbers, URLs, and lookup maps.

### Data flow

```
CLI/API → Orchestrator → specialist agents → data source clients → external APIs
                                          ↕
                                    Pydantic models (models/)
```


### Key patterns

- **BaseClient** (`data_sources/base_client.py`): All clients use `async with Client() as c:` for session lifecycle. Provides `_rest_get()`, `_graphql()`, `_rest_get_xml()` with retry. Subclasses set `_source_name` property.
- **Open Targets** uses file-based caching (`_cache/` dir, 5-day TTL) to avoid redundant GraphQL calls.
- **pytest-asyncio** is set to `asyncio_mode = "auto"` — async test functions run automatically without `@pytest.mark.asyncio`.

## Test Layout

```
tests/
├── conftest.py              # shared fixtures (sample_drug, sample_indication)
├── unit/                    # no network, no external deps
│   └── test_<module>.py
└── integration/             # hits real external APIs
    ├── conftest.py          # client fixtures (open_targets_client, pubmed_client, etc.)
    └── test_<source>.py
```

## Project Rules
- Always use the **logging** module; never use `print()` (including in tests).
- Use `logging.getLogger(__name__)` or follow the existing `indication_scout.<module>` naming convention.
- Do not remove existing code unless explicitly asked.
- Do not refactor or "clean up" code unless explicitly asked.
- Prioritize **correctness** and safety over speed or brevity of implementation.
- Prefer small, incremental changes over large rewrites, unless explicitly requested.
- Do NOT modify existing logic without explicitly asking me first.
- When you identify a logic change is needed, describe what you want to change and wait for approval.
- Only make the specific changes I request - no "while I'm here" improvements.
- If you see potential bugs or improvements, list them separately rather than fixing automatically.
- When asked to make changes, only make those changes, do not introduce new functionality without getting approval.

## Planning & Implementation Rules

Before implementing any feature that depends on external data:

1. **Verify API endpoints exist** - Check the actual API documentation or make a test request
2. **Confirm field names** - Don't assume field names; inspect actual response shapes
3. **Document assumptions** - List any assumptions made and how they were validated

When creating a plan:
- [ ] Identify all external dependencies (APIs, databases, services)
- [ ] Verify each dependency's schema/contract before writing code
- [ ] If you can't verify, explicitly flag it as an assumption that needs human confirmation

## Plan Tracking Workflow
- When asked to design or plan a feature, write the plan to `PLAN.md` in the **project root** as a checklist (`- [ ]` items) before touching any code.
- While implementing, mark each step complete (`- [x]`) in `PLAN.md` as it is finished.
- After a feature is done, leave `PLAN.md` in place as a record; overwrite it when the next plan starts.

## Design & Implementation Guidelines
- Favor clear, explicit control flow over clever patterns.
- When refactoring, preserve existing behavior; avoid changing public interfaces unless requested.
- Before introducing new abstractions (classes, services, helpers), explain:
  - The current pain point or duplication it addresses.
  - How the change will be tested and validated.
- Prefer dependency injection and pure functions where feasible, to simplify testing and reasoning.
- Keep domain logic (clinical/genomic reasoning) separate from I/O, UI, and orchestration code.
- When modifying or adding configuration, avoid implicit defaults that might affect scientific outputs.
- Magic numbers and other constants should be in `constants.py`.
- Do not include any fallback values, default cases, or 'TODO' comments. If a required value is missing, ask me, I will decide on outcome.
- Use Python 3.10+ type hint syntax (`dict[str, Any]`, `int | None`). All function parameters and return types must be annotated.
- All external data contracts must use Pydantic `BaseModel`. Do not use raw dicts for structured data crossing module boundaries.
- All data source clients must extend `BaseClient` (async context manager, lazy session via `_get_session()`).
- Raise `DataSourceError` (not generic exceptions) from data source clients. Include source name and context in the error.

## Pydantic Defensive Defaults

**Every** Pydantic model that ingests external data (APIs, DBs, parsed JSON) must include this `model_validator`
— it does not inherit to nested models, so add it to every model in the hierarchy:
```
@model_validator(mode="before")
@classmethod
def coerce_nones(cls, values):
    for field_name, field_info in cls.model_fields.items():
        if values.get(field_name) is None and field_info.default is not None:
            values[field_name] = field_info.default
    return values
```

This keeps factory methods and downstream code clean — no `or []` / `or ""` guards needed.

**Factory classmethods** should still:
- Guard comprehensions with `if value` to exclude empty strings from output lists.
- Use `list(dict.fromkeys(...))` for order-preserving dedup.
- Build fields inline in the constructor.

### Example
```
# API returns: {"name": null, "synonyms": null, "year_approved": null}

class Drug(BaseModel):
    name: str = ""
    synonyms: list[str] = []
    year_approved: int | None = None  # genuinely optional — stays None

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values):
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values

drug = Drug(**{"name": None, "synonyms": None, "year_approved": None})
# drug.name == ""
# drug.synonyms == []
# drug.year_approved == None  (default is None, so left alone)
```

## Architecture Rules
- If a change touches more than one of these layers, clearly separate:
  - Prompt-level changes.
  - Service/orchestration changes.
  - Evidence or data-processing changes.

## CRITICAL RULES
- NEVER create fallback or default values for scientific or clinical data (e.g. IC50, prevalence, scores, probabilities, clinical labels).
- If required data is unavailable, return `None`/null/empty structures as appropriate; do not fabricate or guess values.
- This is a clinical genomics tool. Fake or inferred data not explicitly supported by upstream sources can harm patients.
- If any code is changed:
  - Remove associated dead code, unused variables, and unused constants that are now provably unreachable.
  - Do not remove code that might still be used indirectly without first verifying usage (e.g. via search, tests, or references).
  - Listen carefully to what the client says. If she asks for a list of things, make sure to implement them all

