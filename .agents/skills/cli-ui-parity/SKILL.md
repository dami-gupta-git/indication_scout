---
name: cli-ui-parity
description: Check that the CLI, the HTTP API, and the React front end still expose the same surface. Reports drift only — never edits. Use when the CLI or API has changed, before a release, or when asked whether the UI is behind the CLI.
disable-model-invocation: true
---

# CLI / API / UI Parity

The same capability has to be declared in four places, and nothing enforces agreement:

| Layer | Location |
|---|---|
| CLI commands and options | `src/indication_scout/cli/cli.py` |
| API request/response schemas | `src/indication_scout/api/schemas/`, routed in `api/routes/` |
| TS mirror of those schemas | `frontend/src/types.ts` (hand-written, header names its sources) |
| API client and UI controls | `frontend/src/api.ts`, `frontend/src/App.tsx` |

Two seams drift: CLI-to-API (a flag the API can't accept) and API-to-TS (a schema field the mirror lacks).

## Process

### 1. Read the CLI surface

List every command and, per command, every option: name, type, required, default. Include the group-level options.

### 2. Read the API surface

List the routes and, per route, the request body fields and the response model. Note which CLI commands have no route at all.

### 3. Read the TS mirror

`frontend/src/types.ts` is maintained by hand against the Pydantic models its header lists. Compare field-by-field against those models: missing fields, extra fields, and nullability that disagrees.

Two conventions bind the comparison, both from the header:
- Backend `T | None` is `T | null` here.
- Fields with Pydantic defensive defaults (list, dict, str, int) are always present and never null, so `T | null` on one of those is itself the drift.

### 4. Read the client and controls

Check which request fields `api.ts` actually sends, and which of those the UI lets a user set. A field the schema accepts but no control populates is unreachable capability.

### 5. Report

Under `## Missing in API`, `## Missing in UI`, `## Type mirror drift`, and `## Unreachable`, one line per finding naming the layer that has it and the layer that lacks it. State what a user can do in one layer and not the other — that is the finding, not the symbol names.

Close with a one-line verdict: in parity, or the count per category.

Rank by what a user would notice. A CLI command absent from the UI outranks a nullability mismatch on a display-only field.

## Rules

- Read-only. Report drift; never add the missing option, field, or control.
- Deliberate asymmetry is not drift. Developer-facing and file-path options (report output directories, stdout switches, golden-file comparison) have no business in a browser. Say so once and stop re-reporting them.
- A type-mirror mismatch is a real defect even when nothing visibly breaks: it silently misleads the next person reading the TS.
- Never infer a field's meaning from its name alone. Read the model.
