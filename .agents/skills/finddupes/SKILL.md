---
name: finddupes
description: Find duplicated or near-duplicated logic in the codebase (repeated validation blocks, near-identical client methods, copy-pasted parsing/normalization). Read-only report, no auto-fix.
---

# finddupes

Scan `src/indication_scout/` for duplicated or near-duplicated code. Report only — never edit files.

## Steps

1. Determine scope: default to `src/indication_scout/`, or a path/module the user names.
2. Launch an Explore agent (or general-purpose if the scope is broad) to search for:
   - Structurally identical or near-identical functions/methods across files (e.g. the same
     parsing loop repeated in two `data_sources/` clients).
   - Repeated Pydantic validation logic that should be a shared validator (note: `coerce_nones`
     is *expected* to be repeated per-model per AGENTS.md — do not flag that pattern itself, only
     flag other duplicated validators/logic).
   - Copy-pasted normalization/formatting code (string cleanup, date parsing, ID extraction) that
     appears in more than one module.
   - Near-identical test setup/fixtures duplicated instead of shared via `conftest.py`.
3. For each candidate, confirm both locations by reading the actual files — don't rely on the
   agent's summary alone for anything you'll report.
4. Report as a table: file:line pairs, a one-line description of the overlap, and a suggested
   consolidation point (helper function, shared base method, fixture) — but do not implement it
   unless asked.

## Output format

| Location A | Location B | Overlap | Suggested fix |
|---|---|---|---|
| `path/file.py:42` | `path/other.py:88` | same X | extract to Y |

Keep descriptions short. No prose write-up, no editorializing about severity.
