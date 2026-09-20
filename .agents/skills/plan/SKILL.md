---
name: plan
description: Write a plan, design doc, or spec for a feature before any code is written. Produces PLAN_<name>.md as a phased checklist where every phase states how it will be verified. Use when asked to plan, design, spec, or scope a feature or change.
---

# plan

Turn a feature request into a written plan before touching code. Nothing in this skill edits
source files.

## Steps

1. **Scope.** Restate the feature in two or three sentences: what it does, what it does not do,
   and which layers it touches (`data_sources/`, `models/`, `agents/`, `services/`, `api/`, CLI).
   If the request is ambiguous in a way that changes the design, ask in plain text — one question
   at a time — before writing anything.

2. **Verify external dependencies.** Apply the verification rules in AGENTS.md to every API,
   database, or service the feature depends on. Anything that cannot be verified goes in the plan
   as a flagged assumption, never as established fact.

3. **Survey the existing code.** Name the files the plan will touch and the existing patterns it
   must follow. A plan that reinvents an existing helper is a defect.

4. **Write the plan** to `PLAN_<name>.md` in the project root, in the format below.

5. **Stop.** Do not implement until told to. Once implementing, tick items off in the plan file as
   each is finished.

## Plan format

Phased checklist. Each phase is a group of related items that lands as one coherent change.

```markdown
# PLAN_<name>

## Scope
<Two or three sentences: what this does, what it excludes, which layers it touches.>

## Assumptions
<Each external contract verified, and how. Each unverified assumption, flagged as such.
Omit this section only if the feature touches nothing external.>

## Phase 1 — <short verb phrase>
- [ ] <item>
- [ ] <item>

**Verification:** <see below>

## Phase 2 — <short verb phrase>
...
```

### Verification is mandatory

Every phase carries a `**Verification:**` block. It is not optional and it is not a placeholder.
It states:

- **What to write** — the specific test, named, with the file it belongs in
  (`tests/unit/services/test_foo.py`, `tests/integration/data_sources/test_bar.py`).
- **Which CI or tooling changes are needed** — a new marker, a fixture, a cassette, a lint or
  type-check target. Write "none" if none.
- **What "working" looks like** — the observable result that distinguishes a passing phase from a
  failing one. "The client returns a populated `MoleculeData` for a known ChEMBL ID and raises
  `DataSourceError` for an unknown one" is a verification. "Tests pass" is not.

If a phase genuinely cannot be covered by an automated test — a prompt wording change, a report
formatting judgement — say so, and give the concrete manual check instead: the command to run, the
input to use, and what to look for in the output.

## Design docs

If the request is for a design document rather than an implementation checklist, write prose and
follow the design-doc rules in the global AGENTS.md.

A design doc still names how the design will be verified, in its own section, at the same level of
concreteness the checklist format requires.

## Rules

- No 'TODO' items. If something is undecided, ask.
- Prefer a smaller correct plan over a larger speculative one.
