---
name: writedocs
description: Write or edit any markdown document in this project (README, OVERVIEW, design docs, ARCHITECTURE, ROADMAP, or any other .md file not already owned by a more specific workflow). Use whenever asked to write, update, or clean up a doc. Enforces brevity via a mandatory self-edit pass.
---

# writedocs

## Steps

1. **Check for a more specific owner first.** `PLAN_<name>.md` goes through the `plan` skill.
   `for_me/findings.md`, `PROJECT_STATE.md`, and `session_*.md` follow the workflows in the project
   CLAUDE.md. Use this skill for everything else.

2. **Draft.** Documentation is brief and to the point, and never repeats a statement.
   - Overview-level docs: concepts only, no code blocks, no type signatures, no schema literals.
     Describe a data shape in a sentence rather than showing the class. Prose, tables, and ASCII
     flow diagrams; terminal-session illustrations are fine. Detailed design docs may include code.
   - Full sentences and paragraphs, not bullet-fragment shorthand. Bullets/tables only for
     genuinely enumerable data. Tabular data goes in a table.
   - Describe what a part does and what constrains it — no philosophy, no justification prose, no
     persuasion.
   - Cut any sentence that argues for a decision already stated.
   - No summary/recap/"through-line" sections that restate earlier content.
   - No editorial flourish ("the honest part", "deliberately", "genuine", "the whole value of X").
     State the reasoning plainly instead.
   - Plain, verb-based section headers (Read, Select, Apply) — not "The trust boundary".
   - If something is absent, its absence is the statement — don't explain why it's missing.
   - Keep text naming a concrete failure a mechanism catches, and constraints that change what you
     would build. Those are mechanism, not rationale.
   - Prefer deferring complexity to a separate doc marked *Future* over building it now.

3. **Mandatory self-edit pass, after the draft is complete.** Reread the whole doc sentence by
   sentence, in order. For each sentence, check whether it restates a point already made earlier
   in the same doc — the same fact, decision, or constraint said again in different words, including
   restatements disguised as elaboration, a summary line, or a table repeating what the prose just
   said. Delete every sentence that fails this check. Do not soften this to "keep if it adds a
   little color" — if the point was already made, cut it.

4. **Report** only what changed and why, in one or two sentences. Do not re-print the whole doc in
   chat unless asked.
