# Claude Code Setup

## Agents

Agents are markdown files with a YAML frontmatter header. The frontmatter defines metadata (name, description, model, tools, etc.) and everything after `---` is the system prompt for that agent instance.

Claude auto-invokes agents based on their `description` field — no explicit trigger needed. Agents can also be explicitly instructed to run from slash commands or chat.

Agent file locations:
- `~/.claude/agents/` — global agents (available in all projects)
- `.claude/agents/` — project-local agents (this project)

### Global Agents (`~/.claude/agents/`)

| Agent | Description |
|---|---|
| **docs-engineer** | Audits, updates, and creates project markdown documentation. Enforces a two-tier pattern: concise plain files + detailed `_DETAILS.md` variants. Mandatory creates `PLAN.md`, `ARCHITECTURE.md`, `TODO.md`, `DECISIONS.md` if missing. Runs on `claude-opus`. |

### Project-Local Agents (`.claude/agents/`)

| Agent | Description |
|---|---|
| **docs-engineer** | Project-local override of the global docs-engineer (same role, project-scoped). |
| **project-state-updater** | Appends a dated snapshot to `PROJECT_STATE.md` based on the most recent session file. Invoked by `/remember`. Runs on `claude-haiku`. Never rewrites — append only. |
| **code-reviewer** | Reviews code for correctness, style, architectural conformance, and consistency with project conventions. Triggered by `/review` or phrases like "review this" / "check this code". Reads `ARCHITECTURE.md`, `docs/DESIGN.md`, and `skills/testing.md` before reviewing. Runs on `claude-sonnet`. |

---

## Skills

Skills are markdown files in `skills/`. They can be referenced by CLAUDE.md as rule files, or used directly by Claude during a session.

### Project Skills (`skills/`)

| File | How used | Description |
|---|---|---|
| **session.md** | defined here, used by Claude and `/remember` | Session continuation block format and rules |
| **testing.md** | referenced by CLAUDE.md | Test layout, style rules, and assertion standards |

---

## Slash Commands

Slash commands live in `.claude/commands/` and are invoked via `/command-name`.

| File | Invocation | Description |
|---|---|---|
| **remember.md** | `/remember` | Appends a session continuation block to the current session file, then appends a snapshot to `PROJECT_STATE.md`. Format defined in `skills/session.md`. |

---

## Hooks

Hooks are shell commands that run automatically in response to Claude Code lifecycle events. Configured in `.claude/settings.json`.

### SessionStart

Runs `scripts/session.py startup` at the start of every session:

```json
"hooks": {
  "SessionStart": [
    {
      "hooks": [
        {
          "type": "command",
          "command": "python \"$CLAUDE_PROJECT_DIR/scripts/session.py\" startup",
          "timeout": 10
        }
      ]
    }
  ]
}
```

This prints the current session file path and contents into Claude's context so each session picks up where the last left off.

---

## Session File Management (`scripts/session.py`)

Manages `session_*.md` files in the project root.

| Trigger | Action | Who |
|---|---|---|
| Session start | Create/load session file, print to context | `SessionStart` hook |
| Natural milestones during session | Append session continuation block | Claude, following `skills/session.md` |
| End of session | Append session continuation block + update `PROJECT_STATE.md` | Claude, via `/remember` |

**Rotation rules:**
- Session files older than 20 minutes are rotated to `session_bak/` before a new one is created
- `session_bak/` retains the 5 most recent files; older ones are deleted automatically

**Session file structure:**
```
# IndicationScout — Session

> Started: YYYY-MM-DD HH:MM

## What Was Worked On
## Decisions Made
## Pain Points / Errors Found
## Next Steps Agreed On
```
