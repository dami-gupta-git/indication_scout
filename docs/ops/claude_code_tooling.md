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
| **code-reviewer** | Reviews code for correctness, style, architectural conformance, and consistency with project conventions. Triggered by `/review` or phrases like "review this" / "check this code". Reads `ARCHITECTURE.md`, `for_me/DESIGN.md`, and `skills/testing.md` before reviewing. Runs on `claude-sonnet`. |

---

## Skills

Skills are markdown files in `skills/`. They can be referenced by CLAUDE.md as rule files, or used directly by Claude during a session.

### Project Skills (`skills/`)

| File | How used | Description |
|---|---|---|
| **session.md** | defined here, used by Claude | Session continuation block format and rules |
| **testing.md** | referenced by CLAUDE.md | Test layout, style rules, and assertion standards |

---

## Slash Commands

Slash commands live in `.claude/commands/` and are invoked via `/command-name`. None are currently defined — session writes
and findings promotion happen automatically at context thresholds.

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
| Natural milestones during session | Rewrite the session block | Claude, following `skills/session.md` |
| Context thresholds (20/40/60/75/85%) | Rewrite the session block + promote confirmed findings to `for_me/findings.md` | `UserPromptSubmit` and `Stop` hooks |

**Rotation rules:**
- Two triggers: the active session file exceeding 20 KB, evaluated at session start, and context reaching the 85% threshold,
  evaluated mid-session
- Rotation summarizes the file into `sessions_summary.md`, moves it to `session_archive/`, and creates a replacement
- Nothing in the archive is pruned. `sessions_summary.md` itself rotates by the same rule at 60 KB
- See `design_session_memory.md` for the full design

**Session file structure:**
```
# IndicationScout — Session

> Started: YYYY-MM-DD HH:MM

## What Was Worked On
## Decisions Made
## Pain Points / Errors Found
## Next Steps Agreed On
```
