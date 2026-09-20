#!/bin/bash
# Surfaces the current week's goals once per session, before the first source-file edit.
#
# Runs on PreToolUse for Write|Edit. Plain stdout is not visible to the model on this event, so the
# goals text is returned as JSON under hookSpecificOutput.additionalContext. No permissionDecision
# is set, which leaves normal permission handling in place — this hook warns and never blocks.
#
# Markdown edits are skipped so that editing a plan, a session file or the goals file itself does
# not consume the session's one firing.
#
# An earlier week's file is never substituted for a missing one. A check against stale goals is
# worse than no check, so the absent week is reported instead.
#
# Config:
#   CLAUDE_GOALS_WEEK  ISO week id to look for (default: current, e.g. 2026-W38)
#   CLAUDE_GOALS_DIR   directory holding the goals files (default: $CLAUDE_PROJECT_DIR/goals)

set -euo pipefail

input=$(cat)

read -r session_id proceed <<<"$(
    python3 -c '
import json, sys

payload = json.load(sys.stdin)
session_id = payload.get("session_id") or ""
path = (payload.get("tool_input") or {}).get("file_path") or ""

# Markdown is documentation, not implementation work.
proceed = 1 if path and not path.lower().endswith(".md") else 0
print(session_id or "-", proceed)
' <<<"$input"
)"

[ "$proceed" -eq 0 ] && exit 0
[ "$session_id" = "-" ] && exit 0

marker="/tmp/claude_goals_scope_${session_id}"
[ -f "$marker" ] && exit 0
touch "$marker"

week="${CLAUDE_GOALS_WEEK:-$(date +%G-W%V)}"
goals_dir="${CLAUDE_GOALS_DIR:-${CLAUDE_PROJECT_DIR:-$(dirname "${BASH_SOURCE[0]}")/../..}/goals}"

# Quoted heredoc: the script below is passed through verbatim, so apostrophes in the prose are safe.
GOALS_FILE="${goals_dir}/GOALS_FOR_${week}.md" GOALS_WEEK="$week" python3 <<'PY'
import json, os

path = os.environ["GOALS_FILE"]
week = os.environ["GOALS_WEEK"]

try:
    with open(path) as fh:
        body = fh.read().strip()
except OSError:
    body = None

if body is None:
    context = (
        f"[goals] No goals file for {week} at {path}.\n\n"
        "Tell the user the week has no goals file and ask whether to write one. Do not infer this "
        "week's goals from an earlier week or from the work in progress. Proceed with the edit "
        "either way."
    )
else:
    context = (
        f"[goals] Current week ({week}):\n\n{body}\n\n"
        "Before continuing, state in one line which of the goals above this work serves, naming its "
        "slug. If it serves none, say so and name what it is instead. If this is not work from a "
        "PLAN_*.md — a bug fix, a one-off edit — say \"not plan work\" and carry on. Make the "
        "statement even when the work is clearly in scope: a silent session means this check never "
        "ran. This is a warning only; the edit proceeds regardless."
    )

print(json.dumps({
    "hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "additionalContext": context,
    }
}))
PY
