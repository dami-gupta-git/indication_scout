#!/bin/bash
# Asks for a session entry when context usage crosses 50 or 80 percent of the window.
# Runs on UserPromptSubmit: stdout with exit 0 is injected alongside the user's message.
#
# Detection lives in session_peak.py — peak usage across assistant turns, window inferred from the
# model id. Thresholds are tokens, not bytes: measured bytes-per-token ranged 2.6-9.8, so a byte
# threshold is a different percentage in every session.
#
# A marker file stops a threshold firing twice. Its name carries an era — the number of times
# occupancy has fallen in the transcript, which is the number of times compaction has run. Each
# compaction therefore gives every threshold a fresh marker name and they all fire again. The era is
# recomputed from the transcript on every call, so there is no stored state to advance or strand.
#
# Config:
#   CLAUDE_SESSION_WINDOW  context window in tokens (default: inferred from model id)

set -euo pipefail

THRESHOLDS=(50 80)

input=$(cat)

read -r pct era session_id <<<"$(
    CTX_WINDOW="${CLAUDE_SESSION_WINDOW:-}" \
    python3 "$(dirname "${BASH_SOURCE[0]}")/session_peak.py" <<<"$input"
)"

[ -z "${session_id:-}" ] && exit 0

# Highest crossed threshold only — one entry covers the lower one.
target=0
for t in "${THRESHOLDS[@]}"; do
    [ "${pct:-0}" -ge "$t" ] && target=$t
done
[ "$target" -eq 0 ] && exit 0

marker="/tmp/claude_session_${session_id}_${target}_${era}"
[ -f "$marker" ] && exit 0
touch "$marker"

cat <<EOF
[session-memory] Context is at ${pct}% — append a session entry now.

Append (do not rewrite) a dated entry to the current session file. One line per bullet, omit empty
sections. Format in skills/session.md.

Then: has anything been *settled* since the last entry that is expensive to rediscover and invisible
from the code? Gotchas, decisions whose reasoning leaves no trace in the repo (including decisions
not to build something), and standing rules all qualify. If so, append it to for_me/findings.md.
Most of the time the answer is nothing — that is the normal outcome, not a failure.
EOF
