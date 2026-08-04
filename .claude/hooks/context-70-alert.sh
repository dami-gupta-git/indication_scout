#!/bin/bash
# Plays a sound when Claude Code context usage crosses a percentage of the window.
#
# Reads real token counts from the transcript's `usage` blocks rather than estimating
# from file size. Measured bytes-per-token across real sessions ranged 2.6-9.8, so any
# byte threshold means a different percentage in every session.
#
# Config:
#   CLAUDE_CTX70_ALERT_PCT     percentage to fire at (default 70)
#   CLAUDE_CTX70_ALERT_WINDOW  context window in tokens (default: inferred from model id)

set -euo pipefail

input=$(cat)

read -r pct session_id <<<"$(
    CTX_PCT="${CLAUDE_CTX70_ALERT_PCT:-70}" \
    CTX_WINDOW="${CLAUDE_CTX70_ALERT_WINDOW:-}" \
    /usr/bin/python3 -c '
import json, os, sys

payload = json.load(sys.stdin)
transcript = payload.get("transcript_path", "")
session_id = payload.get("session_id", "")

# Context occupancy = the largest assistant-turn usage total. Cache reads count
# toward the window exactly like fresh input, so all four fields are summed.
peak, model = 0, ""
try:
    with open(transcript) as fh:
        for line in fh:
            try:
                msg = json.loads(line).get("message") or {}
            except ValueError:
                continue
            usage = msg.get("usage") or {}
            if not usage:
                continue
            total = (usage.get("input_tokens", 0)
                     + usage.get("cache_read_input_tokens", 0)
                     + usage.get("cache_creation_input_tokens", 0)
                     + usage.get("output_tokens", 0))
            if total > peak:
                peak, model = total, msg.get("model", "")
except OSError:
    print("0", session_id)
    sys.exit(0)

window = os.environ.get("CTX_WINDOW", "").strip()
window = int(window) if window else (1_000_000 if "[1m]" in model else 200_000)

threshold = int(os.environ.get("CTX_PCT", "70"))
current = int(peak * 100 / window) if window else 0
print(current if current >= threshold else 0, session_id)
' <<<"$input"
)"

# Already alerted this session — skip
marker="/tmp/claude_ctx70_alert_${session_id}"
[ -f "$marker" ] && exit 0

if [ "${pct:-0}" -gt 0 ]; then
    afplay /System/Library/Sounds/Sosumi.aiff &
    touch "$marker"
fi
