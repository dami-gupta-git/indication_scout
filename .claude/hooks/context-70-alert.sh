#!/bin/bash
# Plays a sound when Claude Code context usage hits ~80%.
#
# Heuristic: 200k-token context window produces ~1.6 MB of JSONL transcript
# at capacity. 80% of that is ~1.28 MB.
#
# Note: transcript file grows monotonically (compacted messages remain),
# so this may over-estimate after compaction.

set -euo pipefail

input=$(cat)
transcript=$(echo "$input" | /usr/bin/python3 -c "import json,sys; print(json.load(sys.stdin)['transcript_path'])")
session_id=$(echo "$input" | /usr/bin/python3 -c "import json,sys; print(json.load(sys.stdin)['session_id'])")

marker="/tmp/claude_ctx70_alert_${session_id}"

# Already alerted this session — skip
[ -f "$marker" ] && exit 0

# Transcript file size in bytes (macOS stat)
size=$(stat -f%z "$transcript" 2>/dev/null || echo 0)

# 80% of ~1.6 MB ≈ 1.28 MB
BYTE_THRESHOLD=${CLAUDE_CTX70_ALERT_BYTES:-896000}

if [ "$size" -gt "$BYTE_THRESHOLD" ]; then
    afplay /System/Library/Sounds/Sosumi.aiff &
    touch "$marker"
fi