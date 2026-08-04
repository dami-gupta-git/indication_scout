#!/usr/bin/env python3
"""
Reads a Claude Code hook payload on stdin, prints "<pct> <era> <session_id>" on stdout.

Called by session-threshold.sh. Lives in its own file rather than inline in the shell script: the
caller would have to embed it in a single-quoted string, where one apostrophe anywhere — including
in a comment — closes the string and breaks the hook with a syntax error reported at the last line
of the script.

Config:
    CTX_WINDOW  context window in tokens (default: 1M — the transcript carries no signal for the
                actual window, so this is a guess, not an inference; see the comment in main())
"""

import json
import os
import sys

# Default window when CTX_WINDOW is unset. The transcript carries no signal for 200K vs 1M, so this
# is a guess, not an inference — see the CTX_WINDOW comment below for why it defaults high.
WINDOW_1M = 1_000_000

# A fall is credited when occupancy sits below peak * FALL_FLOOR for FALL_TURNS consecutive turns.
FALL_FLOOR = 0.9
FALL_TURNS = 2


def _usage_total(usage: dict) -> int:
    """Context occupancy for one turn. Cache reads count toward the window exactly like fresh input."""
    return (
        usage.get("input_tokens", 0)
        + usage.get("cache_read_input_tokens", 0)
        + usage.get("cache_creation_input_tokens", 0)
        + usage.get("output_tokens", 0)
    )


def scan(transcript: str) -> tuple[int, int]:
    """Return (peak, era) for the transcript.

    `era` counts how many times occupancy has FALLEN, which is how many times compaction has run:
    occupancy only decreases when the window is compacted. Marker names carry it, so each compaction
    gives every threshold a fresh name and they all fire again. It is derived from the transcript on
    every call, so there is no stored counter to advance, miss, or leave stranded.

    The era must count falls rather than bucket the current peak: a bucket is a function of where
    occupancy sits, so climbing back to a previously-seen level regenerates a name already marked and
    the compaction goes unnoticed.
    """
    peak, era, low = 0, 0, 0
    with open(transcript) as fh:
        for line in fh:
            try:
                msg = json.loads(line).get("message") or {}
            except ValueError:
                continue
            usage = msg.get("usage") or {}
            if not usage:
                continue
            total = _usage_total(usage)
            # A fall is only credited when occupancy STAYS down: `low` counts consecutive turns under
            # the floor, and FALL_TURNS in a row are required. Not every usage record is a main-loop
            # turn — sub-agent and tool turns carry their own, far smaller, usage — so a single small
            # record between two large ones is noise, not a compaction. Compaction persists: every
            # turn after it is small.
            #
            # Compared against the running PEAK rather than the previous turn, so a decline spread
            # over several turns (each too small to trip the floor on its own) is still caught.
            #
            # The floor clears ordinary jitter. Chosen to sit clear of noise rather than fitted to
            # observed compactions — none of the project transcripts has ever compacted, so there was
            # no data to fit.
            if peak and total < peak * FALL_FLOOR:
                low += 1
                if low >= FALL_TURNS:
                    era += 1
                    peak, low = 0, 0
            else:
                low = 0
            if total > peak:
                peak = total
    return peak, era


def main() -> None:
    # A malformed payload must not fail the hook: it runs on every UserPromptSubmit, and a non-zero
    # exit under `set -euo pipefail` is user-visible on every message. Print nothing; the empty
    # session_id guard in the caller then exits cleanly.
    try:
        payload = json.load(sys.stdin)
    except ValueError:
        sys.exit(0)

    transcript = payload.get("transcript_path", "")
    session_id = payload.get("session_id", "")

    try:
        peak, era = scan(transcript)
    except OSError:
        print("0", "0", session_id)
        sys.exit(0)

    # The transcript never records a "[1m]"-style suffix on `model` — only the bare model id
    # (e.g. "claude-opus-5") regardless of the session's actual window — so window size cannot be
    # inferred from the transcript at all. Default to the larger window: understating occupancy
    # fires the threshold late, overstating it fires up to 5x early, so this is the safer guess.
    # CTX_WINDOW overrides for a session actually running the smaller 200K window.
    raw = os.environ.get("CTX_WINDOW", "").strip()
    try:
        window = int(raw) if raw else WINDOW_1M
    except ValueError:
        window = WINDOW_1M

    print(int(peak * 100 / window) if window > 0 else 0, era, session_id)


if __name__ == "__main__":
    main()