# Session Memory — Design

## Overview

A session's working context is lost when compaction runs, and today nothing is written unless the user asks. One hook watches
token occupancy and, at two thresholds, asks the model to append an entry to the session file and promote anything settled into
`for_me/findings.md`. Entries are appended, never rewritten, so the hook needs no way to find or compare an earlier one. Marker
files stop a threshold firing twice; their names carry a count of how many times occupancy has fallen, so each compaction gives
every threshold a fresh name and they all fire again. At 20 KB the session file is summarized into `sessions_summary.md` and
archived.

> Replaces: a `SessionStart` hook printing the session file in full, rotation to `session_bak/` with a five-file prune, and
> `/remember`, which is removed along with `PROJECT_STATE.md` and the agent that wrote it.

## Trigger

- **50 and 80 percent** of the window, measured in tokens. Not higher: auto-compaction fires at an environment-configurable
  point below the window, so a mark close to it may never fire. Not bytes: bytes per token ranged 2.6–9.8 across forty-six
  sessions.
- Occupancy per turn sums fresh input, cache read, cache creation, and output; the largest is the session peak. Cache reads
  count toward the window like fresh input.
- Window from the model id, 1M or 200K, overridable by environment variable.
- `.claude/hooks/context-70-alert.sh` is the reference; its peak scan and window inference transfer unchanged into
  `session_peak.py`.

## Mechanism

- **One hook, on `UserPromptSubmit`**: plain stdout, exit 0, injected alongside the user's message. Fires once per message, so
  a threshold crossed mid-tool-run is acted on at the next one.
- **The hook is two files.** `session-threshold.sh` holds the threshold comparison, the marker check, and the instruction text;
  `session_peak.py` reads the payload and returns occupancy percentage, era, and session id. The scan is a separate file rather
  than an inline `python3 -c` string: the shell would have to single-quote it, where one apostrophe anywhere — including in a
  comment — closes the string and breaks the hook with a syntax error reported at the last line of the script.
- **A malformed payload exits 0 without output.** The hook runs on every message, and a non-zero exit under `set -euo pipefail`
  surfaces to the user each time; the caller treats an empty session id as nothing to do.
- **A marker file per threshold** stops it firing twice. Named `{session}_{threshold}_{era}` and kept in `/tmp`, so markers
  vanish on restart.
- The **era** is the number of times occupancy has fallen across the transcript, which is the number of times compaction has
  run — occupancy only decreases when the window is compacted. Each compaction therefore gives every threshold a fresh marker
  name and they all fire again.
- The era is recomputed from the transcript on every call, so there is no stored counter to advance, miss, or leave stranded.
- It must count falls rather than bucket the current peak. A bucket is a function of where occupancy sits, so climbing back to
  a previously-seen level regenerates a marker name already written and the compaction goes unnoticed.
- A fall counts when occupancy sits more than a tenth below the running peak for **two consecutive turns**. Not every usage
  record is a main-loop turn — sub-agent and tool turns carry their own, far smaller, usage — so a single small record between
  two large ones is noise. Compaction persists, so every turn after it is small.
- The comparison is against the running peak rather than the previous turn, so a decline spread over several turns, each too
  small to trip the floor alone, is still caught.
- When a fall is credited the peak resets, so the reported percentage reflects occupancy in the new era rather than the old
  high-water mark. A session that compacts from 85 percent down to 30 is then correctly silent until it climbs past 50 again.
- `PreCompact` is unusable — its documented context injection is contradicted by the shipped implementation, and it is absent
  from the table of events accepting `additionalContext`. `PreToolUse` returns permission decisions only.

## Write

- **Appended, never rewritten.** With nothing to supersede there is no entry identity, no content comparison, and no record of
  what was injected when.
- Each entry: date, short title, what was worked on, decisions with reasoning, problems found, next steps. Format in
  `skills/session.md`.
- The same instruction asks whether anything has been **settled** since the last entry; if so it goes to
  `for_me/findings.md`.
- **The filter is expensive to rediscover and invisible from the code** — gotchas, decisions whose reasoning leaves no trace
  (including decisions not to build something), standing rules. Strangeness alone is too narrow: it drops the negative results
  and the decisions, which are hardest to recover.
- Settled, not discovered. A hypothesis that survived one test is not a finding. Promoting nothing is normal.

## Files

| Path | Role |
| --- | --- |
| `session_{datetime}.md` (root) | Active file. One per rotation, not per session. |
| `sessions_summary.md` (root) | Running summary, append-only. |
| `session_archive/` | Rotated files. Replaces `session_bak/`; no prune. |
| `/tmp` | Threshold markers. Transient. |

## Rotate

Past 20 KB, in order: moved to `session_archive/`, its summary appended to `sessions_summary.md`, replaced with a new empty
file. The move precedes the append so a failed move leaves no summary behind for a retry to write twice.

- **Evaluated at session start only**, before the read.
- Summarization needs a model, so it spans two steps: the hook reports rotation due and emits an instruction; the model writes
  the summary and calls the script for the file operations. The script never summarizes.
- **When rotation is due the hook prints no file content** — printing the oversized file would incur the cost the size limit
  exists to prevent, and the replacement does not yet exist.
- **At most 15 lines** per rotation: date range, what changed, decisions with reasoning, unresolved problems.
- Nothing is pruned. The current five-file prune is what makes rotation lose material.
- `sessions_summary.md` rotates the same way at 60 KB — summarized in place, then archived under a rotation-datetime name.
  Summarizing first matters because the summary derives from the file being displaced.

## Read

Startup reads the session file and `sessions_summary.md` in full. Both are bounded by their rotation rules, so cold-start cost
has a ceiling rather than growing with project history.

## What changes

- **`scripts/session.py`** keeps `startup`, gains the summary read, rotation-before-read, archival without pruning,
  `rotate-check`, and `rotate` (taking the model's summary). `append` is dropped.
- **One new hook** on `UserPromptSubmit`, as `session-threshold.sh` plus `session_peak.py`. `SessionStart` keeps its command and
  gains the rotation check; it is registered in `.claude/settings.json`.
- **Every entry point invokes `python3`.** The system binary is 3.9 on macOS and cannot parse the 3.10+ annotations in
  `scripts/session.py` and `session_peak.py`; a hook's error fallback turns that failure into a plausible default, so nothing is
  ever written.
- **Markers live in `/tmp`**, not the platform temp directory — on macOS `tempfile.gettempdir()` is `$TMPDIR` while shell hooks
  name `/tmp`, so the two would disagree and neither would error.
- `skills/session.md` documents the append format and the promotion rule. `.claude/commands/remember.md` and
  `.claude/agents/project-state-updater.md` are deleted, with their rows removed from `docs/ops/claude_code_tooling.md`.

## Relationship to other records

| Store | Written by | Read at startup | Deletion |
| --- | --- | --- | --- |
| Conversation transcripts | Automatic | Never | — |
| Session files | Automatic, at each threshold | Yes | Rotated to archive |
| `sessions_summary.md` | Model, at rotation | Yes | Rotated to archive |
| `for_me/findings.md` | Automatic, promoted at each threshold | Per project instructions | Never |

Confirmed conclusions belong in `for_me/findings.md`; session files carry in-progress state and narrative. `PROJECT_STATE.md` is
retired and kept as a historical record.

## Open items

- Archive compression is deferred; files are stored uncompressed.
- The era's fall threshold — a tenth below the peak, sustained two turns — was chosen to sit clear of ordinary jitter, not
  fitted to observed compactions: none of the project transcripts had ever compacted, so there was no data to fit. It scores
  zero falls across six real transcripts, including one of 403 turns. Revisit once real compactions are on disk.
