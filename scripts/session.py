"""
Session file manager for IndicationScout.

Usage:
    python scripts/session.py startup                      # rotation check, then print session file + summary
    python scripts/session.py rotate-check                  # prints "due" or "ok"
    python scripts/session.py rotate --summary-file <path>  # append summary, archive, create replacement

Rules:
- Session files are named session_{datetime}.md in the project root, one per rotation (not per session).
- Rotation is a size rule, evaluated at session start: past MAX_SIZE_BYTES the file is summarized into
  sessions_summary.md, moved to session_archive/, and replaced.
- Summarizing needs a model, so it happens outside this script: `startup` reports that rotation is due and the
  model calls `rotate` with the summary it wrote.
- Nothing in the archive is pruned. sessions_summary.md rotates by the same rule at SUMMARY_MAX_SIZE_BYTES.

See design_session_memory.md.
"""

import argparse
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SESSION_ARCHIVE = PROJECT_ROOT / "session_archive"
SUMMARY_FILE = PROJECT_ROOT / "sessions_summary.md"
LEGACY_BAK = PROJECT_ROOT / "session_bak"

MAX_SIZE_BYTES = 20 * 1024
SUMMARY_MAX_SIZE_BYTES = 60 * 1024


def _current_session_file() -> Path | None:
    """Return the most recent session_*.md in the project root, or None."""
    files = sorted(PROJECT_ROOT.glob("session_*.md"))
    return files[-1] if files else None


def _new_session_file() -> Path:
    """Create and return a new session file with current timestamp."""
    now = datetime.now()
    path = PROJECT_ROOT / ("session_" + now.strftime("%Y-%m-%d_%H-%M") + ".md")
    path.write_text(f"# IndicationScout — Session\n\n> Started: {now.strftime('%Y-%m-%d %H:%M')}\n\n")
    logger.info("Created new session file: %s", path.name)
    return path


def _archive(path: Path, name: str | None = None) -> Path:
    """Move path into session_archive/ under `name`. No pruning."""
    SESSION_ARCHIVE.mkdir(exist_ok=True)
    dest = SESSION_ARCHIVE / (name or path.name)
    shutil.move(str(path), dest)
    logger.info("Archived %s → %s", path.name, dest.name)
    return dest


def _is_too_large(path: Path, limit: int) -> bool:
    return path.exists() and path.stat().st_size >= limit


def rotation_due() -> bool:
    current = _current_session_file()
    return current is not None and _is_too_large(current, MAX_SIZE_BYTES)


def _migrate_legacy_bak() -> None:
    """One-time move of session_bak/ contents into session_archive/. The prune is gone."""
    if not LEGACY_BAK.is_dir():
        return
    for path in sorted(LEGACY_BAK.glob("session_*.md")):
        SESSION_ARCHIVE.mkdir(exist_ok=True)
        dest = SESSION_ARCHIVE / path.name
        if dest.exists():
            continue
        shutil.move(str(path), dest)
        logger.info("Migrated %s → session_archive/", path.name)


def _rotate_summary_if_needed() -> None:
    """sessions_summary.md rotates by the same rule one level up; summarizing happens before this call."""
    if not _is_too_large(SUMMARY_FILE, SUMMARY_MAX_SIZE_BYTES):
        return
    _archive(SUMMARY_FILE, name=f"sessions_summary_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.md")


def cmd_rotate(summary_file: Path) -> None:
    """Append the model's summary, archive the file, create the replacement."""
    current = _current_session_file()
    if current is None:
        logger.error("No session file to rotate")
        sys.exit(1)

    summary = summary_file.read_text().strip()
    if not summary:
        logger.error("Empty summary; refusing to rotate (the summary is the only compressed record)")
        sys.exit(1)

    # Archive before appending: a failed move must not leave a summary behind, or the retry appends it twice.
    name = current.name
    _archive(current)

    with SUMMARY_FILE.open("a") as fh:
        fh.write(f"\n---\n\n## {name} (archived {datetime.now().strftime('%Y-%m-%d %H:%M')})\n\n")
        fh.write(summary + "\n")

    fresh = _new_session_file()
    _rotate_summary_if_needed()
    print(fresh)


def cmd_startup() -> None:
    """Rotation is evaluated before the read; when due, print the instruction and no file content."""
    _migrate_legacy_bak()

    if rotation_due():
        current = _current_session_file()
        print(
            "ROTATION DUE — before anything else:\n"
            f"1. Summarize {current.name} in at most 15 lines: date range, what changed, decisions with "
            "reasoning, unresolved problems. Summarize that file alone; do not re-read anything else.\n"
            "2. Write the summary to a file, then run:\n"
            "   python scripts/session.py rotate --summary-file <path>\n"
            "3. Read the new session file and sessions_summary.md."
        )
        return

    current = _current_session_file() or _new_session_file()
    print(f"Session file: {current}\n")
    print(current.read_text())
    if SUMMARY_FILE.exists():
        print(f"\n--- {SUMMARY_FILE.name} ---\n")
        print(SUMMARY_FILE.read_text())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Manage IndicationScout session files.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("startup", help="Rotation check, then print the session file and summary.")
    sub.add_parser("rotate-check", help="Print 'due' or 'ok'.")

    rot = sub.add_parser("rotate", help="Archive the current file and create its replacement.")
    rot.add_argument("--summary-file", required=True, type=Path, help="File holding the model-written summary.")

    args = parser.parse_args()

    if args.command == "startup":
        cmd_startup()
    elif args.command == "rotate-check":
        print("due" if rotation_due() else "ok")
    elif args.command == "rotate":
        cmd_rotate(args.summary_file)


if __name__ == "__main__":
    sys.exit(main())
