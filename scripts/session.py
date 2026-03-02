"""
Session file manager for IndicationScout.

Usage:
    python scripts/session.py append "Some note to add"
    python scripts/session.py startup   # prints path + contents of current session file

Rules:
- Session files are named session_{datetime}.md in the project root.
- Appending to a file older than 30 minutes rotates it to session_bak/ first.
- session_bak/ retains the 5 most recent files; older ones are deleted.
"""

import argparse
import logging
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SESSION_BAK = PROJECT_ROOT / "session_bak"
MAX_AGE_MINUTES = 20
MAX_BAK_FILES = 5


def _current_session_file() -> Path | None:
    """Return the most recent session_*.md in the project root, or None."""
    files = sorted(PROJECT_ROOT.glob("session_*.md"))
    return files[-1] if files else None


def _rotate(path: Path) -> None:
    """Move path to session_bak/ and prune to MAX_BAK_FILES."""
    SESSION_BAK.mkdir(exist_ok=True)
    dest = SESSION_BAK / path.name
    shutil.move(str(path), dest)
    logger.info("Rotated %s → %s", path.name, dest)

    bak_files = sorted(SESSION_BAK.glob("session_*.md"))
    while len(bak_files) > MAX_BAK_FILES:
        oldest = bak_files.pop(0)
        oldest.unlink()
        logger.info("Deleted oldest backup: %s", oldest.name)


def _new_session_file() -> Path:
    """Create and return a new session file with current timestamp."""
    name = "session_" + datetime.now().strftime("%Y-%m-%d_%H-%M") + ".md"
    path = PROJECT_ROOT / name
    path.write_text(
        f"# IndicationScout — Session\n\n"
        f"> Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        f"## What Was Worked On\n\n"
        f"## Decisions Made\n\n"
        f"## Pain Points / Errors Found\n\n"
        f"## Next Steps Agreed On\n\n"
    )
    logger.info("Created new session file: %s", path.name)
    return path


def _is_too_old(path: Path) -> bool:
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return datetime.now() - mtime > timedelta(minutes=MAX_AGE_MINUTES)


def get_or_create_session() -> Path:
    """Return the active session file, rotating/creating as needed."""
    current = _current_session_file()
    if current is None:
        return _new_session_file()
    if _is_too_old(current):
        _rotate(current)
        return _new_session_file()
    return current


def cmd_append(text: str) -> None:
    path = get_or_create_session()
    with path.open("a") as f:
        f.write(text.rstrip() + "\n")
    print(path)


def cmd_startup() -> None:
    path = get_or_create_session()
    print(f"Session file: {path}\n")
    print(path.read_text())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Manage IndicationScout session files.")
    sub = parser.add_subparsers(dest="command", required=True)

    app = sub.add_parser("append", help="Append text to the current session file.")
    app.add_argument("text", help="Text to append.")

    sub.add_parser("startup", help="Print path to the current (or new) session file.")

    args = parser.parse_args()

    if args.command == "append":
        cmd_append(args.text)
    elif args.command == "startup":
        cmd_startup()


if __name__ == "__main__":
    sys.exit(main())