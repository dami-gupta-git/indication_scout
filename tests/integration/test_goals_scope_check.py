"""Integration tests for the weekly-goals hook wiring.

Unlike the unit tests, these use no environment overrides: the hook resolves the goals directory and
the ISO week itself, and the registration in `.claude/settings.json` is checked against the file on
disk. No external API or network call is involved — the hook has no such dependency. What is
exercised here is the real wiring: settings registration, path resolution, executable bit, the
system date, and the repo's own goals directory.
"""

import json
import os
import subprocess
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HOOK = REPO_ROOT / ".claude" / "hooks" / "goals-scope-check.sh"
SETTINGS = REPO_ROOT / ".claude" / "settings.json"
GOALS_DIR = REPO_ROOT / "goals"


def current_week() -> str:
    """The ISO week id the hook will resolve, from the same system date the hook uses."""
    return subprocess.run(
        ["date", "+%G-W%V"], capture_output=True, text=True, check=True
    ).stdout.strip()


def test_hook_is_registered_in_settings() -> None:
    """`.claude/settings.json` registers the hook for Write and Edit, and the script is executable."""
    settings = json.loads(SETTINGS.read_text())

    pre_tool_use = settings["hooks"]["PreToolUse"]
    matching = [entry for entry in pre_tool_use if entry.get("matcher") == "Write|Edit"]
    assert len(matching) == 1

    commands = [hook["command"] for hook in matching[0]["hooks"]]
    assert commands == ['"$CLAUDE_PROJECT_DIR"/.claude/hooks/goals-scope-check.sh']

    assert HOOK.is_file()
    assert os.access(HOOK, os.X_OK)


def test_claude_md_carries_the_rule() -> None:
    """The standing instruction is the half of the feature the hook cannot enforce."""
    claude_md = (REPO_ROOT / "CLAUDE.md").read_text()

    assert "## Weekly Goals" in claude_md
    assert "goals/GOALS_FOR_<ISO week>.md" in claude_md
    assert "Before implementing any `PLAN_*.md`" in claude_md


def test_hook_resolves_the_real_week_unaided() -> None:
    """With no overrides the hook finds the repo goals directory and this week's file, or names it missing."""
    week = current_week()
    payload = {
        "session_id": f"integration-{uuid.uuid4()}",
        "hook_event_name": "PreToolUse",
        "tool_name": "Edit",
        "tool_input": {
            "file_path": str(REPO_ROOT / "src" / "indication_scout" / "config.py")
        },
    }

    result = subprocess.run(
        [str(HOOK)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env={**os.environ, "CLAUDE_PROJECT_DIR": str(REPO_ROOT)},
    )
    assert result.returncode == 0, result.stderr

    hook_output = json.loads(result.stdout)["hookSpecificOutput"]
    assert hook_output["hookEventName"] == "PreToolUse"
    assert "permissionDecision" not in hook_output
    context = hook_output["additionalContext"]

    goals_file = GOALS_DIR / f"GOALS_FOR_{week}.md"
    if goals_file.is_file():
        assert f"[goals] Current week ({week})" in context
        assert goals_file.read_text().strip() in context
    else:
        assert f"[goals] No goals file for {week}" in context
        assert str(goals_file) in context


def test_goals_directory_holds_only_week_files() -> None:
    """Every file in the goals directory follows the naming the hook resolves."""
    names = sorted(path.name for path in GOALS_DIR.iterdir() if path.is_file())
    assert names, "goals directory is empty"

    for name in names:
        assert name.startswith("GOALS_FOR_") and name.endswith(".md"), name
