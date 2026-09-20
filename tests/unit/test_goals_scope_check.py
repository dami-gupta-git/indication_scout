"""Unit tests for the weekly-goals PreToolUse hook (`.claude/hooks/goals-scope-check.sh`).

The hook is driven as a subprocess with synthetic event JSON on stdin. `CLAUDE_GOALS_DIR` and
`CLAUDE_GOALS_WEEK` pin it to a temporary goals directory so neither the repo's real goals file nor
the real calendar week affects the result.
"""

import json
import os
import subprocess
import uuid
from pathlib import Path

import pytest

HOOK = (
    Path(__file__).resolve().parents[2] / ".claude" / "hooks" / "goals-scope-check.sh"
)

WEEK = "2026-W38"
GOALS_BODY = (
    "# Goals for 2026-W38\n\n## Goals\n\n1. `ct-cache` - Finish the per-trial cache.\n"
)
STALE_BODY = "# Goals for 2026-W37\n\n## Goals\n\n1. `old-goal` - Last week's work.\n"


@pytest.fixture
def goals_dir(tmp_path: Path) -> Path:
    """A goals directory holding both the current week's file and an earlier week's file."""
    (tmp_path / f"GOALS_FOR_{WEEK}.md").write_text(GOALS_BODY)
    (tmp_path / "GOALS_FOR_2026-W37.md").write_text(STALE_BODY)
    return tmp_path


def run_hook(goals_dir: Path, file_path: str | None, session_id: str) -> str:
    """Run the hook against one synthetic Write event and return its stdout."""
    payload: dict[str, object] = {
        "session_id": session_id,
        "hook_event_name": "PreToolUse",
        "tool_name": "Write",
    }
    if file_path is not None:
        payload["tool_input"] = {"file_path": file_path}

    result = subprocess.run(
        [str(HOOK)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "CLAUDE_GOALS_DIR": str(goals_dir),
            "CLAUDE_GOALS_WEEK": WEEK,
        },
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def new_session() -> str:
    """A session id unused by any earlier run, so the once-per-session marker starts clean."""
    return f"test-{uuid.uuid4()}"


def test_source_edit_injects_goals(goals_dir: Path) -> None:
    """A source-file edit returns the goals text as additionalContext and does not block the edit."""
    payload = json.loads(
        run_hook(goals_dir, "src/indication_scout/services/llm.py", new_session())
    )

    assert set(payload) == {"hookSpecificOutput"}
    hook_output = payload["hookSpecificOutput"]
    assert set(hook_output) == {"hookEventName", "additionalContext"}
    assert hook_output["hookEventName"] == "PreToolUse"

    context = hook_output["additionalContext"]
    assert "`ct-cache` - Finish the per-trial cache." in context
    assert "[goals] Current week (2026-W38)" in context
    assert "not plan work" in context
    assert "old-goal" not in context


def test_fires_once_per_session(goals_dir: Path) -> None:
    """The second source edit of the same session is silent."""
    session_id = new_session()

    first = run_hook(goals_dir, "src/indication_scout/services/llm.py", session_id)
    second = run_hook(
        goals_dir, "src/indication_scout/services/retrieval.py", session_id
    )

    assert json.loads(first)["hookSpecificOutput"]["hookEventName"] == "PreToolUse"
    assert second == ""


@pytest.mark.parametrize(
    "file_path",
    [
        "PLAN_goals_scope_guard.md",
        "goals/GOALS_FOR_2026-W38.md",
        "for_me/findings.md",
        "docs/OVERVIEW.MD",
    ],
)
def test_markdown_edits_are_skipped(goals_dir: Path, file_path: str) -> None:
    """Markdown is documentation, not implementation, so it must not consume the session's firing."""
    assert run_hook(goals_dir, file_path, new_session()) == ""


def test_missing_file_path_is_skipped(goals_dir: Path) -> None:
    """An event carrying no file path produces no output."""
    assert run_hook(goals_dir, None, new_session()) == ""


def test_missing_week_is_reported_without_stale_content(tmp_path: Path) -> None:
    """With no file for the current week, the absent week is named and no earlier week is substituted."""
    (tmp_path / "GOALS_FOR_2026-W37.md").write_text(STALE_BODY)

    output = run_hook(tmp_path, "src/indication_scout/services/llm.py", new_session())

    context = json.loads(output)["hookSpecificOutput"]["additionalContext"]
    assert f"[goals] No goals file for {WEEK}" in context
    assert "ask whether to write one" in context
    assert "old-goal" not in context
    assert "2026-W37" not in context
