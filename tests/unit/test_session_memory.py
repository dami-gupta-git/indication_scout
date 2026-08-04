"""Unit tests for the session-memory hook and session file manager.

The era logic lives in `.claude/hooks/session_peak.py`, so it is exercised by running the hook (which
calls it) as a subprocess against a synthetic transcript. No network, no external deps.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HOOK = PROJECT_ROOT / ".claude" / "hooks" / "session-threshold.sh"
SESSION_PY = PROJECT_ROOT / "scripts" / "session.py"

WINDOW = 200_000


def _write_transcript(path: Path, totals: list[int]) -> Path:
    """Write a transcript whose assistant turns carry the given occupancy totals."""
    path.write_text(
        "".join(
            json.dumps(
                {
                    "message": {
                        "model": "claude-opus-5",
                        "usage": {
                            "input_tokens": total,
                            "cache_read_input_tokens": 0,
                            "cache_creation_input_tokens": 0,
                            "output_tokens": 0,
                        },
                    }
                }
            )
            + "\n"
            for total in totals
        )
    )
    return path


def _run_hook(transcript: Path, session_id: str, tmp_path: Path) -> tuple[str, list[str]]:
    """Run the hook once; return its stdout and the marker suffixes it left behind."""
    payload = json.dumps({"transcript_path": str(transcript), "session_id": session_id})
    result = subprocess.run(
        ["bash", str(HOOK)],
        input=payload,
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "CLAUDE_PROJECT_DIR": str(PROJECT_ROOT),
            "TMPDIR": str(tmp_path),
            "CLAUDE_SESSION_WINDOW": str(WINDOW),
        },
    )
    assert result.returncode == 0, result.stderr
    markers = sorted(
        p.name.replace(f"claude_session_{session_id}_", "") for p in Path("/tmp").glob(f"claude_session_{session_id}_*")
    )
    return result.stdout, markers


@pytest.fixture
def session_id(request) -> str:
    """A per-test session id, with its markers removed before and after.

    Only alphanumerics survive: a parametrized node name carries `[` and `]`, which Path.glob reads as
    a character class, so the marker lookup would silently never match.
    """
    sid = "pytest-" + "".join(c if c.isalnum() else "-" for c in request.node.name)

    def _clear() -> None:
        for path in Path("/tmp").glob(f"claude_session_{sid}_*"):
            path.unlink()

    _clear()
    yield sid
    _clear()


@pytest.mark.parametrize(
    "name,totals,expected_markers",
    [
        # Below the lowest threshold: nothing fires.
        ("below_threshold", [40_000], []),
        # 60% of the window crosses 50 in era 0.
        ("crosses_fifty", [60_000, 120_000], ["50_0"]),
        # A jump past 50 straight to 85 fires only the higher mark — one entry covers both.
        ("jump_to_eighty", [10_000, 170_000], ["80_0"]),
        # A single small turn between two large ones is a sub-agent, not a compaction: era stays 0.
        ("small_turn_is_noise", [170_000, 500, 170_500], ["80_0"]),
        # Compaction, then a climb back through 60%: the era advances so 50 fires a second time.
        ("compaction_refires", [60_000, 120_000, 170_000, 40_000, 42_000, 120_000], ["50_1"]),
    ],
)
def test_hook_marker_for_occupancy(name, totals, expected_markers, tmp_path, session_id):
    transcript = _write_transcript(tmp_path / f"{name}.jsonl", totals)
    stdout, markers = _run_hook(transcript, session_id, tmp_path)

    assert markers == expected_markers
    if expected_markers:
        assert "[session-memory]" in stdout
        assert "append a session entry" in stdout
    else:
        assert stdout == ""


def test_hook_fires_once_per_threshold(tmp_path, session_id):
    """A second call at the same occupancy is silent: the marker is what stops the repeat."""
    transcript = _write_transcript(tmp_path / "repeat.jsonl", [60_000, 120_000])

    first_stdout, first_markers = _run_hook(transcript, session_id, tmp_path)
    second_stdout, second_markers = _run_hook(transcript, session_id, tmp_path)

    assert first_markers == ["50_0"]
    assert "[session-memory]" in first_stdout
    assert second_markers == ["50_0"]
    assert second_stdout == ""


def test_hook_survives_missing_transcript(tmp_path, session_id):
    """An unreadable transcript yields no percentage and no marker rather than an error."""
    stdout, markers = _run_hook(tmp_path / "does-not-exist.jsonl", session_id, tmp_path)

    assert stdout == ""
    assert markers == []


def _run_session_py(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SESSION_PY), *args], capture_output=True, text=True, cwd=str(cwd)
    )


def test_rotate_check_reports_size(tmp_path, monkeypatch):
    """rotate-check is a size rule: 'ok' under the limit, 'due' at or over it."""
    monkeypatch.setattr("pathlib.Path.cwd", lambda: tmp_path)
    session_file = tmp_path / "session_2026-01-01_00-00.md"
    session_file.write_text("# S\n\n")

    import importlib.util

    spec = importlib.util.spec_from_file_location("session_mod", SESSION_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)

    assert module.rotation_due() is False

    session_file.write_text("x" * module.MAX_SIZE_BYTES)
    assert module.rotation_due() is True


def test_rotate_archives_and_replaces(tmp_path, monkeypatch):
    """rotate appends the summary, archives the original, and leaves exactly one new session file."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("session_mod2", SESSION_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "SESSION_ARCHIVE", tmp_path / "session_archive")
    monkeypatch.setattr(module, "SUMMARY_FILE", tmp_path / "sessions_summary.md")
    monkeypatch.setattr(module, "LEGACY_BAK", tmp_path / "session_bak")

    original = tmp_path / "session_2026-01-01_00-00.md"
    original.write_text("# S\n\nwork happened\n")
    summary_source = tmp_path / "summary.txt"
    summary_source.write_text("Did the thing. Decided the other thing.")

    module.cmd_rotate(summary_source)

    assert not original.exists()
    assert (tmp_path / "session_archive" / "session_2026-01-01_00-00.md").read_text() == "# S\n\nwork happened\n"

    summary = (tmp_path / "sessions_summary.md").read_text()
    assert "session_2026-01-01_00-00.md" in summary
    assert "Did the thing. Decided the other thing." in summary

    replacements = list(tmp_path.glob("session_*.md"))
    assert len(replacements) == 1
    assert replacements[0].read_text().startswith("# IndicationScout — Session")


def test_rotate_refuses_empty_summary(tmp_path, monkeypatch):
    """An empty summary aborts the rotation: the summary is the only compressed record of the file."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("session_mod3", SESSION_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)

    original = tmp_path / "session_2026-01-01_00-00.md"
    original.write_text("# S\n")
    empty = tmp_path / "empty.txt"
    empty.write_text("   \n")

    with pytest.raises(SystemExit) as exc:
        module.cmd_rotate(empty)

    assert exc.value.code == 1
    assert original.exists()
