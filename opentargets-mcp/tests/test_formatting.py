"""Unit tests for the rendering helpers. No network."""

from opentargets_mcp.constants import MAX_ASSOCIATION_ROWS
from opentargets_mcp.resolve import Entity, Resolution
from opentargets_mcp.tools import _clamp, _matched_line, _number, _shown_line, _truncate


def test_matched_line_names_the_match() -> None:
    resolution = Resolution(match=Entity("ENSG1", "JAK1", "target", None), alternatives=[])
    assert "JAK1" in _matched_line(resolution)
    assert "ENSG1" in _matched_line(resolution)


def test_matched_line_lists_alternatives() -> None:
    resolution = Resolution(
        match=Entity("ENSG1", "JAK1", "target", None),
        alternatives=[Entity("ENSG2", "JAK2", "target", None)],
    )
    assert "Other candidates" in _matched_line(resolution)
    assert "JAK2" in _matched_line(resolution)


def test_shown_line_flags_truncation() -> None:
    assert _shown_line(20, 1660, "associated diseases") == "Showing 20 of 1660 associated diseases."


def test_shown_line_is_plain_when_complete() -> None:
    assert _shown_line(7, 7, "associated diseases") == "7 associated diseases."


def test_clamp_holds_the_range() -> None:
    assert _clamp(0, MAX_ASSOCIATION_ROWS) == 1
    assert _clamp(5000, MAX_ASSOCIATION_ROWS) == MAX_ASSOCIATION_ROWS
    assert _clamp(20, MAX_ASSOCIATION_ROWS) == 20


def test_number_rounds_and_keeps_absence_visible() -> None:
    assert _number(0.8336899836911389) == "0.834"
    assert _number(None) == ""


def test_truncate_marks_the_cut() -> None:
    assert _truncate("abcdef", 100) == "abcdef"
    assert _truncate("abcdef", 3).endswith("[…]")
