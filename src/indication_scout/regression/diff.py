"""Structured diff between a golden and current SupervisorOutput.

Severity levels:
- error: must-fix (missing required field, severe set divergence, score out of
  range, empty summary, etc.). A regression test fails if any error-severity
  Diff is produced.
- warn: visible but non-blocking drift (small numeric drift within tolerance,
  prose length swing). Rendered by the diff CLI but does not fail tests.
- info: notable but expected differences (e.g. blurb only present on top-5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Severity = Literal["error", "warn", "info"]


@dataclass(frozen=True)
class Diff:
    """One observed difference between golden and current at a JSON-pointer-ish path."""

    path: str
    kind: str
    severity: Severity
    detail: str
    golden: object | None = None
    current: object | None = None


def jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity. Returns 1.0 for two empty sets (degenerate match)."""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)
