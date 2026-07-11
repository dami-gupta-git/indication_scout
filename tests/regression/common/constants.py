"""Test-only constants for the regression suite (paths + cassette mode env).

Numeric thresholds for the comparison harness live in
`indication_scout.regression.constants` so they're importable from both the
test suite and the `scout diff-report` CLI.
"""

from pathlib import Path

REGRESSION_DIR = Path(__file__).parent.parent  # tests/regression (this file is in common/)
GOLD_STANDARD_DIR = REGRESSION_DIR / "gold_standard"
CASSETTE_DIR = REGRESSION_DIR / "pipeline_replay" / "cassettes"

CASSETTE_MODE_ENV = "SCOUT_CASSETTE_MODE"
CASSETTE_MODE_REPLAY = "replay"
CASSETTE_MODE_RECORD = "record"
CASSETTE_MODE_LIVE = "live"
