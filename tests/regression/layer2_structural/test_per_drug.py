"""Layer 2: spec-driven structural regression test.

Parametrized over every YAML in `tests/regression/specs/`. For each drug, the
test loads the spec and runs every assertion against the most recent
`test_reports/<drug>_*.json` — the payload a fresh `scout find -d <drug>`
writes. The spec itself is authored from the frozen `gold_standard/` snapshot
(see the scaffolder and README), but the test checks the LATEST run, so a
regression in the live pipeline is what fails here.

Each spec encodes the invariants extracted from the gold standard — the ranked
candidates, curated NCTs/PMIDs, demotions, and factual guards that a
regression must NOT silently flip. If an assertion fails, the latest run
stopped surfacing a signal the gold standard captured.

To produce a payload to test against:

    scout find -d bupropion            # writes test_reports/bupropion_<timestamp>.json

Failures roll up by bucket so the bioRxiv failure-mode analysis can reuse the
same taxonomy. Run:

    pytest -m regression_layer2
    pytest -m regression_layer2 -k bupropion
"""

from __future__ import annotations

from pathlib import Path

import pytest

from indication_scout.agents.supervisor.supervisor_output import SupervisorOutput

from tests.regression.failure_buckets import has_errors, render, summarize_buckets
from tests.regression.layer2_structural.assertions import run_spec
from tests.regression.layer2_structural.loader import discover_specs, load_spec

SPECS_DIR = Path(__file__).parent.parent / "specs"
TEST_REPORTS_DIR = Path(__file__).parent.parent.parent.parent / "test_reports"


def _latest_payload(drug: str) -> Path | None:
    """Most recent test_reports/<drug>_*.json by mtime, or None if missing."""
    candidates = sorted(
        TEST_REPORTS_DIR.glob(f"{drug}_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _spec_paths() -> list[Path]:
    if not SPECS_DIR.exists():
        return []
    return discover_specs(SPECS_DIR)


@pytest.mark.regression_layer2
@pytest.mark.parametrize(
    "spec_path",
    _spec_paths(),
    ids=lambda p: p.stem,
)
def test_spec(spec_path: Path) -> None:
    spec = load_spec(spec_path)
    payload_path = _latest_payload(spec.drug)
    if payload_path is None:
        pytest.skip(
            f"No payload at {TEST_REPORTS_DIR}/{spec.drug}_*.json. "
            f"Produce one with:  scout find -d {spec.drug}"
        )

    report = SupervisorOutput.model_validate_json(payload_path.read_text())

    # Layer 2 reads the markdown if it sits next to the payload, so
    # forbidden_phrase assertions can scan the rendered output. Optional —
    # phrase scoping to "summary" or "blurb" works without the MD.
    md_path = payload_path.with_suffix(".md")
    rendered_md = md_path.read_text() if md_path.exists() else ""

    diffs = run_spec(spec, report, rendered_md=rendered_md)

    bucket_summary = summarize_buckets(diffs)
    if has_errors(diffs):
        pytest.fail(
            f"\nLayer 2 regression failed for {spec.drug} "
            f"(payload: {payload_path.name}):\n"
            f"{render(diffs)}\n\n"
            f"Failures by bucket:\n"
            + "\n".join(f"  {b.value}: {n}" for b, n in bucket_summary.items())
        )
