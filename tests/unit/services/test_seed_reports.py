"""Unit tests for services/seed_reports — no network/LLM/DB; seed dir is patched."""

import json
from types import SimpleNamespace

import pytest

from indication_scout.agents.supervisor.supervisor_output import SupervisorOutput
from indication_scout.services import seed_reports


@pytest.fixture(autouse=True)
def seed_reports_enabled(monkeypatch):
    """Default the feature flag on; the disabled-path test overrides it."""
    monkeypatch.setattr(
        seed_reports, "get_settings", lambda: SimpleNamespace(seed_reports_enabled=True)
    )


def _write_seed(seed_dir, drug, captured_epoch, *, write_report=True):
    """Write a captured_at manifest entry and (optionally) the report file."""
    (seed_dir / "captured_at.json").write_text(json.dumps({drug: captured_epoch}))
    if write_report:
        report = SupervisorOutput(drug_name=drug, summary="seeded summary")
        (seed_dir / f"{drug}.json").write_text(report.model_dump_json())


@pytest.fixture
def seed_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(seed_reports, "EXAMPLE_SEED_DIR", tmp_path)
    return tmp_path


def test_returns_report_when_fresh(seed_dir, monkeypatch):
    monkeypatch.setattr(seed_reports.time, "time", lambda: 1000.0)
    _write_seed(seed_dir, "metformin", 1000.0 - 5 * 86400)  # 5 days old

    result = seed_reports.load_fresh_seed_report("Metformin")

    assert result is not None
    assert result.drug_name == "metformin"
    assert result.summary == "seeded summary"


def test_returns_none_when_disabled(seed_dir, monkeypatch):
    monkeypatch.setattr(seed_reports.time, "time", lambda: 1000.0)
    _write_seed(seed_dir, "metformin", 1000.0 - 5 * 86400)  # fresh, but flag off
    monkeypatch.setattr(
        seed_reports, "get_settings", lambda: SimpleNamespace(seed_reports_enabled=False)
    )

    assert seed_reports.load_fresh_seed_report("metformin") is None


def test_returns_none_when_stale(seed_dir, monkeypatch):
    monkeypatch.setattr(seed_reports.time, "time", lambda: 1000.0)
    _write_seed(seed_dir, "metformin", 1000.0 - 31 * 86400)  # 31 days old

    assert seed_reports.load_fresh_seed_report("metformin") is None


def test_returns_none_when_untracked(seed_dir):
    # manifest exists but has no entry for this drug
    (seed_dir / "captured_at.json").write_text(json.dumps({"aspirin": 1.0}))

    assert seed_reports.load_fresh_seed_report("metformin") is None


def test_returns_none_when_manifest_missing(seed_dir):
    assert seed_reports.load_fresh_seed_report("metformin") is None


def test_returns_none_when_report_file_missing(seed_dir, monkeypatch):
    monkeypatch.setattr(seed_reports.time, "time", lambda: 1000.0)
    _write_seed(seed_dir, "metformin", 1000.0, write_report=False)

    assert seed_reports.load_fresh_seed_report("metformin") is None


def test_returns_none_when_report_corrupt(seed_dir, monkeypatch):
    monkeypatch.setattr(seed_reports.time, "time", lambda: 1000.0)
    (seed_dir / "captured_at.json").write_text(json.dumps({"metformin": 1000.0}))
    (seed_dir / "metformin.json").write_text("{not valid json")

    assert seed_reports.load_fresh_seed_report("metformin") is None
