"""Unit tests for scripts/seed_examples_from_reports — holdout exclusion. No network/DB."""

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[3] / "scripts" / "seed_examples_from_reports.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("seed_examples_from_reports", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def seed_mod(tmp_path, monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "TEST_REPORTS_DIR", tmp_path)
    return module


def _write_report(reports_dir, filename, *, date_before):
    (reports_dir / filename).write_text(
        json.dumps({"drug_name": "metformin", "date_before": date_before})
    )


def test_is_holdout_production(seed_mod, tmp_path):
    _write_report(tmp_path, "metformin_2026-07-13_16-00-00.json", date_before=None)
    assert seed_mod._is_holdout(tmp_path / "metformin_2026-07-13_16-00-00.json") is False


def test_is_holdout_when_date_before_set(seed_mod, tmp_path):
    _write_report(tmp_path, "metformin_2026-07-13_16-00-00.json", date_before="2022-01-01")
    assert seed_mod._is_holdout(tmp_path / "metformin_2026-07-13_16-00-00.json") is True


def test_is_holdout_missing_key_is_production(seed_mod, tmp_path):
    (tmp_path / "old.json").write_text(json.dumps({"drug_name": "metformin"}))
    assert seed_mod._is_holdout(tmp_path / "old.json") is False


def test_is_holdout_unreadable_is_excluded(seed_mod, tmp_path):
    (tmp_path / "bad.json").write_text("{not valid json")
    assert seed_mod._is_holdout(tmp_path / "bad.json") is True


def test_latest_report_excludes_holdout(seed_mod, tmp_path):
    # Holdout run is newer, but the older production run must be chosen.
    _write_report(tmp_path, "metformin_2026-07-13_16-00-00.json", date_before=None)
    _write_report(
        tmp_path,
        "metformin_holdout_2022-01-01_2026-07-13_18-00-00.json",
        date_before="2022-01-01",
    )

    found = seed_mod._latest_report("metformin")

    assert found is not None
    assert found[0].name == "metformin_2026-07-13_16-00-00.json"


def test_latest_report_none_when_only_holdout(seed_mod, tmp_path):
    _write_report(
        tmp_path,
        "metformin_holdout_2022-01-01_2026-07-13_18-00-00.json",
        date_before="2022-01-01",
    )

    assert seed_mod._latest_report("metformin") is None


def test_all_report_drugs_drops_holdout_only(seed_mod, tmp_path):
    _write_report(tmp_path, "bupropion_2026-07-13_16-00-00.json", date_before=None)
    _write_report(
        tmp_path,
        "metformin_holdout_2022-01-01_2026-07-13_18-00-00.json",
        date_before="2022-01-01",
    )

    assert seed_mod._all_report_drugs() == ["bupropion"]
