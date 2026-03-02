"""Unit tests for indication_scout.utils.cache."""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from indication_scout.utils.cache import cache_get, cache_key, cache_set


def _write_entry(path: Path, data: object, age_seconds: int, ttl: int) -> None:
    cached_at = (datetime.now() - timedelta(seconds=age_seconds)).isoformat()
    path.write_text(json.dumps({"data": data, "cached_at": cached_at, "ttl": ttl}))


def test_cache_get_returns_none_on_miss(tmp_path: Path) -> None:
    result = cache_get("ns", {"k": "v"}, tmp_path)
    assert result is None


def test_cache_get_returns_none_on_expired(tmp_path: Path) -> None:
    ttl = 100
    key = cache_key("ns", {"k": "v"})
    entry_path = tmp_path / f"{key}.json"
    _write_entry(entry_path, "stale_data", age_seconds=200, ttl=ttl)

    result = cache_get("ns", {"k": "v"}, tmp_path)

    assert result is None
    assert not entry_path.exists()


def test_cache_get_returns_data_on_hit(tmp_path: Path) -> None:
    ttl = 86400
    key = cache_key("ns", {"k": "v"})
    _write_entry(tmp_path / f"{key}.json", {"foo": "bar"}, age_seconds=10, ttl=ttl)

    result = cache_get("ns", {"k": "v"}, tmp_path)

    assert result == {"foo": "bar"}


def test_cache_get_handles_corrupt_file(tmp_path: Path) -> None:
    key = cache_key("ns", {"k": "v"})
    entry_path = tmp_path / f"{key}.json"
    entry_path.write_text("not valid json{{")

    result = cache_get("ns", {"k": "v"}, tmp_path)

    assert result is None
    assert not entry_path.exists()


def test_cache_set_writes_file(tmp_path: Path) -> None:
    cache_set("ns", {"k": "v"}, ["result1", "result2"], tmp_path)

    key = cache_key("ns", {"k": "v"})
    entry_path = tmp_path / f"{key}.json"
    assert entry_path.exists()

    entry = json.loads(entry_path.read_text())
    assert entry["data"] == ["result1", "result2"]
    assert "cached_at" in entry
    assert "ttl" in entry


def test_cache_set_custom_ttl(tmp_path: Path) -> None:
    cache_set("ns", {"k": "v"}, "data", tmp_path, ttl=999)

    key = cache_key("ns", {"k": "v"})
    entry = json.loads((tmp_path / f"{key}.json").read_text())
    assert entry["ttl"] == 999


def test_cache_key_is_deterministic() -> None:
    key1 = cache_key("organ_term", {"disease_name": "colorectal cancer"})
    key2 = cache_key("organ_term", {"disease_name": "colorectal cancer"})
    assert key1 == key2


def test_cache_key_namespace_separates_identical_params() -> None:
    key1 = cache_key("organ_term", {"disease_name": "colorectal cancer"})
    key2 = cache_key("expand_search_terms", {"disease_name": "colorectal cancer"})
    assert key1 != key2
