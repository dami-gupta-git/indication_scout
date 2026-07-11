"""Unit tests for spec loading + validation. No LLM, no network."""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from tests.regression.failure_buckets import Bucket
from tests.regression.layer2_structural.loader import load_spec


def _write_yaml(tmp_path, data: dict):
    p = tmp_path / "drug.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


def test_minimal_spec_loads(tmp_path):
    p = _write_yaml(tmp_path, {"drug": "x"})
    spec = load_spec(p)
    assert spec.drug == "x"
    assert spec.required_ncts_surfaced == []


def test_bucket_string_validates_against_enum(tmp_path):
    p = _write_yaml(
        tmp_path,
        {
            "drug": "x",
            "required_ncts_surfaced": [
                {
                    "bucket": "literature_coverage",
                    "indication": "adhd",
                    "ncts": ["NCT001"],
                }
            ],
        },
    )
    spec = load_spec(p)
    assert spec.required_ncts_surfaced[0].bucket is Bucket.LITERATURE_COVERAGE


def test_invalid_bucket_raises(tmp_path):
    p = _write_yaml(
        tmp_path,
        {
            "drug": "x",
            "required_ncts_surfaced": [
                {
                    "bucket": "not_a_bucket",
                    "indication": "adhd",
                    "ncts": ["NCT001"],
                }
            ],
        },
    )
    with pytest.raises(ValidationError):
        load_spec(p)


def test_real_bupropion_spec_loads():
    # Sanity-check the actual committed spec parses cleanly.
    from pathlib import Path

    spec_path = (
        Path(__file__).parent.parent / "specs" / "bupropion.yaml"
    )
    spec = load_spec(spec_path)
    assert spec.drug == "bupropion"
    assert spec.ranked_order
    assert spec.forbidden_in_ranked
    assert spec.required_ncts_surfaced
