"""Load and validate per-drug regression specs from YAML."""

from __future__ import annotations

from pathlib import Path

import yaml

from tests.regression.layer2_structural.spec import DrugSpec


def load_spec(path: Path) -> DrugSpec:
    """Parse one YAML file into a validated `DrugSpec`.

    Raises `pydantic.ValidationError` if the spec is malformed — callers can
    let that propagate; a broken spec should fail the test loudly, not be
    silently skipped.
    """
    raw = yaml.safe_load(path.read_text())
    return DrugSpec.model_validate(raw)


def discover_specs(specs_dir: Path) -> list[Path]:
    """All *.yaml files in the specs directory, sorted for stable test order."""
    return sorted(specs_dir.glob("*.yaml"))
