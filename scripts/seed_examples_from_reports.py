"""Populate seed_examples/ from saved test_reports/ payloads.

For each requested drug, picks the latest test_reports/{drug}_{timestamp}.json
payload, validates it as a SupervisorOutput, copies it to
seed_examples/{drug}.json, and records the report's capture time (parsed from the
filename timestamp) into seed_examples/captured_at.json. Existing manifest entries
for drugs not processed are left untouched.

Usage:
    python scripts/seed_examples_from_reports.py                 # every drug in test_reports/
    python scripts/seed_examples_from_reports.py metformin bupropion
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from indication_scout.agents.supervisor.supervisor_output import SupervisorOutput
from indication_scout.constants import EXAMPLE_SEED_DIR
from indication_scout.helpers.drug_helpers import normalize_drug_name

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_REPORTS_DIR = _PROJECT_ROOT / "test_reports"
# Timestamp suffix written by `scout find` (e.g. metformin_2026-06-12_17-21-39.json).
_TIMESTAMP_FMT = "%Y-%m-%d_%H-%M-%S"


def _latest_report(drug: str) -> tuple[Path, float] | None:
    """Return (path, capture_epoch) for the newest test_reports payload, or None.

    The capture epoch is parsed from the filename timestamp (naive local time),
    matching what seed_reports/examples expect in captured_at.json.
    """
    latest: tuple[Path, float] | None = None
    for path in TEST_REPORTS_DIR.glob(f"{drug}_*.json"):
        stamp = path.stem[len(drug) + 1 :]  # strip "{drug}_"
        try:
            epoch = datetime.strptime(stamp, _TIMESTAMP_FMT).timestamp()
        except ValueError:
            logger.warning("Skipping unparseable filename %s", path.name)
            continue
        if latest is None or epoch > latest[1]:
            latest = (path, epoch)
    return latest


def _all_report_drugs() -> list[str]:
    """Return every distinct drug with a parseable test_reports payload, sorted."""
    drugs: set[str] = set()
    for path in TEST_REPORTS_DIR.glob("*.json"):
        stem = path.stem
        sep = stem.rfind("_", 0, stem.rfind("_"))  # split before the date_time stamp
        if sep == -1:
            logger.warning("Skipping unparseable filename %s", path.name)
            continue
        drug, stamp = stem[:sep], stem[sep + 1 :]
        try:
            datetime.strptime(stamp, _TIMESTAMP_FMT)
        except ValueError:
            logger.warning("Skipping unparseable filename %s", path.name)
            continue
        drugs.add(drug)
    return sorted(drugs)


def seed_from_reports(drugs: list[str]) -> None:
    """Copy the latest payload for each drug into the seed dir and update the manifest."""
    EXAMPLE_SEED_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = EXAMPLE_SEED_DIR / "captured_at.json"
    captured_at: dict[str, float] = {}
    if manifest_path.exists():
        captured_at = json.loads(manifest_path.read_text())

    for raw in drugs:
        drug = normalize_drug_name(raw)
        found = _latest_report(drug)
        if found is None:
            logger.warning("No test_reports payload for %s; leaving existing seed", drug)
            continue
        src, epoch = found

        payload = src.read_text()
        try:
            SupervisorOutput.model_validate_json(payload)
        except Exception:  # noqa: BLE001 — never seed a payload that won't load at serve time
            logger.exception("Payload for %s failed to validate; skipping %s", drug, src.name)
            continue

        (EXAMPLE_SEED_DIR / f"{drug}.json").write_text(payload)
        captured_at[drug] = epoch
        logger.info("Seeded %s from %s (captured %s)", drug, src.name,
                    datetime.fromtimestamp(epoch).isoformat())

    manifest_path.write_text(json.dumps(captured_at, indent=2, sort_keys=True) + "\n")
    logger.info("Updated manifest %s", manifest_path)


def main() -> None:
    drugs = sys.argv[1:] or _all_report_drugs()
    seed_from_reports(drugs)


if __name__ == "__main__":
    main()
