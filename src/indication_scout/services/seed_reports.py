"""Seed-report lookup for the analyse shortcut.

When a user analyses a drug, a committed seed report under EXAMPLE_SEED_DIR that is
younger than SEED_REPORT_TTL_SECONDS (per captured_at.json) is served instead of
running the agents. The drug name is normalized so synonyms/salt forms resolve to
the same seed file. Any miss — absent file, stale, untracked, or corrupt — returns
None so the caller falls through to the live pipeline.
"""

import json
import logging
import time

from indication_scout.agents.supervisor.supervisor_output import SupervisorOutput
from indication_scout.config import get_settings
from indication_scout.constants import EXAMPLE_SEED_DIR, SEED_REPORT_TTL_SECONDS
from indication_scout.helpers.drug_helpers import normalize_drug_name

logger = logging.getLogger(__name__)


def load_fresh_seed_report(drug_name: str) -> SupervisorOutput | None:
    """Return a fresh seed report for the drug, or None on any miss.

    Miss cases (all return None, logged): seed reports disabled via
    SEED_REPORTS_ENABLED, no captured_at.json, no entry for the drug, report file
    absent, report older than SEED_REPORT_TTL_SECONDS, or a report file that fails
    to validate.
    """
    if not get_settings().seed_reports_enabled:
        return None

    drug = normalize_drug_name(drug_name)

    manifest_path = EXAMPLE_SEED_DIR / "captured_at.json"
    if not manifest_path.exists():
        return None
    try:
        captured_at: dict[str, float] = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, ValueError):
        logger.warning("Corrupt seed manifest at %s; ignoring", manifest_path)
        return None

    captured = captured_at.get(drug)
    if captured is None:
        return None  # untracked → cannot prove freshness

    age = time.time() - captured
    if age > SEED_REPORT_TTL_SECONDS:
        logger.info("Seed report for %s is stale (%.0f days); running live", drug, age / 86400)
        return None

    report_path = EXAMPLE_SEED_DIR / f"{drug}.json"
    if not report_path.exists():
        logger.warning("Seed manifest lists %s but %s is missing", drug, report_path)
        return None

    try:
        report = SupervisorOutput.model_validate_json(report_path.read_text())
    except Exception:  # noqa: BLE001 — a corrupt seed file should fall through, not 500
        logger.exception("Corrupt seed report for %s; running live", drug)
        return None

    logger.info("Seed report hit for %s (%.0f days old)", drug, age / 86400)
    return report
