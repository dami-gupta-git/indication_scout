"""Lightweight metrics instrumentation.

Appends one JSON object per line (JSONL) to a metrics file. Each record captures a single
measured event (e.g. an end-to-end repurposing run). Failures to write are logged and
swallowed — instrumentation must never break a pipeline run.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from indication_scout.constants import METRICS_PATH

logger = logging.getLogger(__name__)


def record_metric(event: str, path: Path = METRICS_PATH, **fields: Any) -> None:
    """Append a single metric record to the JSONL file.

    `event` names the measurement (e.g. "repurposing_run"). `fields` are arbitrary
    JSON-serializable values merged into the record. A UTC ISO-8601 `timestamp` is added.
    """
    record = {"timestamp": datetime.now().isoformat(), "event": event, **fields}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as exc:
        logger.warning("Failed to write metric %r to %s: %s", event, path, exc)
