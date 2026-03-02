"""
Shared file-based cache utility.

Used by data source clients and services to avoid redundant network/LLM calls.
Cache entries are JSON files keyed by a SHA-256 hash of (namespace, params).
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from indication_scout.constants import CACHE_TTL

logger = logging.getLogger(__name__)


def cache_key(namespace: str, params: dict[str, Any]) -> str:
    """Return a deterministic hex digest for the given namespace and params."""
    raw = json.dumps({"ns": namespace, **params}, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def cache_get(
    namespace: str,
    params: dict[str, Any],
    cache_dir: Path,
) -> Any | None:
    """Return cached data if present and unexpired, otherwise None."""
    path = cache_dir / f"{cache_key(namespace, params)}.json"
    if not path.exists():
        return None
    try:
        entry = json.loads(path.read_text())
        age = (
            datetime.now() - datetime.fromisoformat(entry["cached_at"])
        ).total_seconds()
        if age > entry.get("ttl", CACHE_TTL):
            path.unlink(missing_ok=True)
            return None
        return entry["data"]
    except (json.JSONDecodeError, KeyError, ValueError):
        path.unlink(missing_ok=True)
        return None


def cache_set(
    namespace: str,
    params: dict[str, Any],
    data: Any,
    cache_dir: Path,
    ttl: int | None = None,
) -> None:
    """Write data to the cache under the given namespace and params."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "data": data,
        "cached_at": datetime.now().isoformat(),
        "ttl": ttl if ttl is not None else CACHE_TTL,
    }
    (cache_dir / f"{cache_key(namespace, params)}.json").write_text(
        json.dumps(entry, default=str)
    )
