"""Per-agent LLM cost/token summary for one Langfuse trace.

Usage:
    python scripts/trace_summary.py <trace_id>
    python scripts/trace_summary.py            # defaults to the most recent trace

Collapses a trace's hundreds of spans into a per-agent table (calls, input/output tokens,
cost) by attributing each LLM generation to the supervisor tool/phase it nests under. This
is the answer to "how many LLM calls + tokens + cost did each agent make".

Reads Langfuse creds from .env: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL.
"""

import base64
import json
import os
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Supervisor tools / phases an LLM call may sit under. The nearest such ancestor name is
# the bucket we attribute a generation to; anything else rolls up to the top-level supervisor.
PHASE_NAMES = {
    "analyze_literature",
    "analyze_clinical_trials",
    "analyze_mechanism",
    "find_candidates",
    "investigate_top_candidates",
    "merge_and_dedup",
    "finalize_supervisor",
}


def _auth_header() -> dict[str, str]:
    pk = os.environ["LANGFUSE_PUBLIC_KEY"]
    sk = os.environ["LANGFUSE_SECRET_KEY"]
    token = base64.b64encode(f"{pk}:{sk}".encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {token}"}


def _get(path: str) -> dict:
    base = os.environ["LANGFUSE_BASE_URL"].rstrip("/")
    req = urllib.request.Request(base + path, headers=_auth_header())
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def _latest_trace_id() -> str:
    data = _get("/api/public/traces?limit=1")
    if not data.get("data"):
        sys.exit("No traces found in this project.")
    return data["data"][0]["id"]


def _all_observations(trace_id: str) -> list[dict]:
    obs: list[dict] = []
    page = 1
    while True:
        d = _get(f"/api/public/observations?traceId={trace_id}&limit=100&page={page}")
        obs += d["data"]
        if page >= d["meta"]["totalPages"]:
            break
        page += 1
    return obs


def _bucket_for(obs: dict, by_id: dict[str, dict]) -> str:
    """Climb the parent chain; attribute to the nearest known phase name."""
    cur = obs
    seen: set[str] = set()
    while cur:
        if (cur.get("name") or "") in PHASE_NAMES:
            return cur["name"]
        pid = cur.get("parentObservationId")
        if not pid or pid in seen:
            break
        seen.add(pid)
        cur = by_id.get(pid)
    return "supervisor (top-level)"


def main() -> None:
    trace_id = sys.argv[1] if len(sys.argv) > 1 else _latest_trace_id()
    obs = _all_observations(trace_id)
    by_id = {o["id"]: o for o in obs}
    gens = [o for o in obs if o.get("type") == "GENERATION"]

    agg: dict[str, dict] = defaultdict(
        lambda: {"calls": 0, "in": 0, "out": 0, "cost": 0.0}
    )
    for g in gens:
        bucket = _bucket_for(g, by_id)
        usage = g.get("usage") or {}
        agg[bucket]["calls"] += 1
        agg[bucket]["in"] += usage.get("input") or 0
        agg[bucket]["out"] += usage.get("output") or 0
        agg[bucket]["cost"] += (
            g.get("calculatedTotalCost") or g.get("totalCost") or 0
        )

    print(f"trace: {trace_id}")
    print(f"observations: {len(obs)} | LLM generations: {len(gens)}\n")
    header = f"{'agent/phase':<32}{'calls':>6}{'in_tok':>10}{'out_tok':>9}{'cost$':>10}"
    print(header)
    print("-" * len(header))
    total = defaultdict(float)
    for name, v in sorted(agg.items(), key=lambda kv: -kv[1]["cost"]):
        print(
            f"{name:<32}{v['calls']:>6}{v['in']:>10}{v['out']:>9}{v['cost']:>10.4f}"
        )
        for k in ("calls", "in", "out", "cost"):
            total[k] += v[k]
    print("-" * len(header))
    print(
        f"{'TOTAL':<32}{int(total['calls']):>6}{int(total['in']):>10}"
        f"{int(total['out']):>9}{total['cost']:>10.4f}"
    )


if __name__ == "__main__":
    main()
