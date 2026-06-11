"""Before/after benchmark for the trailing-turn removal.

Runs the full pipeline (run_analysis) warm: one throwaway warm-up run, then one timed run.
Captures the [TIMING] run_analysis total and the per-agent [LLMTURN] turn counts so the
before/after can be compared on turns (deterministic) and seconds (noisy ±15s).

Usage: python scripts/trailing_turn_bench.py <drug>
"""

import asyncio
import logging
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
CONSTANTS_FILE = ".env.constants"
constants_path = Path(__file__).parent.parent / CONSTANTS_FILE
load_dotenv(constants_path)
os.environ["CONSTANTS_FILE"] = str(constants_path)

# .env sets SCOUT_CACHE_DIR=/cache (the Railway/Docker mount), which is read-only
# locally. Point at the repo-local warm cache so the bench runs on this machine.
os.environ.pop("SCOUT_CACHE_DIR", None)

from indication_scout.services.analysis_runner import run_analysis

# Capture [LLMTURN] "<agent> ...: N turns" lines and [TIMING] totals.
_TURN_SUMMARY = re.compile(r"\[LLMTURN\] (.+?): (\d+) turns")


class _TurnCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.turn_lines: list[tuple[str, int]] = []

    def emit(self, record):
        msg = record.getMessage()
        m = _TURN_SUMMARY.search(msg)
        if m:
            self.turn_lines.append((m.group(1).strip(), int(m.group(2))))


async def _timed_run(drug: str) -> float:
    t0 = time.perf_counter()
    await run_analysis(drug)
    return time.perf_counter() - t0


async def main(drug: str):
    logging.basicConfig(level=logging.WARNING)
    cap = _TurnCapture()
    logging.getLogger().addHandler(cap)

    print(f"=== warm-up run ({drug}) ===")
    await _timed_run(drug)

    cap.turn_lines.clear()
    print(f"\n=== timed run ({drug}) ===")
    elapsed = await _timed_run(drug)

    print(f"\n=== result ({drug}) ===")
    print(f"total: {elapsed:.1f}s")
    print("turn counts this run:")
    for agent, turns in cap.turn_lines:
        print(f"  {agent}: {turns} turns")


if __name__ == "__main__":
    drug = sys.argv[1] if len(sys.argv) > 1 else "metformin"
    asyncio.run(main(drug))
