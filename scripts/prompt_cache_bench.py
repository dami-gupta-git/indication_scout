"""Before/after benchmark for Anthropic prompt caching, full pipeline, no monkeypatching.

Runs the full pipeline (run_analysis: supervisor + mechanism + literature + clinical_trials
agents) twice, back-to-back, for the same drug:

  - run 1 ("COLD"): nothing for this drug/prompt combination is in Anthropic's server-side
    cache yet (assuming it hasn't been run recently) - each agent's first turn WRITES the
    cache_control breakpoints (system prompt + tool defs) rather than reading them.
  - run 2 ("WARM"): run 1 already wrote those breakpoints, and they're within the 5-minute
    (or 1-hour) ephemeral TTL, so run 2's equivalent turns READ from cache instead.

This exercises the real shipped code path with no changes to agent source or behavior - it
simply compares a cold start to a warm one, which is what prompt caching is actually for
(e.g. concurrent/back-to-back requests for the same drug, or a user re-running after a partial
result). Captures usage_metadata by patching ChatAnthropic._agenerate (read-only tap, not a
caching interceptor) so no agent behavior changes - this only observes what the SDK returns.

Reports: wall-clock, cache hit rate (cache_read / (cache_read + input_tokens), summed over all
turns), USD cost from published per-token rates (see _RATES - correct if pricing changes), and
token breakdown.

Usage:
    python scripts/prompt_cache_bench.py <drug>

Requires ANTHROPIC_API_KEY / DB / etc. set up as for a normal `scout find` run. Each run is a
real, full, API-billed pipeline run.
"""

import argparse
import asyncio
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
CONSTANTS_FILE = ".env.constants"
constants_path = Path(__file__).parent.parent / CONSTANTS_FILE
load_dotenv(constants_path)
os.environ["CONSTANTS_FILE"] = str(constants_path)

# Override just for this benchmark process (not .env.constants, which is the shipped
# production value of 3) to exercise a wider concurrent fan-out.
os.environ["SUPERVISOR_INVESTIGATION_CAP"] = "5"

# .env sets SCOUT_CACHE_DIR=/cache (the Railway/Docker mount), which is read-only locally.
# Point at the repo-local warm cache so the bench runs on this machine.
os.environ.pop("SCOUT_CACHE_DIR", None)

from langchain_anthropic import ChatAnthropic  # noqa: E402

from indication_scout.services.analysis_runner import run_analysis  # noqa: E402

# Per-MTok USD rates. Sonnet-tier pricing as published by Anthropic; update here if pricing
# changes or llm_model (config.Settings.llm_model) moves to a different tier.
_RATES = {
    "input": 3.00,
    "cache_write_5m": 3.75,
    "cache_write_1h": 6.00,
    "cache_read": 0.30,
    "output": 15.00,
}


class _TurnUsageCapture:
    def __init__(self):
        self.turns: list[dict] = []

    def record(self, usage_metadata: dict | None) -> None:
        if not usage_metadata:
            return
        details = usage_metadata.get("input_token_details", {}) or {}
        self.turns.append(
            {
                "input_tokens": usage_metadata.get("input_tokens", 0),
                "output_tokens": usage_metadata.get("output_tokens", 0),
                "cache_read": details.get("cache_read", 0),
                "cache_write_5m": details.get("ephemeral_5m_input_tokens", 0),
                "cache_write_1h": details.get("ephemeral_1h_input_tokens", 0),
            }
        )


def _install_usage_tap(capture: _TurnUsageCapture):
    """Monkeypatch ChatAnthropic._agenerate to record usage_metadata off every LLM call.

    Read-only tap, not a behavior change: it calls straight through to the original
    _agenerate and only inspects the result. Returns a restore() callable. This is the
    single seam that sees every LLM call made by every agent (supervisor, mechanism,
    literature, clinical_trials) without touching per-agent source, since only the
    supervisor's own [LLMTURN] logging is currently live.
    """
    orig_agenerate = ChatAnthropic._agenerate

    async def patched_agenerate(self, *args, **kwargs):
        result = await orig_agenerate(self, *args, **kwargs)
        for generation in result.generations:
            msg = generation.message
            capture.record(getattr(msg, "usage_metadata", None))
        return result

    ChatAnthropic._agenerate = patched_agenerate

    def restore():
        ChatAnthropic._agenerate = orig_agenerate

    return restore


def _cost(turns: list[dict]) -> float:
    total = 0.0
    for t in turns:
        total += t["input_tokens"] / 1e6 * _RATES["input"]
        total += t["cache_write_5m"] / 1e6 * _RATES["cache_write_5m"]
        total += t["cache_write_1h"] / 1e6 * _RATES["cache_write_1h"]
        total += t["cache_read"] / 1e6 * _RATES["cache_read"]
        total += t["output_tokens"] / 1e6 * _RATES["output"]
    return total


def _hit_rate(turns: list[dict]) -> float | None:
    cache_read = sum(t["cache_read"] for t in turns)
    input_tokens = sum(t["input_tokens"] for t in turns)
    denom = cache_read + input_tokens
    return cache_read / denom if denom else None


async def _one_run(drug: str) -> dict:
    capture = _TurnUsageCapture()
    restore_tap = _install_usage_tap(capture)
    try:
        t0 = time.perf_counter()
        await run_analysis(drug)
        elapsed = time.perf_counter() - t0
    finally:
        restore_tap()
    return {
        "elapsed_s": elapsed,
        "turns": capture.turns,
        "n_turns": len(capture.turns),
        "cost_usd": _cost(capture.turns),
        "hit_rate": _hit_rate(capture.turns),
        "total_input_tokens": sum(t["input_tokens"] for t in capture.turns),
        "total_cache_read": sum(t["cache_read"] for t in capture.turns),
        "total_cache_write": sum(
            t["cache_write_5m"] + t["cache_write_1h"] for t in capture.turns
        ),
        "total_output_tokens": sum(t["output_tokens"] for t in capture.turns),
    }


def _report(label: str, result: dict) -> None:
    print(f"\n--- {label} ---")
    print(f"  wall-clock:  {result['elapsed_s']:.1f}s")
    print(f"  cost:        ${result['cost_usd']:.4f}")
    print(f"  hit rate:    {result['hit_rate']}")
    print(f"  turns:       {result['n_turns']}")
    print(f"  input tok:   {result['total_input_tokens']}")
    print(f"  cache_read:  {result['total_cache_read']}")
    print(f"  cache_write: {result['total_cache_write']}")
    print(f"  output tok:  {result['total_output_tokens']}")


async def main(drug: str):
    logging.basicConfig(level=logging.WARNING)

    print(f"=== run 1 (COLD — first run for {drug}, nothing cached yet) ===")
    cold = await _one_run(drug)
    _report("COLD (run 1)", cold)

    print(f"\n=== run 2 (WARM — run 1's cache_control breakpoints still within TTL) ===")
    warm = await _one_run(drug)
    _report("WARM (run 2)", warm)

    print(f"\n=== DELTA ===")
    print(
        f"  latency:  {cold['elapsed_s']:.1f}s -> {warm['elapsed_s']:.1f}s "
        f"({(1 - warm['elapsed_s'] / cold['elapsed_s']):.1%} faster)"
    )
    print(
        f"  cost:     ${cold['cost_usd']:.4f} -> ${warm['cost_usd']:.4f} "
        f"({(1 - warm['cost_usd'] / cold['cost_usd']):.1%} cheaper)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("drug")
    args = parser.parse_args()
    asyncio.run(main(args.drug))
