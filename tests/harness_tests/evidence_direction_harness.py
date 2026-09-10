"""Harness for evidence-quality weighting after per-paper relevance and direction are known.

Run: .venv/bin/python tests/harness_tests/evidence_direction_harness.py
"""

import asyncio
import json
import logging
from pathlib import Path

from indication_scout.services.retrieval import (
    AbstractResult,
    _judge_overall_evidence_direction,
)

logger = logging.getLogger(__name__)

_REPORT = (
    Path(__file__).parents[1]
    / "regression/gold_standard/bupropion_2026-09-08_22-45-20.json"
)
_VERDICTS = {
    "39175424": "supporting",
    "17414245": "contradicting",
    "27126398": "contradicting",
}


def _ptsd_abstracts() -> list[AbstractResult]:
    report = json.loads(_REPORT.read_text())
    finding = next(
        item
        for item in report["disease_findings"]
        if item["disease"] == "post-traumatic stress disorder"
    )
    by_pmid = {
        item["pmid"]: item for item in finding["literature"]["semantic_search_results"]
    }
    return [AbstractResult(**by_pmid[pmid]) for pmid in _VERDICTS]


async def main() -> None:
    judgment = await _judge_overall_evidence_direction(
        "bupropion",
        "post-traumatic stress disorder",
        _ptsd_abstracts(),
        _VERDICTS,
    )
    if judgment is None:
        raise AssertionError("The focused evidence judgment was unavailable")
    if judgment.direction != "contradicts":
        raise AssertionError(f"Expected contradicts, got {judgment.direction}")
    if not judgment.summary:
        raise AssertionError("Expected a cited summary")
    if not judgment.key_findings:
        raise AssertionError("Expected cited key findings")
    logger.info("OK bupropion x PTSD: %s", judgment.direction)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
