"""Regression harness for isolated literature therapeutic-target judgments.

Uses checked-in sildenafil and metformin regression abstracts. Run with:

    .venv/bin/python tests/harness_tests/literature_target_gate_harness.py [runs]
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from indication_scout.services.retrieval import (
    AbstractResult,
    _judge_pmid_treats_disease,
)

logger = logging.getLogger(__name__)

_FIXTURES = (
    Path(__file__).resolve().parents[1]
    / "regression"
    / "gold_standard"
    / "sildenafil_2026-09-08_21-37-44.json",
    Path(__file__).resolve().parents[1]
    / "regression"
    / "gold_standard"
    / "metformin_2026-09-08_19-13-23.json",
)
_ISCHEMIC_STROKE_EXPECTED = {
    "12411660": True,
    "28343223": False,
    "29092891": True,
}
_ENDOTHELIAL_DYSFUNCTION_EXPECTED = {
    "19837434": False,
    "21680809": False,
    "18036451": False,
}
_PREECLAMPSIA_EXPECTED = {
    "34551918": True,
    "39236318": True,
    "25467617": True,
}
_PMIDS = (
    set(_ISCHEMIC_STROKE_EXPECTED)
    | set(_ENDOTHELIAL_DYSFUNCTION_EXPECTED)
    | set(_PREECLAMPSIA_EXPECTED)
)


def _collect_abstracts(value: Any, found: dict[str, AbstractResult]) -> None:
    if isinstance(value, dict):
        pmid = str(value.get("pmid", ""))
        if pmid in _PMIDS and value.get("title") and value.get("abstract"):
            found.setdefault(
                pmid,
                AbstractResult(
                    pmid=pmid,
                    title=value["title"],
                    abstract=value["abstract"],
                    similarity=float(value.get("similarity", 0.0)),
                    pubtype=value.get("pubtype") or [],
                ),
            )
        for child in value.values():
            _collect_abstracts(child, found)
    elif isinstance(value, list):
        for child in value:
            _collect_abstracts(child, found)


def _load_abstracts() -> dict[str, AbstractResult]:
    found: dict[str, AbstractResult] = {}
    for fixture in _FIXTURES:
        _collect_abstracts(json.loads(fixture.read_text()), found)
    missing = _PMIDS - set(found)
    if missing:
        raise RuntimeError(f"Missing fixture abstracts: {sorted(missing)}")
    return found


async def _evaluate_once(abstracts: dict[str, AbstractResult]) -> bool:
    with TemporaryDirectory() as cache_dir:
        stroke = await _judge_pmid_treats_disease(
            "CHEMBL192",
            "sildenafil",
            "ischemic stroke",
            [abstracts[pmid] for pmid in _ISCHEMIC_STROKE_EXPECTED],
            Path(cache_dir),
        )
        endothelial = await _judge_pmid_treats_disease(
            "CHEMBL192",
            "sildenafil",
            "endothelial dysfunction",
            [abstracts[pmid] for pmid in _ENDOTHELIAL_DYSFUNCTION_EXPECTED],
            Path(cache_dir),
        )
        preeclampsia = await _judge_pmid_treats_disease(
            "CHEMBL1431",
            "metformin",
            "preeclampsia",
            [abstracts[pmid] for pmid in _PREECLAMPSIA_EXPECTED],
            Path(cache_dir),
        )
    if (
        stroke != _ISCHEMIC_STROKE_EXPECTED
        or endothelial != _ENDOTHELIAL_DYSFUNCTION_EXPECTED
        or preeclampsia != _PREECLAMPSIA_EXPECTED
    ):
        logger.error("Stroke observed: %s", stroke)
        logger.error("Endothelial observed: %s", endothelial)
        logger.error("Preeclampsia observed: %s", preeclampsia)
        return False
    return True


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    if runs < 1:
        raise ValueError("runs must be at least 1")

    abstracts = _load_abstracts()
    outcomes = await asyncio.gather(*(_evaluate_once(abstracts) for _ in range(runs)))
    passed = sum(outcomes)
    logger.info("Literature therapeutic-target gate: %d/%d matches", passed, runs)
    if passed != runs:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
