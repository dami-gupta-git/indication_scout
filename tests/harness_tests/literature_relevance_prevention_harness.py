"""Regression harness for prevention evidence in the literature synthesis prompt.

The full 15-paper metformin-preeclampsia bundle is required to reproduce the failure. Run with:

    .venv/bin/python tests/harness_tests/literature_relevance_prevention_harness.py [runs]
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, model_validator
from sqlalchemy import create_engine, text

from indication_scout.config import get_settings
from indication_scout.services.llm import parse_last_json_object, query_llm

logger = logging.getLogger(__name__)

_PROMPT = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "indication_scout"
    / "prompts"
    / "synthesize.txt"
).read_text()
_VERDICT = Literal["contaminated", "supporting", "contradicting", "mixed"]

_PMIDS: tuple[str, ...] = (
    "34551918",
    "27435163",
    "39236318",
    "25467617",
    "30704873",
    "20926533",
    "40176581",
    "29490031",
    "17535842",
    "40448705",
    "17330831",
    "30399639",
    "29044702",
    "41130438",
    "34499672",
)

_EXPECTED_RELEVANT = {
    "34551918",
    "39236318",
    "25467617",
    "20926533",
    "30399639",
    "29044702",
}
_EXPECTED_CONTAMINATED = set(_PMIDS) - _EXPECTED_RELEVANT


class HarnessAbstract(BaseModel):
    pmid: str
    title: str
    abstract: str

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values):
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


class PreventionOutcome(BaseModel):
    verdicts: dict[str, _VERDICT]
    evidence_basis: Literal["drug_specific", "approved", "class_level", "none"]
    study_count: int
    strength: Literal["strong", "moderate", "weak", "none"]
    direction: Literal["supports", "contradicts", "mixed", "none"]
    is_observational: bool | None = None
    is_animal_only: bool | None = None

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values):
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


def _load_abstracts() -> list[HarnessAbstract]:
    settings = get_settings()
    engine = create_engine(settings.database_url)
    try:
        with engine.connect() as connection:
            rows = connection.execute(
                text(
                    "SELECT pmid, title, abstract FROM pubmed_abstracts "
                    "WHERE pmid = ANY(:pmids)"
                ),
                {"pmids": list(_PMIDS)},
            ).fetchall()
    finally:
        engine.dispose()

    by_pmid = {
        str(row[0]): HarnessAbstract(
            pmid=str(row[0]),
            title=row[1],
            abstract=row[2],
        )
        for row in rows
    }
    missing = set(_PMIDS) - set(by_pmid)
    if missing:
        raise RuntimeError(f"Missing PubMed abstracts in pgvector: {sorted(missing)}")
    return [by_pmid[pmid] for pmid in _PMIDS]


async def _evaluate_once(abstracts: list[HarnessAbstract]) -> PreventionOutcome | None:
    formatted = "\n\n".join(
        f"PMID: {item.pmid}\nTitle: {item.title}\nAbstract: {item.abstract}"
        for item in abstracts
    )
    prompt = _PROMPT.format(
        drug_name="metformin",
        disease_name="Preeclampsia",
        approved_indications="Type 2 diabetes mellitus",
        abstracts=formatted,
    )
    try:
        response = await query_llm(prompt)
        parsed = parse_last_json_object(response)
        if parsed is None:
            logger.error("The prevention bundle returned no JSON object")
            return None
        return PreventionOutcome.model_validate(parsed)
    except Exception:
        logger.exception("The prevention bundle could not be evaluated")
        return None


def _matches(outcome: PreventionOutcome | None) -> bool:
    if outcome is None:
        return False
    relevant = {
        pmid for pmid, verdict in outcome.verdicts.items() if verdict != "contaminated"
    }
    contaminated = {
        pmid for pmid, verdict in outcome.verdicts.items() if verdict == "contaminated"
    }
    return (
        set(outcome.verdicts) == set(_PMIDS)
        and relevant == _EXPECTED_RELEVANT
        and contaminated == _EXPECTED_CONTAMINATED
        and outcome.evidence_basis == "drug_specific"
        and outcome.study_count == 6
        and outcome.strength == "strong"
        and outcome.direction == "mixed"
        and outcome.is_observational is False
        and outcome.is_animal_only is False
    )


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    if runs < 1:
        raise ValueError("runs must be at least 1")

    abstracts = _load_abstracts()
    outcomes = await asyncio.gather(*(_evaluate_once(abstracts) for _ in range(runs)))
    passed = sum(_matches(outcome) for outcome in outcomes)
    logger.info("Metformin preeclampsia prevention bundle: %d/%d matches", passed, runs)
    if passed != runs:
        for outcome in outcomes:
            if not _matches(outcome):
                logger.info(
                    "Observed: %s",
                    outcome.model_dump_json() if outcome is not None else "null",
                )
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
