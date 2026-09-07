"""Evaluate therapeutic-intent classification in the production literature synthesis prompt.

The harness sends real PubMed abstracts from the local pgvector database through the current
``synthesize.txt`` prompt. It checks two cases where the candidate disease is only the patient
population and one control where bupropion directly treats the candidate disease.

Run: .venv/bin/python tests/harness_tests/literature_relevance_intent_harness.py [runs]
"""

import asyncio
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, model_validator
from sqlalchemy import create_engine, text

from indication_scout.config import get_settings
from indication_scout.services.llm import parse_last_json_object, query_llm

logger = logging.getLogger(__name__)

_PROMPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "indication_scout"
    / "prompts"
    / "synthesize.txt"
)
_PROMPT = _PROMPT_PATH.read_text()
_VERDICT = Literal["contaminated", "supporting", "contradicting", "mixed"]
_BASIS = Literal["drug_specific", "approved", "class_level", "none"]
_STRENGTH = Literal["strong", "moderate", "weak", "none"]
_DIRECTION = Literal["supports", "contradicts", "mixed", "none"]


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


class IntentOutcome(BaseModel):
    verdicts: dict[str, _VERDICT]
    evidence_basis: _BASIS
    study_count: int
    strength: _STRENGTH
    direction: _DIRECTION
    is_observational: bool | None = None
    is_animal_only: bool | None = None

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values):
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


@dataclass(frozen=True)
class IntentCase:
    name: str
    drug: str
    disease: str
    pmids: tuple[str, ...]
    expected: IntentOutcome


CASES: tuple[IntentCase, ...] = (
    IntentCase(
        name="MS population, sexual-dysfunction treatment",
        drug="bupropion",
        disease="Multiple Sclerosis",
        pmids=("36410223",),
        expected=IntentOutcome(
            verdicts={"36410223": "contaminated"},
            evidence_basis="none",
            study_count=0,
            strength="none",
            direction="none",
            is_observational=None,
            is_animal_only=None,
        ),
    ),
    IntentCase(
        name="Schizophrenia population, smoking and weight treatment",
        drug="bupropion",
        disease="Schizophrenia",
        pmids=("12079730", "11694208", "17632223", "15876899", "34735098"),
        expected=IntentOutcome(
            verdicts={
                "12079730": "contaminated",
                "11694208": "contaminated",
                "17632223": "contaminated",
                "15876899": "contaminated",
                "34735098": "contaminated",
            },
            evidence_basis="none",
            study_count=0,
            strength="none",
            direction="none",
            is_observational=None,
            is_animal_only=None,
        ),
    ),
    IntentCase(
        name="ADHD direct-treatment control",
        drug="bupropion",
        disease="Attention Deficit-Hyperactivity Disorder",
        pmids=("11156812", "15820237", "25325205"),
        expected=IntentOutcome(
            verdicts={
                "11156812": "supporting",
                "15820237": "supporting",
                "25325205": "supporting",
            },
            evidence_basis="drug_specific",
            study_count=3,
            strength="strong",
            direction="supports",
            is_observational=False,
            is_animal_only=False,
        ),
    ),
)


def _load_abstracts(pmids: set[str]) -> dict[str, HarnessAbstract]:
    settings = get_settings()
    engine = create_engine(settings.database_url)
    try:
        with engine.connect() as connection:
            rows = connection.execute(
                text(
                    "SELECT pmid, title, abstract FROM pubmed_abstracts "
                    "WHERE pmid = ANY(:pmids)"
                ),
                {"pmids": sorted(pmids)},
            ).fetchall()
    finally:
        engine.dispose()

    abstracts = {
        str(row[0]): HarnessAbstract(
            pmid=row[0],
            title=row[1],
            abstract=row[2],
        )
        for row in rows
    }
    missing = pmids - set(abstracts)
    if missing:
        raise RuntimeError(f"Missing PubMed abstracts in pgvector: {sorted(missing)}")
    return abstracts


def _format_abstracts(
    case: IntentCase, abstracts: dict[str, HarnessAbstract]
) -> str:
    return "\n\n".join(
        f"PMID: {abstracts[pmid].pmid}\n"
        f"Title: {abstracts[pmid].title}\n"
        f"Abstract: {abstracts[pmid].abstract}"
        for pmid in case.pmids
    )


async def _evaluate_once(
    case: IntentCase, abstracts: dict[str, HarnessAbstract]
) -> IntentOutcome | None:
    prompt = _PROMPT.format(
        drug_name=case.drug,
        disease_name=case.disease,
        approved_indications="(none)",
        abstracts=_format_abstracts(case, abstracts),
    )
    try:
        response = await query_llm(prompt)
        parsed = parse_last_json_object(response)
        if parsed is None:
            logger.error("%s returned no JSON object", case.name)
            return None
        return IntentOutcome.model_validate(parsed)
    except Exception:
        logger.exception("%s could not be evaluated", case.name)
        return None


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    runs_per_case = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    if runs_per_case < 1:
        raise ValueError("runs must be at least 1")

    all_pmids = {pmid for case in CASES for pmid in case.pmids}
    abstracts = _load_abstracts(all_pmids)
    all_passed = True

    for case in CASES:
        outcomes = await asyncio.gather(
            *(_evaluate_once(case, abstracts) for _ in range(runs_per_case))
        )
        passed = sum(outcome == case.expected for outcome in outcomes)
        case_passed = passed == runs_per_case
        all_passed = all_passed and case_passed
        logger.info(
            "%s: %d/%d exact matches",
            case.name,
            passed,
            runs_per_case,
        )
        if not case_passed:
            logger.info("Expected: %s", case.expected.model_dump_json())
            for outcome in outcomes:
                if outcome != case.expected:
                    logger.info(
                        "Observed: %s",
                        outcome.model_dump_json() if outcome is not None else "null",
                    )

    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
