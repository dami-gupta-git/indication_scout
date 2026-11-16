"""Harness for the design fields of the per-paper grade prompt (is_human, is_controlled, is_interventional, is_pilot).

A literature signal closes only on a contradicting paper that is human, interventional and not a pilot, so a wrong
is_pilot reopens a closed signal and a wrong is_interventional closes one on observational data. The 2026-11-11 runs
marked two small placebo-controlled trials as pilots although neither abstract calls itself one, which reopened
bupropion x PTSD (17414245) and sildenafil x schizophrenia (18716761). The other cases are the metformin x Parkinson
closure papers and two non-pilot trials that must keep closing.

Runs the production prompt by default, or a candidate prompt file given as the second argument. Abstracts come from
pgvector. Run with:

    .venv/bin/python tests/harness_tests/pmid_grade_design_harness.py [runs] [prompt_path]
"""

import asyncio
import logging
import sys
from pathlib import Path

from pydantic import BaseModel, model_validator
from sqlalchemy import create_engine, text

from indication_scout.config import get_settings
from indication_scout.services.llm import parse_last_json_object, query_small_llm

logger = logging.getLogger(__name__)

_PRODUCTION_PROMPT = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "indication_scout"
    / "prompts"
    / "pmid_grade.txt"
)


class DesignCase(BaseModel):
    pmid: str
    drug: str
    disease: str
    is_human: bool
    is_controlled: bool
    is_interventional: bool
    is_pilot: bool
    note: str


class DesignJudgment(BaseModel):
    is_human: bool | None = None
    is_controlled: bool | None = None
    is_interventional: bool | None = None
    is_pilot: bool | None = None

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values):
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


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


_CASES: tuple[DesignCase, ...] = (
    DesignCase(
        pmid="17414245",
        drug="bupropion",
        disease="post-traumatic stress disorder",
        is_human=True,
        is_controlled=True,
        is_interventional=True,
        is_pilot=False,
        note="30-patient placebo-controlled trial, never called a pilot",
    ),
    DesignCase(
        pmid="18716761",
        drug="sildenafil",
        disease="schizophrenia",
        is_human=True,
        is_controlled=True,
        is_interventional=True,
        is_pilot=False,
        note="17-patient single-dose placebo crossover, never called a pilot",
    ),
    DesignCase(
        pmid="40385486",
        drug="metformin",
        disease="Parkinson disease",
        is_human=True,
        is_controlled=True,
        is_interventional=True,
        is_pilot=True,
        note="title says placebo-controlled pilot study",
    ),
    DesignCase(
        pmid="42216231",
        drug="metformin",
        disease="Parkinson disease",
        is_human=True,
        is_controlled=False,
        is_interventional=False,
        is_pilot=False,
        note="retrospective active-comparator cohort",
    ),
    DesignCase(
        pmid="42627515",
        drug="metformin",
        disease="Parkinson disease",
        is_human=True,
        is_controlled=False,
        is_interventional=False,
        is_pilot=False,
        note="Mendelian randomization",
    ),
    DesignCase(
        pmid="31514200",
        drug="imatinib",
        disease="Glioblastoma",
        is_human=True,
        is_controlled=False,
        is_interventional=True,
        is_pilot=False,
        note="single-arm phase II, drug given",
    ),
    DesignCase(
        pmid="30792154",
        drug="metformin",
        disease="Miscarriage",
        is_human=True,
        is_controlled=True,
        is_interventional=True,
        is_pilot=False,
        note="PregMet2 multicentre placebo RCT",
    ),
)

_FIELDS: tuple[str, ...] = ("is_human", "is_controlled", "is_interventional", "is_pilot")


def _load_abstracts() -> dict[str, HarnessAbstract]:
    settings = get_settings()
    engine = create_engine(settings.database_url)
    pmids = [case.pmid for case in _CASES]
    try:
        with engine.connect() as connection:
            rows = connection.execute(
                text(
                    "SELECT pmid, title, abstract FROM pubmed_abstracts "
                    "WHERE pmid = ANY(:pmids)"
                ),
                {"pmids": pmids},
            ).fetchall()
    finally:
        engine.dispose()

    by_pmid = {
        str(row[0]): HarnessAbstract(pmid=str(row[0]), title=row[1], abstract=row[2])
        for row in rows
    }
    missing = set(pmids) - set(by_pmid)
    if missing:
        raise RuntimeError(f"Missing PubMed abstracts in pgvector: {sorted(missing)}")
    return by_pmid


async def _grade_once(
    prompt_template: str, case: DesignCase, item: HarnessAbstract
) -> DesignJudgment | None:
    prompt = prompt_template.format(
        drug=case.drug,
        disease=case.disease,
        pmid=item.pmid,
        title=item.title,
        abstract=item.abstract,
    )
    try:
        response = await query_small_llm(prompt)
    except Exception:
        logger.exception("Grade call failed for PMID %s", case.pmid)
        return None
    parsed = parse_last_json_object(response)
    if not isinstance(parsed, dict):
        logger.error("PMID %s returned no JSON object: %s", case.pmid, response)
        return None
    return DesignJudgment.model_validate(parsed)


def _wrong_fields(case: DesignCase, judgment: DesignJudgment | None) -> list[str]:
    if judgment is None:
        return list(_FIELDS)
    return [
        field
        for field in _FIELDS
        if getattr(judgment, field) is not getattr(case, field)
    ]


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    if runs < 1:
        raise ValueError("runs must be at least 1")
    prompt_path = Path(sys.argv[2]) if len(sys.argv) > 2 else _PRODUCTION_PROMPT
    prompt_template = prompt_path.read_text()
    logger.info("Prompt: %s", prompt_path)

    abstracts = _load_abstracts()
    semaphore = asyncio.Semaphore(8)

    async def bounded(case: DesignCase) -> DesignJudgment | None:
        async with semaphore:
            return await _grade_once(prompt_template, case, abstracts[case.pmid])

    all_ok = True
    results = await asyncio.gather(
        *(bounded(case) for case in _CASES for _ in range(runs))
    )
    for index, case in enumerate(_CASES):
        judgments = results[index * runs : (index + 1) * runs]
        wrong = [_wrong_fields(case, judgment) for judgment in judgments]
        passed = sum(not fields for fields in wrong)
        all_ok = all_ok and passed == runs
        missed = sorted({field for fields in wrong for field in fields})
        logger.info(
            "%s %s x %s (%s): %d/%d%s",
            case.pmid,
            case.drug,
            case.disease,
            case.note,
            passed,
            runs,
            f"  wrong: {', '.join(missed)}" if missed else "",
        )
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
