"""Harness for the treats-disease gate prompt: does the paper test the drug against this disease (treatment or prevention)?

Runs the production prompt by default, or a candidate prompt file given as the second argument, so a simplified prompt
can be compared case by case. Cases are papers with settled answers from earlier gate work, plus the semaglutide SUSTAIN
11 trial (35546450): a type 2 diabetes trial that recorded severe hypoglycaemia as a safety endpoint, which the gate
admitted for semaglutide x hypoglycemia and which then closed that signal on 2026-11-11. The semaglutide reject cases
after it are papers the shortened gate of 84e3333 admitted on 2026-11-16: adverse-event studies read as prevention, a
result for a different disease, an abstract with no result, and weight trials in a schizophrenia population. The type 1
diabetes case guards the population rule against rejecting a trial whose outcome measures the candidate disease.
Examples written into the prompt are kept different from these cases. Abstracts come from pgvector.
Run with:

    .venv/bin/python tests/harness_tests/pmid_treats_disease_harness.py [runs] [prompt_path]
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
    / "pmid_treats_disease.txt"
)


class GateCase(BaseModel):
    pmid: str
    drug: str
    disease: str
    treats: bool
    note: str


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


_CASES: tuple[GateCase, ...] = (
    GateCase(
        pmid="35546450",
        drug="semaglutide",
        disease="hypoglycemia",
        treats=False,
        note="T2D trial, severe hypoglycaemia only a safety endpoint",
    ),
    GateCase(
        pmid="12411660",
        drug="sildenafil",
        disease="ischemic stroke",
        treats=True,
        note="rat stroke recovery",
    ),
    GateCase(
        pmid="28343223",
        drug="sildenafil",
        disease="ischemic stroke",
        treats=False,
        note="neonatal hypoxic-ischemic brain injury",
    ),
    GateCase(
        pmid="29092891",
        drug="sildenafil",
        disease="ischemic stroke",
        treats=True,
        note="LVAD patients, stroke incidence lower on sildenafil",
    ),
    GateCase(
        pmid="19837434",
        drug="sildenafil",
        disease="endothelial dysfunction",
        treats=False,
        note="penile endothelial function in an ED trial",
    ),
    GateCase(
        pmid="21680809",
        drug="sildenafil",
        disease="endothelial dysfunction",
        treats=False,
        note="endothelial function as a marker in T2D men",
    ),
    GateCase(
        pmid="18036451",
        drug="sildenafil",
        disease="endothelial dysfunction",
        treats=False,
        note="heart failure trial measuring endothelial function",
    ),
    GateCase(
        pmid="34551918",
        drug="metformin",
        disease="preeclampsia",
        treats=True,
        note="RCT treating preterm pre-eclampsia",
    ),
    GateCase(
        pmid="39236318",
        drug="metformin",
        disease="preeclampsia",
        treats=True,
        note="prevention: preeclampsia incidence in diabetic pregnancy",
    ),
    GateCase(
        pmid="25467617",
        drug="metformin",
        disease="preeclampsia",
        treats=True,
        note="prevention: GDM trial reporting preeclampsia rates",
    ),
    GateCase(
        pmid="42627515",
        drug="metformin",
        disease="Parkinson disease",
        treats=False,
        note="Mendelian randomization, nobody given the drug",
    ),
    GateCase(
        pmid="42216231",
        drug="metformin",
        disease="Parkinson disease",
        treats=True,
        note="prevention: parkinsonism incidence in a cohort",
    ),
    GateCase(
        pmid="40385486",
        drug="metformin",
        disease="Parkinson disease",
        treats=True,
        note="RCT treating Parkinson's",
    ),
    GateCase(
        pmid="40810985",
        drug="semaglutide",
        disease="eye disorder",
        treats=False,
        note="meta-analysis of ocular adverse events",
    ),
    GateCase(
        pmid="39151204",
        drug="semaglutide",
        disease="eye disorder",
        treats=False,
        note="review of diabetes drugs worsening retinopathy",
    ),
    GateCase(
        pmid="36596264",
        drug="semaglutide",
        disease="eye disorder",
        treats=False,
        note="GLP-1RA trials and retinopathy worsening",
    ),
    GateCase(
        pmid="41192935",
        drug="semaglutide",
        disease="eye disorder",
        treats=False,
        note="cohort asking whether semaglutide raises retinopathy risk",
    ),
    GateCase(
        pmid="38806481",
        drug="semaglutide",
        disease="nicotine dependence",
        treats=False,
        note="alcohol use disorder incidence, a different disease",
    ),
    GateCase(
        pmid="36502289",
        drug="semaglutide",
        disease="heart failure",
        treats=False,
        note="SELECT baseline characteristics, no outcome",
    ),
    GateCase(
        pmid="42418179",
        drug="semaglutide",
        disease="schizophrenia",
        treats=False,
        note="antipsychotic-induced weight gain in schizophrenia patients",
    ),
    GateCase(
        pmid="42179170",
        drug="semaglutide",
        disease="schizophrenia",
        treats=False,
        note="weight and glycaemia in schizophrenia patients",
    ),
    GateCase(
        pmid="38444317",
        drug="semaglutide",
        disease="type 1 diabetes mellitus",
        treats=True,
        note="adjunct semaglutide for glycaemic control in T1D",
    ),
)


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


async def _judge_once(
    prompt_template: str, case: GateCase, item: HarnessAbstract
) -> bool | None:
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
        logger.exception("Gate call failed for PMID %s", case.pmid)
        return None
    parsed = parse_last_json_object(response)
    if not isinstance(parsed, dict):
        logger.error("PMID %s returned no JSON object: %s", case.pmid, response)
        return None
    verdict = str(parsed.get("verdict", "")).strip().lower()
    if verdict not in ("treats", "not_treats"):
        logger.error("PMID %s returned verdict %r", case.pmid, verdict)
        return None
    return verdict == "treats"


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

    async def bounded(case: GateCase) -> bool | None:
        async with semaphore:
            return await _judge_once(prompt_template, case, abstracts[case.pmid])

    results = await asyncio.gather(
        *(bounded(case) for case in _CASES for _ in range(runs))
    )
    all_ok = True
    for index, case in enumerate(_CASES):
        answers = results[index * runs : (index + 1) * runs]
        passed = sum(answer is case.treats for answer in answers)
        all_ok = all_ok and passed == runs
        logger.info(
            "%s %s x %s (%s): expected %s, %d/%d",
            case.pmid,
            case.drug,
            case.disease,
            case.note,
            "treats" if case.treats else "not_treats",
            passed,
            runs,
        )
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
