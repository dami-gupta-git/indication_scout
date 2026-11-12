"""Integration test for the treats-disease gate — the LIVE LLM that decides whether a paper studies the candidate disease
as a treatment or prevention target. Hits real Anthropic; abstracts are real PubMed text.

A Mendelian randomization study gives nobody the drug: it uses genetic proxies for drug exposure, so it is not prevention
evidence. It passed the gate for metformin x Parkinson disease and was counted against metformin. The retrospective
cohort comparing metformin initiators with DPP-4 inhibitor users on incident parkinsonism is the control: a prevention
outcome in people who took the drug still passes, and the pilot trial of adjunctive metformin is direct treatment.
"""

import logging
from pathlib import Path

import pytest

from indication_scout.services.retrieval import (
    AbstractResult,
    _judge_pmid_treats_disease,
)

from ._metformin_parkinson_abstracts import DPP4_COHORT, MENDELIAN, PILOT_RCT

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.asyncio(loop_scope="module")


async def test_pmid_treats_disease_genetic_proxy(tmp_path: Path) -> None:
    abstracts = [
        AbstractResult(
            pmid=d["pmid"], title=d["title"], abstract=d["abstract"], similarity=0.9
        )
        for d in (MENDELIAN, DPP4_COHORT, PILOT_RCT)
    ]

    decisions = await _judge_pmid_treats_disease(
        "CHEMBL_TEST", "metformin", "Parkinson disease", abstracts, tmp_path
    )

    assert decisions == {"42627515": False, "42216231": True, "40385486": True}
