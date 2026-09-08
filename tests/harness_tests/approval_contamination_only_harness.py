"""Exercise a proposed contaminated-versus-none approval rule.

Run: .venv/bin/python tests/harness_tests/approval_contamination_only_harness.py
"""

import asyncio
import json
import logging
from collections import Counter
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from indication_scout.services.llm import parse_last_json_object, query_llm

logger = logging.getLogger(__name__)

RUNS_PER_CASE = 5
LABELS = frozenset(("contaminated", "none"))


class ContaminatedDecision(BaseModel):
    """A contamination decision anchored to one supplied approved indication."""

    model_config = ConfigDict(extra="forbid")

    label: Literal["contaminated"]
    matched_approved_indication: str
    reason: str

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict[str, Any]) -> dict[str, Any]:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


class NoneDecision(BaseModel):
    """A decision with no relationship to a supplied approved indication."""

    model_config = ConfigDict(extra="forbid")

    label: Literal["none"]
    reason: str

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict[str, Any]) -> dict[str, Any]:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


CandidateDecision = Annotated[
    ContaminatedDecision | NoneDecision, Field(discriminator="label")
]
DECISION_ADAPTER = TypeAdapter(CandidateDecision)


ISOLATED_PROMPT = """You are a clinical pharmacology expert. Given the FDA-approved indications for one drug and a list of candidate diseases,
classify each candidate independently as exactly "contaminated" or "none".

"contaminated" means the candidate is a clinically recognized broader disease category that contains at least one supplied approved indication plus other distinct diseases.
"none" means this broader-parent relationship does not hold. Distinct sibling diseases, related diseases, shared mechanisms, shared drug classes, common comorbidities, and shared words are "none".

Do not predict what a registry or literature search might retrieve. Use only the clinical parent-child relationship between the candidate and the supplied approved indications.

Return only a JSON object mapping every candidate verbatim to one of these two object shapes:
- For "contaminated": include "label", "matched_approved_indication", and "reason". The matched indication must exactly equal one item from the supplied approved-indication list.
- For "none": include only "label" and "reason". Do not include "matched_approved_indication" because no match exists.
The reason must be one sentence explaining the relationship.
The "label" field value must be exactly "contaminated" or "none". Do not put the candidate name in the "label" field. Do not add a "classification", "label_result", or other field.

Required format example:
{{
  "candidate parent": {{"label": "contaminated", "matched_approved_indication": "approved child", "reason": "The candidate is the broader parent of the approved child."}},
  "separate candidate": {{"label": "none", "reason": "The candidate is not a broader parent of any supplied approved indication."}}
}}

FDA-approved indications:
{approved_indications}

Candidate diseases:
{candidate_diseases}
"""


CASES: list[
    tuple[str, list[str], dict[str, tuple[str, frozenset[str]]]]
] = [
    (
        "certolizumab pegol",
        [
            "Crohn disease",
            "rheumatoid arthritis",
            "psoriatic arthritis",
            "ankylosing spondylitis",
            "non-radiographic axial spondyloarthritis",
            "plaque psoriasis",
        ],
        {
            "inflammatory bowel disease": (
                "contaminated",
                frozenset(("Crohn disease",)),
            ),
            "ulcerative colitis": ("none", frozenset()),
            "celiac disease": ("none", frozenset()),
            "multiple sclerosis": ("none", frozenset()),
        },
    ),
    (
        "baricitinib",
        ["rheumatoid arthritis", "COVID-19", "severe alopecia areata"],
        {
            "arthritis": ("contaminated", frozenset(("rheumatoid arthritis",))),
            "inflammatory bowel disease": ("none", frozenset()),
            "ulcerative colitis": ("none", frozenset()),
            "Crohn disease": ("none", frozenset()),
        },
    ),
    (
        "secukinumab",
        [
            "plaque psoriasis",
            "psoriatic arthritis",
            "ankylosing spondylitis",
            "non-radiographic axial spondyloarthritis",
            "hidradenitis suppurativa",
        ],
        {
            "arthritis": ("contaminated", frozenset(("psoriatic arthritis",))),
            "inflammatory bowel disease": ("none", frozenset()),
            "ulcerative colitis": ("none", frozenset()),
            "Crohn disease": ("none", frozenset()),
        },
    ),
    (
        "rituximab",
        [
            "non-Hodgkin lymphoma",
            "chronic lymphocytic leukemia",
            "rheumatoid arthritis",
            "granulomatosis with polyangiitis",
            "microscopic polyangiitis",
        ],
        {
            "vasculitis": (
                "contaminated",
                frozenset(
                    (
                        "granulomatosis with polyangiitis",
                        "microscopic polyangiitis",
                    )
                ),
            ),
            "inflammatory bowel disease": ("none", frozenset()),
            "multiple sclerosis": ("none", frozenset()),
            "systemic lupus erythematosus": ("none", frozenset()),
        },
    ),
    (
        "vemurafenib",
        ["BRAF V600E-mutated melanoma"],
        {
            "melanoma": (
                "contaminated",
                frozenset(("BRAF V600E-mutated melanoma",)),
            ),
            "uveal melanoma": ("none", frozenset()),
            "basal cell carcinoma": ("none", frozenset()),
            "acute myeloid leukemia": ("none", frozenset()),
        },
    ),
    (
        "sildenafil",
        ["pulmonary arterial hypertension", "erectile dysfunction"],
        {
            # The clinical relationship is none: systemic hypertension is not a
            # parent of PAH. Production handles the verified CT.gov MeSH-ancestor
            # collision through the curated contamination table.
            "hypertension": ("none", frozenset()),
            "essential hypertension": ("none", frozenset()),
            "coronary artery disease": ("none", frozenset()),
            "multiple sclerosis": ("none", frozenset()),
        },
    ),
]


async def classify(
    approved_indications: list[str], candidates: list[str]
) -> dict[str, CandidateDecision]:
    """Send one isolated contamination classification to the configured model."""
    prompt = ISOLATED_PROMPT.format(
        approved_indications=json.dumps(approved_indications),
        candidate_diseases=json.dumps(candidates),
    )
    parsed = parse_last_json_object(await query_llm(prompt))
    if parsed is None:
        return {}
    decisions: dict[str, CandidateDecision] = {}
    for key, value in parsed.items():
        if not isinstance(key, str):
            continue
        try:
            decisions[key] = DECISION_ADAPTER.validate_python(value)
        except ValidationError as exc:
            logger.error("Invalid decision for %s: %s", key, exc)
    return decisions


async def main() -> int:
    """Run all cases and return nonzero when any repetition is incorrect."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    failures = 0

    logger.info(
        "Contamination-only prompt: %d cases, %d repetitions per case",
        len(CASES),
        RUNS_PER_CASE,
    )
    for drug, approved_indications, expected in CASES:
        candidates = list(expected)
        runs = await asyncio.gather(
            *(
                classify(approved_indications, candidates)
                for _ in range(RUNS_PER_CASE)
            )
        )
        for candidate, (correct_label, correct_anchors) in expected.items():
            decisions = [result.get(candidate) for result in runs]
            votes = Counter(
                decision.label if decision is not None else "MISSING"
                for decision in decisions
            )
            anchors = Counter(
                decision.matched_approved_indication
                if isinstance(decision, ContaminatedDecision)
                else None
                if isinstance(decision, NoneDecision)
                else "MISSING"
                for decision in decisions
            )
            reasons = Counter(
                decision.reason if decision is not None else "MISSING"
                for decision in decisions
            )
            valid = all(label in LABELS for label in votes) and all(
                decision is not None and decision.reason.strip()
                for decision in decisions
            )
            passed = valid and all(
                decision is not None
                and decision.label == correct_label
                and (
                    decision.matched_approved_indication in correct_anchors
                    if isinstance(decision, ContaminatedDecision)
                    else not correct_anchors
                )
                for decision in decisions
            )
            if not passed:
                failures += 1
            vote_text = ", ".join(
                f"{label}={count}" for label, count in votes.most_common()
            )
            logger.info(
                "%s | %s | expected=%s anchors=%s | labels: %s | anchors: %s | %s",
                drug,
                candidate,
                correct_label,
                sorted(correct_anchors),
                vote_text,
                ", ".join(
                    f"{anchor}={count}" for anchor, count in anchors.most_common()
                ),
                "PASS" if passed else "FAIL",
            )
            for reason, count in reasons.most_common():
                logger.info("  reason (%d/%d): %s", count, RUNS_PER_CASE, reason)

    total_candidates = sum(len(expected) for _, _, expected in CASES)
    logger.info("Result: %d/%d candidate rows failed", failures, total_candidates)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
