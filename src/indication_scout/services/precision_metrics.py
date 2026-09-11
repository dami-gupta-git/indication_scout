"""Contracts and calculations for expert-reviewed candidate precision."""

from __future__ import annotations

import csv
import hashlib
import math
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ReviewDecision = Literal["valid", "invalid", "uncertain"]
CandidateSource = Literal["competitor", "mechanism", "both"]

_REVIEW_CONTEXT_FIELDS = {"drug", "cutoff", "position", "source", "disease"}
_REVIEW_FIELDS = {"review_id", "decision", "reason_category", "rationale", "evidence"}
_REVIEW_TEMPLATE_FIELDS = _REVIEW_CONTEXT_FIELDS | _REVIEW_FIELDS


class CandidatePrediction(BaseModel):
    """One candidate that was eligible to enter the investigation fan-out."""

    model_config = ConfigDict(extra="forbid")

    review_id: str
    drug: str
    cutoff: date
    position: int = Field(gt=0)
    source: CandidateSource
    disease: str
    investigation_limit: int = Field(gt=0)
    supervisor_candidate_cap: int = Field(gt=0)
    mechanism_associations_per_target: int = Field(gt=0)
    mechanism_top_candidates: int = Field(gt=0)

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


class CandidateReview(BaseModel):
    """One completed expert decision for a candidate prediction."""

    model_config = ConfigDict(extra="forbid")

    review_id: str
    decision: ReviewDecision
    reason_category: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    evidence: str = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


class DrugPrecision(BaseModel):
    """Precision counts and estimates for one drug."""

    model_config = ConfigDict(extra="forbid")

    drug: str
    valid: int = Field(ge=0)
    invalid: int = Field(ge=0)
    uncertain: int = Field(ge=0)
    reviewed_precision: float | None
    lower_bound: float
    upper_bound: float

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


class PrecisionSummary(BaseModel):
    """Micro, macro, and uncertainty-aware precision results."""

    model_config = ConfigDict(extra="forbid")

    predictions: int = Field(ge=0)
    valid: int = Field(ge=0)
    invalid: int = Field(ge=0)
    uncertain: int = Field(ge=0)
    reviewed_precision: float | None
    reviewed_precision_ci_low: float | None
    reviewed_precision_ci_high: float | None
    lower_bound: float | None
    upper_bound: float | None
    macro_precision: float | None
    per_drug: list[DrugPrecision]

    @model_validator(mode="before")
    @classmethod
    def coerce_nones(cls, values: dict) -> dict:
        for field_name, field_info in cls.model_fields.items():
            if values.get(field_name) is None and field_info.default is not None:
                values[field_name] = field_info.default
        return values


def stable_review_id(drug: str, cutoff: date | str, disease: str) -> str:
    """Return a stable identifier for one drug, cutoff, and candidate disease."""
    cutoff_text = cutoff.isoformat() if isinstance(cutoff, date) else cutoff
    normalized = "\x1f".join(
        (drug.lower().strip(), cutoff_text.strip(), disease.lower().strip())
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]


def load_predictions(path: Path) -> list[CandidatePrediction]:
    """Load and validate candidate predictions from CSV."""
    with path.open(newline="", encoding="utf-8") as handle:
        predictions = [
            CandidatePrediction.model_validate(row) for row in csv.DictReader(handle)
        ]
    _reject_duplicate_ids(
        [prediction.review_id for prediction in predictions], "prediction"
    )
    return predictions


def load_reviews(path: Path) -> list[CandidateReview]:
    """Load and validate completed expert reviews from CSV."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        actual_fields = set(reader.fieldnames or [])
        if actual_fields != _REVIEW_TEMPLATE_FIELDS:
            missing = sorted(_REVIEW_TEMPLATE_FIELDS - actual_fields)
            unexpected = sorted(actual_fields - _REVIEW_TEMPLATE_FIELDS)
            parts: list[str] = []
            if missing:
                parts.append(f"missing review columns: {', '.join(missing)}")
            if unexpected:
                parts.append(f"unknown review columns: {', '.join(unexpected)}")
            raise ValueError("; ".join(parts))
        reviews = [
            CandidateReview.model_validate(
                {field_name: row[field_name] for field_name in _REVIEW_FIELDS}
            )
            for row in reader
        ]
    _reject_duplicate_ids([review.review_id for review in reviews], "review")
    return reviews


def write_review_template(predictions: list[CandidatePrediction], path: Path) -> None:
    """Write a review worksheet containing candidate context and blank decision fields."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "review_id",
        "drug",
        "cutoff",
        "position",
        "source",
        "disease",
        "decision",
        "reason_category",
        "rationale",
        "evidence",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for prediction in predictions:
            writer.writerow(
                {
                    "review_id": prediction.review_id,
                    "drug": prediction.drug,
                    "cutoff": prediction.cutoff.isoformat(),
                    "position": prediction.position,
                    "source": prediction.source,
                    "disease": prediction.disease,
                    "decision": "",
                    "reason_category": "",
                    "rationale": "",
                    "evidence": "",
                }
            )


def score_reviews(
    predictions: list[CandidatePrediction], reviews: list[CandidateReview]
) -> PrecisionSummary:
    """Join completed reviews to predictions and calculate precision estimates."""
    prediction_ids = {prediction.review_id for prediction in predictions}
    reviews_by_id = {review.review_id: review for review in reviews}
    missing = sorted(prediction_ids - reviews_by_id.keys())
    unexpected = sorted(reviews_by_id.keys() - prediction_ids)
    if missing or unexpected:
        parts: list[str] = []
        if missing:
            parts.append(f"missing reviews: {', '.join(missing)}")
        if unexpected:
            parts.append(f"unknown review ids: {', '.join(unexpected)}")
        raise ValueError("; ".join(parts))

    grouped: dict[str, list[CandidateReview]] = defaultdict(list)
    for prediction in predictions:
        grouped[prediction.drug].append(reviews_by_id[prediction.review_id])

    per_drug = [
        _score_drug(drug, drug_reviews)
        for drug, drug_reviews in sorted(grouped.items())
    ]
    valid = sum(item.valid for item in per_drug)
    invalid = sum(item.invalid for item in per_drug)
    uncertain = sum(item.uncertain for item in per_drug)
    predictions_count = len(predictions)
    scored_count = valid + invalid
    reviewed_precision = valid / scored_count if scored_count else None
    ci_low, ci_high = _wilson_interval(valid, scored_count)
    drug_precisions = [
        item.reviewed_precision
        for item in per_drug
        if item.reviewed_precision is not None
    ]

    return PrecisionSummary(
        predictions=predictions_count,
        valid=valid,
        invalid=invalid,
        uncertain=uncertain,
        reviewed_precision=reviewed_precision,
        reviewed_precision_ci_low=ci_low,
        reviewed_precision_ci_high=ci_high,
        lower_bound=valid / predictions_count if predictions_count else None,
        upper_bound=(
            (valid + uncertain) / predictions_count if predictions_count else None
        ),
        macro_precision=(
            sum(drug_precisions) / len(drug_precisions) if drug_precisions else None
        ),
        per_drug=per_drug,
    )


def write_precision_report(summary: PrecisionSummary, path: Path) -> None:
    """Write a concise Markdown precision report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Candidate precision",
        "",
        "Precision is calculated from completed expert reviews. Uncertain candidates remain in the uncertainty bounds and "
        "are excluded only from the reviewed-point estimate.",
        "",
        "| Measurement | Result |",
        "|---|---:|",
        f"| Predictions reviewed | {summary.predictions} |",
        f"| Valid | {summary.valid} |",
        f"| Invalid | {summary.invalid} |",
        f"| Uncertain | {summary.uncertain} |",
        f"| Reviewed precision | {_format_percent(summary.reviewed_precision)} |",
        "| Reviewed precision, 95% Wilson interval | "
        f"{_format_interval(summary.reviewed_precision_ci_low, summary.reviewed_precision_ci_high)} |",
        f"| Precision lower bound | {_format_percent(summary.lower_bound)} |",
        f"| Precision upper bound | {_format_percent(summary.upper_bound)} |",
        f"| Macro precision across drugs | {_format_percent(summary.macro_precision)} |",
        "",
        "## By drug",
        "",
        "| Drug | Valid | Invalid | Uncertain | Precision | Lower bound | Upper bound |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary.per_drug:
        lines.append(
            f"| {item.drug} | {item.valid} | {item.invalid} | {item.uncertain} | "
            f"{_format_percent(item.reviewed_precision)} | {_format_percent(item.lower_bound)} | "
            f"{_format_percent(item.upper_bound)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _score_drug(drug: str, reviews: list[CandidateReview]) -> DrugPrecision:
    valid = sum(review.decision == "valid" for review in reviews)
    invalid = sum(review.decision == "invalid" for review in reviews)
    uncertain = sum(review.decision == "uncertain" for review in reviews)
    scored = valid + invalid
    total = len(reviews)
    return DrugPrecision(
        drug=drug,
        valid=valid,
        invalid=invalid,
        uncertain=uncertain,
        reviewed_precision=valid / scored if scored else None,
        lower_bound=valid / total,
        upper_bound=(valid + uncertain) / total,
    )


def _wilson_interval(successes: int, total: int) -> tuple[float | None, float | None]:
    if total == 0:
        return None, None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + (z * z / total)
    center = (proportion + (z * z / (2 * total))) / denominator
    spread = (
        z
        * math.sqrt(
            (proportion * (1 - proportion) / total) + (z * z / (4 * total * total))
        )
        / denominator
    )
    return center - spread, center + spread


def _reject_duplicate_ids(ids: list[str], label: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for item_id in ids:
        if item_id in seen:
            duplicates.add(item_id)
        seen.add(item_id)
    if duplicates:
        raise ValueError(f"duplicate {label} ids: {', '.join(sorted(duplicates))}")


def _format_percent(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.1%}"


def _format_interval(low: float | None, high: float | None) -> str:
    if low is None or high is None:
        return "unavailable"
    return f"{low:.1%} to {high:.1%}"
