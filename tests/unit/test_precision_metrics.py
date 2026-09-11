import csv
from datetime import date

import pytest

from indication_scout.agents.supervisor.supervisor_output import SupervisorOutput
from indication_scout.services.precision_metrics import (
    CandidatePrecisionSpec,
    CandidatePrediction,
    CandidateReview,
    CandidateValidityLabel,
    load_reviews,
    score_ranked_candidate_precision,
    score_reviews,
    stable_review_id,
    write_review_template,
)


def _prediction(drug: str, disease: str, position: int) -> CandidatePrediction:
    cutoff = date(2020, 1, 1)
    return CandidatePrediction(
        review_id=stable_review_id(drug, cutoff, disease),
        drug=drug,
        cutoff=cutoff,
        position=position,
        source="competitor",
        disease=disease,
        investigation_limit=16,
        supervisor_candidate_cap=60,
        mechanism_associations_per_target=30,
        mechanism_top_candidates=10,
    )


def _review(prediction: CandidatePrediction, decision: str) -> CandidateReview:
    return CandidateReview(
        review_id=prediction.review_id,
        decision=decision,
        reason_category="reviewed",
        rationale="Reviewed against the precision rubric.",
        evidence="PMID:1",
    )


def test_score_reviews_calculates_micro_macro_bounds_and_counts() -> None:
    predictions = [
        _prediction("drug-a", "disease-1", 1),
        _prediction("drug-a", "disease-2", 2),
        _prediction("drug-a", "disease-3", 3),
        _prediction("drug-b", "disease-4", 1),
    ]
    reviews = [
        _review(predictions[0], "valid"),
        _review(predictions[1], "invalid"),
        _review(predictions[2], "uncertain"),
        _review(predictions[3], "valid"),
    ]

    result = score_reviews(predictions, reviews)

    assert result.predictions == 4
    assert result.valid == 2
    assert result.invalid == 1
    assert result.uncertain == 1
    assert result.reviewed_precision == pytest.approx(2 / 3)
    assert result.reviewed_precision_ci_low == pytest.approx(0.2076596008)
    assert result.reviewed_precision_ci_high == pytest.approx(0.9385080553)
    assert result.lower_bound == 0.5
    assert result.upper_bound == 0.75
    assert result.macro_precision == 0.75
    assert [item.model_dump() for item in result.per_drug] == [
        {
            "drug": "drug-a",
            "valid": 1,
            "invalid": 1,
            "uncertain": 1,
            "reviewed_precision": 0.5,
            "lower_bound": 1 / 3,
            "upper_bound": 2 / 3,
        },
        {
            "drug": "drug-b",
            "valid": 1,
            "invalid": 0,
            "uncertain": 0,
            "reviewed_precision": 1.0,
            "lower_bound": 1.0,
            "upper_bound": 1.0,
        },
    ]


def test_score_reviews_rejects_incomplete_or_unknown_review_ids() -> None:
    prediction = _prediction("drug-a", "disease-1", 1)
    unknown = _prediction("drug-b", "disease-2", 1)

    with pytest.raises(
        ValueError,
        match=f"missing reviews: {prediction.review_id}; unknown review ids: {unknown.review_id}",
    ):
        score_reviews([prediction], [_review(unknown, "valid")])


def test_write_review_template_preserves_all_candidate_context(tmp_path) -> None:
    prediction = _prediction("drug-a", "disease-1", 1)
    path = tmp_path / "reviews.csv"

    write_review_template([prediction], path)

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            "review_id": prediction.review_id,
            "drug": "drug-a",
            "cutoff": "2020-01-01",
            "position": "1",
            "source": "competitor",
            "disease": "disease-1",
            "decision": "",
            "reason_category": "",
            "rationale": "",
            "evidence": "",
        }
    ]


def test_load_reviews_accepts_completed_generated_template(tmp_path) -> None:
    prediction = _prediction("drug-a", "disease-1", 1)
    path = tmp_path / "reviews.csv"
    write_review_template([prediction], path)
    text = path.read_text(encoding="utf-8")
    path.write_text(
        text.replace(
            ",,,,\n",
            ",valid,reviewed,Reviewed against the precision rubric.,PMID:1\n",
        ),
        encoding="utf-8",
    )

    reviews = load_reviews(path)

    assert reviews == [_review(prediction, "valid")]


def test_score_ranked_candidate_precision_uses_candidate_labels_and_aliases() -> None:
    spec = CandidatePrecisionSpec(
        top_k=2,
        minimum_precision=0.75,
        labels=[
            CandidateValidityLabel(
                drug="drug-a",
                disease="disease one",
                aliases=["disease 1"],
                decision="valid",
                rationale="Reviewed as a valid candidate.",
            ),
            CandidateValidityLabel(
                drug="drug-a",
                disease="disease two",
                aliases=[],
                decision="invalid",
                rationale="Reviewed as an invalid candidate.",
            ),
            CandidateValidityLabel(
                drug="drug-b",
                disease="disease three",
                aliases=[],
                decision="valid",
                rationale="Reviewed as a valid candidate.",
            ),
            CandidateValidityLabel(
                drug="drug-b",
                disease="disease four",
                aliases=[],
                decision="valid",
                rationale="Reviewed as a valid candidate.",
            ),
        ],
    )
    reports = [
        SupervisorOutput(drug_name="drug-a", top_diseases=["Disease 1", "disease two"]),
        SupervisorOutput(
            drug_name="drug-b",
            top_diseases=["disease three", "disease four"],
        ),
    ]

    result = score_ranked_candidate_precision(reports, spec)

    assert result.top_k == 2
    assert result.minimum_precision == 0.75
    assert result.valid == 3
    assert result.invalid == 1
    assert result.precision == 0.75
    assert result.precision_ci_low == pytest.approx(0.3006418426)
    assert result.precision_ci_high == pytest.approx(0.9544127392)
    assert [item.model_dump() for item in result.per_drug] == [
        {
            "drug": "drug-a",
            "valid": 1,
            "invalid": 1,
            "precision": 0.5,
        },
        {
            "drug": "drug-b",
            "valid": 2,
            "invalid": 0,
            "precision": 1.0,
        },
    ]


def test_ranked_precision_requires_review_for_unknown_candidate() -> None:
    spec = CandidatePrecisionSpec(
        top_k=1,
        minimum_precision=0.9,
        labels=[
            CandidateValidityLabel(
                drug="drug-a",
                disease="reviewed disease",
                aliases=[],
                decision="valid",
                rationale="Reviewed as a valid candidate.",
            )
        ],
    )
    report = SupervisorOutput(drug_name="drug-a", top_diseases=["unreviewed disease"])

    with pytest.raises(
        ValueError,
        match=(
            "review required for unlabeled ranked candidates: "
            "drug-a: unreviewed disease"
        ),
    ):
        score_ranked_candidate_precision([report], spec)
