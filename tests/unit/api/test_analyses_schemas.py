"""Unit tests for api/schemas/analyses — SupervisorOutput round-trips through the response."""

from indication_scout.agents.supervisor.supervisor_output import (
    CandidateFindings,
    SupervisorOutput,
)
from indication_scout.api.schemas.analyses import AnalysisStatusResponse


def _sample_output() -> SupervisorOutput:
    return SupervisorOutput(
        drug_name="metformin",
        candidate_diseases=["breast cancer", "alzheimer disease"],
        disease_findings=[CandidateFindings(disease="breast cancer")],
        top_diseases=["breast cancer"],
        summary="Promising in oncology.",
    )


def test_status_response_serializes_supervisor_output():
    output = _sample_output()
    resp = AnalysisStatusResponse(
        job_id="abc", drug_name="metformin", status="done", result=output
    )

    dumped = resp.model_dump()
    assert dumped["job_id"] == "abc"
    assert dumped["status"] == "done"
    assert dumped["error"] is None
    assert dumped["result"]["drug_name"] == "metformin"
    assert dumped["result"]["candidate_diseases"] == ["breast cancer", "alzheimer disease"]
    assert dumped["result"]["top_diseases"] == ["breast cancer"]
    assert dumped["result"]["summary"] == "Promising in oncology."


def test_status_response_round_trips_through_json():
    resp = AnalysisStatusResponse(
        job_id="abc", drug_name="metformin", status="done", result=_sample_output()
    )

    restored = AnalysisStatusResponse.model_validate_json(resp.model_dump_json())

    assert restored == resp
    assert restored.result.candidate_diseases == ["breast cancer", "alzheimer disease"]


def test_status_response_pending_has_no_result_or_error():
    resp = AnalysisStatusResponse(job_id="abc", drug_name="metformin", status="pending")

    assert resp.result is None
    assert resp.error is None
