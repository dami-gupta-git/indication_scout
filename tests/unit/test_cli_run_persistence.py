"""Unit tests for CLI analysis-run persistence."""

from contextlib import nullcontext
from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from indication_scout.agents.supervisor.supervisor_output import SupervisorOutput
from indication_scout.cli.cli import _run_for_drug, _run_for_pair


class FakeSessionFactory:
    """Minimal sessionmaker stand-in with an observable engine."""

    def __init__(self) -> None:
        self.engine = MagicMock()
        self.kw = {"bind": self.engine}

    def __call__(self):
        return nullcontext(object())


class OneReadIdentity:
    """Simulate an ORM identity that expires after its session transaction."""

    def __init__(self, attribute: str, value: str) -> None:
        self.attribute = attribute
        self.value = value
        self.reads = 0

    def __getattr__(self, attribute: str) -> str:
        if attribute != self.attribute:
            raise AttributeError(attribute)
        self.reads += 1
        if self.reads > 1:
            raise RuntimeError("detached ORM identity was accessed again")
        return self.value


async def test_find_cli_persists_run_attempt_progress_and_result(tmp_path):
    output = SupervisorOutput(drug_name="metformin", summary="Stored report.")
    repository = MagicMock()
    repository.create_run.return_value = OneReadIdentity("run_id", "a" * 32)
    repository.start_attempt.return_value = OneReadIdentity("attempt_id", "b" * 32)
    session_factory = FakeSessionFactory()

    async def run_analysis(drug_name, *, date_before):
        from indication_scout.services.progress import emit_progress

        emit_progress("literature", "Found 12 papers")
        return output, "# Report"

    with (
        patch(
            "indication_scout.db.session.make_session_factory",
            return_value=session_factory,
        ),
        patch(
            "indication_scout.services.run_repository.AnalysisRunRepository",
            return_value=repository,
        ),
        patch(
            "indication_scout.services.analysis_runner.run_analysis",
            side_effect=run_analysis,
        ),
        patch("indication_scout.tracing.setup_tracing"),
        patch("indication_scout.tracing.shutdown_tracing"),
        patch("indication_scout.cli.cli.TEST_REPORTS_DIR", tmp_path),
    ):
        await _run_for_drug(
            "Metformin",
            tmp_path,
            write=False,
            date_before=date(2025, 1, 1),
        )

    repository.create_run.assert_called_once_with(
        "metformin",
        "live",
        submission_source="cli",
        analysis_kind="find",
        disease_name=None,
        date_before=date(2025, 1, 1),
    )
    repository.start_attempt.assert_called_once()
    repository.append_event.assert_called_once_with(
        "a" * 32,
        "analysis.progress",
        "info",
        attempt_id="b" * 32,
        stage="literature",
        attributes={"message": "Found 12 papers"},
    )
    repository.complete_validated_attempt.assert_called_once_with(
        "a" * 32, "b" * 32, output
    )
    repository.fail_attempt.assert_not_called()
    repository.cancel_attempt.assert_not_called()
    session_factory.engine.dispose.assert_called_once_with()


async def test_investigate_cli_persists_fixed_disease_run(tmp_path):
    output = SupervisorOutput(drug_name="sildenafil", summary="Stored report.")
    repository = MagicMock()
    repository.create_run.return_value = SimpleNamespace(run_id="c" * 32)
    repository.start_attempt.return_value = SimpleNamespace(attempt_id="d" * 32)
    session_factory = FakeSessionFactory()

    with (
        patch(
            "indication_scout.db.session.make_session_factory",
            return_value=session_factory,
        ),
        patch(
            "indication_scout.services.run_repository.AnalysisRunRepository",
            return_value=repository,
        ),
        patch(
            "indication_scout.services.analysis_runner.run_pair_analysis",
            return_value=(output, "# Report"),
        ),
        patch("indication_scout.tracing.setup_tracing"),
        patch("indication_scout.tracing.shutdown_tracing"),
    ):
        await _run_for_pair(
            "Sildenafil",
            " Raynaud disease ",
            tmp_path,
            write=False,
        )

    repository.create_run.assert_called_once_with(
        "sildenafil",
        "live",
        submission_source="cli",
        analysis_kind="investigate",
        disease_name="Raynaud disease",
        date_before=None,
    )
    repository.start_attempt.assert_called_once()
    repository.complete_validated_attempt.assert_called_once_with(
        "c" * 32, "d" * 32, output
    )
    repository.fail_attempt.assert_not_called()
    repository.cancel_attempt.assert_not_called()
    session_factory.engine.dispose.assert_called_once_with()


async def test_find_cli_persists_pipeline_failure(tmp_path):
    repository = MagicMock()
    repository.create_run.return_value = SimpleNamespace(run_id="e" * 32)
    repository.start_attempt.return_value = SimpleNamespace(attempt_id="f" * 32)
    session_factory = FakeSessionFactory()

    with (
        patch(
            "indication_scout.db.session.make_session_factory",
            return_value=session_factory,
        ),
        patch(
            "indication_scout.services.run_repository.AnalysisRunRepository",
            return_value=repository,
        ),
        patch(
            "indication_scout.services.analysis_runner.run_analysis",
            side_effect=RuntimeError("pipeline failed"),
        ),
        patch("indication_scout.tracing.setup_tracing"),
        patch("indication_scout.tracing.shutdown_tracing"),
    ):
        with pytest.raises(RuntimeError, match="pipeline failed"):
            await _run_for_drug("metformin", tmp_path, write=False)

    repository.fail_attempt.assert_called_once_with(
        "e" * 32,
        "f" * 32,
        error_code="RuntimeError",
        error_message="RuntimeError: pipeline failed",
        retryable=False,
        integrity_failed=False,
    )
    repository.complete_validated_attempt.assert_not_called()
    repository.cancel_attempt.assert_not_called()
    session_factory.engine.dispose.assert_called_once_with()
