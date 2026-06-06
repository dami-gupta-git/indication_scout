"""Unit tests for api/routes/drilldown — clients are mocked; no network."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from indication_scout.api.main import app
from indication_scout.data_sources.base_client import DataSourceError
from indication_scout.models.model_clinical_trials import Trial
from indication_scout.models.model_open_targets import TargetData
from indication_scout.models.model_pubmed_abstract import PubmedAbstract


@pytest.fixture
def client():
    return TestClient(app)


def _ctx(mock_client: MagicMock):
    """Wrap a mock client as an async-context-manager (mimics `async with Client() as c:`)."""
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_client)
    cm.__aexit__ = AsyncMock(return_value=False)
    return MagicMock(return_value=cm)


def test_get_trial_returns_trial(client):
    mock = MagicMock()
    mock.get_trial = AsyncMock(return_value=Trial(nct_id="NCT12345678", title="A trial", phase="Phase 3"))

    with patch("indication_scout.api.routes.drilldown.ClinicalTrialsClient", _ctx(mock)):
        resp = client.get("/api/trials/NCT12345678")

    assert resp.status_code == 200
    assert resp.json()["nct_id"] == "NCT12345678"
    assert resp.json()["phase"] == "Phase 3"
    mock.get_trial.assert_awaited_once_with("NCT12345678")


def test_get_trial_missing_returns_404(client):
    mock = MagicMock()
    mock.get_trial = AsyncMock(side_effect=DataSourceError("ClinicalTrials", "not found"))

    with patch("indication_scout.api.routes.drilldown.ClinicalTrialsClient", _ctx(mock)):
        resp = client.get("/api/trials/NCT0")

    assert resp.status_code == 404


def test_get_pubmed_returns_abstract(client):
    mock = MagicMock()
    mock.fetch_abstracts = AsyncMock(
        return_value=[PubmedAbstract(pmid="123", title="Paper", journal="Nature")]
    )

    with patch("indication_scout.api.routes.drilldown.PubMedClient", _ctx(mock)):
        resp = client.get("/api/pubmed/123")

    assert resp.status_code == 200
    assert resp.json()["pmid"] == "123"
    assert resp.json()["journal"] == "Nature"
    mock.fetch_abstracts.assert_awaited_once_with(["123"])


def test_get_pubmed_empty_returns_404(client):
    mock = MagicMock()
    mock.fetch_abstracts = AsyncMock(return_value=[])

    with patch("indication_scout.api.routes.drilldown.PubMedClient", _ctx(mock)):
        resp = client.get("/api/pubmed/999")

    assert resp.status_code == 404


def test_get_target_returns_target(client):
    mock = MagicMock()
    mock.get_target_data = AsyncMock(
        return_value=TargetData(target_id="ENSG00000146648", symbol="EGFR", name="epidermal growth factor receptor")
    )

    with patch("indication_scout.api.routes.drilldown.OpenTargetsClient", _ctx(mock)):
        resp = client.get("/api/targets/ENSG00000146648")

    assert resp.status_code == 200
    assert resp.json()["symbol"] == "EGFR"
    assert resp.json()["target_id"] == "ENSG00000146648"
    mock.get_target_data.assert_awaited_once_with("ENSG00000146648")


def test_get_target_missing_returns_404(client):
    mock = MagicMock()
    mock.get_target_data = AsyncMock(side_effect=DataSourceError("OpenTargets", "no target"))

    with patch("indication_scout.api.routes.drilldown.OpenTargetsClient", _ctx(mock)):
        resp = client.get("/api/targets/ENSG0")

    assert resp.status_code == 404
