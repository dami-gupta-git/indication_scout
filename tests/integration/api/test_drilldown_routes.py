"""Integration tests for drill-down routes — hit live ClinicalTrials/PubMed/OpenTargets APIs.

Expected values fetched live from each client (2026-06-06).
"""

import pytest
from fastapi.testclient import TestClient

from indication_scout.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_get_trial_live(client):
    resp = client.get("/api/trials/NCT00000620")
    assert resp.status_code == 200
    body = resp.json()
    assert body["nct_id"] == "NCT00000620"
    assert body["title"] == "Action to Control Cardiovascular Risk in Diabetes (ACCORD)"
    assert body["phase"] == "Phase 3"
    assert body["overall_status"] == "COMPLETED"


def test_get_pubmed_live(client):
    resp = client.get("/api/pubmed/16677156")
    assert resp.status_code == 200
    body = resp.json()
    assert body["pmid"] == "16677156"
    assert body["journal"] == "Journal of gastroenterology and hepatology"
    assert body["title"].startswith("Expression of Ki-67")


def test_get_target_live(client):
    resp = client.get("/api/targets/ENSG00000146648")
    assert resp.status_code == 200
    body = resp.json()
    assert body["target_id"] == "ENSG00000146648"
    assert body["symbol"] == "EGFR"
    assert body["name"] == "epidermal growth factor receptor"
