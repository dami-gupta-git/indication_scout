"""Shared fixtures for integration tests."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from indication_scout.config import get_settings
from indication_scout.constants import TEST_CACHE_DIR
from indication_scout.data_sources.chembl import ChEMBLClient
from indication_scout.data_sources.clinical_trials import ClinicalTrialsClient
from indication_scout.data_sources.open_targets import OpenTargetsClient
from indication_scout.data_sources.pubmed import PubMedClient


@pytest.fixture()
def db_session():
    """Provide a real SQLAlchemy Session connected to the test DB (scout_test).

    Uses TEST_DATABASE_URL from settings — a separate database on the same
    Postgres instance as the main DB, so integration test data never
    contaminates scout. Each test gets a savepoint that is rolled back on
    teardown, so rows inserted during one test don't bleed into the next.

    Prerequisites:
      - Docker container running (docker compose up db)
      - scout_test DB migrated: TEST_DATABASE_URL=... alembic upgrade head
    """
    settings = get_settings()
    if settings.test_database_url is None:
        pytest.skip("TEST_DATABASE_URL not set — skipping DB integration test")

    engine = create_engine(settings.test_database_url)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    session.begin_nested()  # savepoint — rolled back after each test
    yield session
    session.rollback()
    session.close()


@pytest.fixture
async def chembl_client():
    """Create and tear down a ChEMBLClient."""
    c = ChEMBLClient()
    yield c
    await c.close()


@pytest.fixture
async def open_targets_client():
    """Create and tear down an OpenTargetsClient using the test cache."""
    c = OpenTargetsClient(cache_dir=TEST_CACHE_DIR)
    yield c
    await c.close()


@pytest.fixture
async def pubmed_client():
    """Create and tear down a PubMedClient."""
    c = PubMedClient()
    yield c
    await c.close()


@pytest.fixture
async def clinical_trials_client():
    """Create and tear down a ClinicalTrialsClient."""
    c = ClinicalTrialsClient()
    yield c
    await c.close()
