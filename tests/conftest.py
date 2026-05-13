"""Pytest configuration and fixtures."""

import os

# Point Settings at the test constants file before indication_scout.config is
# imported anywhere. get_settings() is @lru_cache'd, so the first import freezes
# whichever file is active at that moment. setdefault preserves an explicit
# CONSTANTS_FILE=... override on the command line.
os.environ.setdefault("CONSTANTS_FILE", ".env.constants.test")

import pytest


@pytest.fixture
def sample_drug() -> dict:
    """Sample drug data for testing."""
    return {
        "id": "DB00316",
        "name": "Acetaminophen",
        "synonyms": ["Tylenol", "Paracetamol"],
    }


@pytest.fixture
def sample_indication() -> dict:
    """Sample indication data for testing."""
    return {
        "id": "C0030193",
        "name": "Pain",
        "synonyms": ["Chronic Pain", "Acute Pain"],
    }


@pytest.fixture
def sample_disease() -> dict:
    """Sample disease data for testing."""
    return {
        "id": "EFO_0003885",
        "name": "Multiple Sclerosis",
    }
