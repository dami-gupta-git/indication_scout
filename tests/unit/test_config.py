"""Unit tests for indication_scout.config."""

import pytest

from indication_scout.config import Settings


@pytest.mark.parametrize(
    "raw_url, expected_url",
    [
        (
            "postgresql://scout:pw@localhost:5432/scout",
            "postgresql+psycopg2://scout:pw@localhost:5432/scout",
        ),
        (
            "postgresql+psycopg2://scout:pw@localhost:5432/scout",
            "postgresql+psycopg2://scout:pw@localhost:5432/scout",
        ),
        (
            "postgresql+psycopg://scout:pw@localhost:5432/scout",
            "postgresql+psycopg://scout:pw@localhost:5432/scout",
        ),
    ],
)
def test_database_url_pins_psycopg2_only_when_driver_unspecified(
    raw_url: str, expected_url: str
) -> None:
    settings = Settings(database_url=raw_url, db_password="pw")
    assert settings.database_url == expected_url


def test_test_database_url_none_is_left_alone() -> None:
    settings = Settings(
        database_url="postgresql://scout:pw@localhost:5432/scout",
        db_password="pw",
        test_database_url=None,
    )
    assert settings.test_database_url is None
