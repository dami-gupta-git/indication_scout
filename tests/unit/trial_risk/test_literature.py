"""Unit tests for trial_risk.literature (date helpers only — no DB calls)."""

from datetime import date

import pytest

from indication_scout.models.model_clinical_trials import Trial
from indication_scout.ml_models.trial_risk.literature import (
    cutoff_for_trial,
    parse_trial_date,
)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("2020-05-12", date(2020, 5, 12)),
        ("2018-03", date(2018, 3, 1)),
        ("", None),
        (None, None),
        ("not-a-date", None),
        ("2021-13-01", None),
    ],
)
def test_parse_trial_date(raw, expected):
    assert parse_trial_date(raw) == expected


@pytest.mark.parametrize(
    "completion, lookback, expected",
    [
        ("2021-08-15", 6, date(2021, 2, 15)),
        ("2020-03-31", 6, date(2019, 9, 28)),  # day clamped to 28
        ("2022-01-15", 12, date(2021, 1, 15)),
        ("2021-01-10", 1, date(2020, 12, 10)),
        (None, 6, None),
    ],
)
def test_cutoff_for_trial(completion, lookback, expected):
    trial = Trial(nct_id="X", completion_date=completion)
    assert cutoff_for_trial(trial, lookback) == expected
