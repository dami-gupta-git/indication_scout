"""Unit tests for is_non_therapeutic_study — the disease-independent trial drop.

A trial whose every typed intervention is a diagnostic test, device, imaging tracer or procedure
studies that object, not the drug. The answer does not depend on which candidate indication is
under investigation, so it is settled once in code; re-asking it per candidate is what let
NCT02440893 count as evidence under coronary artery disorder and be excluded under cardiovascular
disorder in the same report.

Interventions are transcribed from real ClinicalTrials.gov v2 records, title-cased as the client
stores them.
"""

import pytest

from indication_scout.agents._trial_signals import is_non_therapeutic_study
from indication_scout.models.model_clinical_trials import Intervention, Trial


@pytest.mark.parametrize(
    "nct_id, interventions, expected",
    [
        # Sole intervention is the diagnostic test; metformin is not registered at all.
        ("NCT02440893", [("Diagnostic Test", "Corus CAD (ASGES)")], True),
        # 11C-metformin is a PET tracer — the name carries the drug, the type does not.
        ("NCT03122769", [("Radiation", "11C-metformin")], True),
        # A Drug intervention is present (a different agent), so the relevance gate judges it.
        (
            "NCT07209527",
            [
                ("Device", "Truway Portable Ultrasound Device"),
                ("Device", "Truway Blood Glucose Monitor"),
                ("Drug", "Standard Oral Hypoglycemic Agent"),
            ],
            False,
        ),
        # Device and behavioral arms co-listed with the studied drug must not drop the trial.
        (
            "NCT04625946",
            [
                ("Drug", "Metformin"),
                ("Behavioral", "Recommendations for lifestyle modification."),
                ("Device", "AliveCor"),
            ],
            False,
        ),
        # No interventions recorded — absence of a type is not evidence of a diagnostic study.
        ("NCT00510705", [], False),
    ],
)
def test_is_non_therapeutic_study(
    nct_id: str, interventions: list[tuple[str, str]], expected: bool
) -> None:
    trial = Trial(
        nct_id=nct_id,
        interventions=[
            Intervention(intervention_type=t, intervention_name=n)
            for t, n in interventions
        ],
    )

    assert is_non_therapeutic_study(trial) is expected
