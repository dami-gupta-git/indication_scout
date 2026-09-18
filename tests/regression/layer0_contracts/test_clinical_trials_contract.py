"""ClinicalTrials.gov client contract: a single study parses into a Trial as recorded."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from indication_scout.data_sources.clinical_trials import ClinicalTrialsClient

pytestmark = pytest.mark.contract


async def test_get_trial_parses_every_field(
    cassette: Callable[[str], Iterator[None]], contract_cache_dir: Path
) -> None:
    with cassette("ct_trial"):
        async with ClinicalTrialsClient(cache_dir=contract_cache_dir) as client:
            trial = await client.get_trial("NCT04971785")

    assert trial.nct_id == "NCT04971785"
    assert trial.title.startswith("Study of Semaglutide, and Cilofexor/Firsocostat")
    assert trial.brief_summary is not None
    assert "cirrhosis due to NASH" in trial.brief_summary
    assert trial.phase == "Phase 2"
    assert trial.overall_status == "COMPLETED"
    assert trial.why_stopped is None
    assert trial.indications == ["Nonalcoholic Steatohepatitis"]
    assert trial.sponsor == "Gilead Sciences"
    assert trial.enrollment == 457
    assert trial.start_date == "2021-08-09"
    assert trial.completion_date == "2024-11-12"
    assert trial.references == []

    assert len(trial.mesh_conditions) == 1
    assert trial.mesh_conditions[0].id == "D065626"
    assert trial.mesh_conditions[0].term == "Non-alcoholic Fatty Liver Disease"
    assert [m.id for m in trial.mesh_ancestors] == ["D005234", "D008107", "D004066"]

    assert len(trial.interventions) == 4
    first = trial.interventions[0]
    assert first.intervention_type == "Drug"
    assert first.intervention_name == "Semaglutide (SEMA)"
    assert first.description == "Administered as subcutaneous (SC) injection"
    assert first.arm_group_labels == ["SEMA + CILO/FIR FDC", "SEMA + PTM CILO/FIR"]

    assert len(trial.arm_groups) == 4
    arm = trial.arm_groups[0]
    assert arm.label == "SEMA + CILO/FIR FDC"
    assert arm.arm_type == "Experimental"
    assert arm.intervention_names == [
        "Drug: Semaglutide (SEMA)",
        "Drug: Cilofexor (CILO)/Firsocostat (FIR)",
    ]

    assert len(trial.primary_outcomes) == 1
    assert trial.primary_outcomes[0].time_frame == "Week 72"
    assert trial.primary_outcomes[0].measure.startswith(
        "Percentage of Participants Who Achieved"
    )
