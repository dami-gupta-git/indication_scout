"""Unit tests for trial_risk.data."""

import json

from indication_scout.ml_models.trial_risk.data import load_labeled_trials


def _write_cache_entry(cache_dir, namespace, drug, mesh_term, trials):
    ns = cache_dir / namespace
    ns.mkdir(parents=True, exist_ok=True)
    (ns / f"{namespace}_{drug}_{mesh_term}.json").write_text(
        json.dumps(
            {
                "ns": namespace,
                "params": {"drug": drug, "mesh_term": mesh_term, "date_before": None},
                "data": {"total_count": len(trials), "trials": trials},
            }
        )
    )


def test_load_labeled_trials_dedups_and_labels(tmp_path):
    completed_trial = {
        "nct_id": "NCT0000001",
        "title": "C1",
        "phase": "Phase 2",
        "overall_status": "Completed",
        "sponsor": "Acme",
        "enrollment": 50,
        "mesh_conditions": [{"id": "D001", "term": "Diabetes"}],
        "interventions": [{"intervention_type": "Drug", "intervention_name": "X"}],
    }
    terminated_trial = {
        "nct_id": "NCT0000002",
        "title": "T1",
        "phase": "Phase 1",
        "overall_status": "Terminated",
        "why_stopped": "Slow accrual",
        "sponsor": "Pfizer Inc",
        "enrollment": 10,
        "mesh_conditions": [{"id": "D002", "term": "Cancer"}],
        "interventions": [{"intervention_type": "Drug", "intervention_name": "Y"}],
    }

    _write_cache_entry(tmp_path, "ct_completed", "drugA", "Diabetes", [completed_trial])
    # Same trial appears under another mesh — should dedup.
    _write_cache_entry(
        tmp_path, "ct_completed", "drugA", "OtherCondition", [completed_trial]
    )
    _write_cache_entry(tmp_path, "ct_terminated", "drugB", "Cancer", [terminated_trial])

    labeled = load_labeled_trials(tmp_path)
    by_nct = {lt.trial.nct_id: lt for lt in labeled}

    assert len(labeled) == 2
    assert by_nct["NCT0000001"].label == 0
    assert by_nct["NCT0000001"].drug == "drugA"
    assert by_nct["NCT0000001"].trial.phase == "Phase 2"
    assert by_nct["NCT0000002"].label == 1
    assert by_nct["NCT0000002"].drug == "drugB"
    assert by_nct["NCT0000002"].trial.why_stopped == "Slow accrual"


def test_load_labeled_trials_terminated_wins_on_collision(tmp_path):
    same_nct = {
        "nct_id": "NCT0000003",
        "title": "X",
        "phase": "Phase 2",
        "overall_status": "Terminated",
        "why_stopped": "lack of efficacy",
    }
    completed_view = {**same_nct, "overall_status": "Completed", "why_stopped": None}

    _write_cache_entry(tmp_path, "ct_completed", "drugC", "Foo", [completed_view])
    _write_cache_entry(tmp_path, "ct_terminated", "drugC", "Foo", [same_nct])

    labeled = load_labeled_trials(tmp_path)

    assert len(labeled) == 1
    assert labeled[0].label == 1
    assert labeled[0].trial.why_stopped == "lack of efficacy"


def test_load_labeled_trials_empty_dirs(tmp_path):
    labeled = load_labeled_trials(tmp_path)
    assert labeled == []
