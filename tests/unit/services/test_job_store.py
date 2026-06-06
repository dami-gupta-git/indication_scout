"""Unit tests for services/job_store — pure in-memory state, no network/DB/LLM."""

from indication_scout.services.job_store import Job, JobStore


def test_create_registers_pending_job_with_defaults():
    store = JobStore()
    job = store.create("metformin")

    assert isinstance(job, Job)
    assert job.drug_name == "metformin"
    assert job.status == "pending"
    assert job.result is None
    assert job.error is None
    assert job.task is None
    assert isinstance(job.job_id, str)
    assert len(job.job_id) == 32  # uuid4().hex


def test_create_assigns_unique_ids():
    store = JobStore()
    ids = {store.create("metformin").job_id for _ in range(5)}

    assert len(ids) == 5


def test_get_returns_same_instance_for_known_id():
    store = JobStore()
    job = store.create("aspirin")

    assert store.get(job.job_id) is job


def test_get_returns_none_for_unknown_id():
    store = JobStore()
    store.create("aspirin")

    assert store.get("does-not-exist") is None


def test_all_returns_every_job_in_insertion_order():
    store = JobStore()
    first = store.create("metformin")
    second = store.create("aspirin")
    third = store.create("rapamycin")

    jobs = store.all()

    assert [j.job_id for j in jobs] == [first.job_id, second.job_id, third.job_id]
    assert [j.drug_name for j in jobs] == ["metformin", "aspirin", "rapamycin"]


def test_lifecycle_fields_persist_when_mutated():
    store = JobStore()
    job = store.create("metformin")

    job.status = "error"
    job.error = "boom"

    fetched = store.get(job.job_id)
    assert fetched.status == "error"
    assert fetched.error == "boom"
    assert fetched.result is None
