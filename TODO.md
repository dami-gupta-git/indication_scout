# TODO

Active tasks organized by component. Items here are actionable and derived from stubs, known gaps, and code issues in the codebase.

---

## RAG Pipeline

- [x] Implement `synthesize()` in `services/retrieval.py`
- [x] Define `EvidenceSummary` Pydantic model in `models/model_evidence_summary.py`
- [x] Write `prompts/synthesize.txt` prompt template
- [x] Implement `run_rag()` in `runners/rag_runner.py`
- [x] Unit tests for `run_rag` in `tests/unit/runners/test_rag_runner.py`
- [ ] Integration tests for `synthesize` in `tests/integration/services/test_retrieval.py`
- [ ] Integration test for `run_rag` in `tests/integration/services/test_retrieval.py`

## Agents

- [ ] Implement `Orchestrator.run()` in `agents/orchestrator.py`
- [ ] Implement `LiteratureAgent.run()` in `agents/literature.py`
- [ ] Implement `ClinicalTrialsAgent.run()` in `agents/clinical_trials.py`
- [ ] Implement `MechanismAgent.run()` in `agents/mechanism.py`
- [ ] Implement `SafetyAgent.run()` in `agents/safety.py`

## Data Sources

- [ ] Implement `DrugBankClient.get_drug()` and `get_interactions()` in `data_sources/drugbank.py`

## API

- [ ] Define API routes in `api/routes/`
- [ ] Define request/response schemas in `api/schemas/`

## CLI

- [ ] Implement CLI module (`indication_scout.cli.cli`) referenced in `pyproject.toml`

## Infrastructure

- [ ] Add connection pooling singleton to `db/session.py` (currently creates new engine per call)

## Code Quality

- [ ] Fix `runners/pubmed_runner.py` to use `logging` instead of `print()`
- [ ] Remove superseded tests in `tests/integration/services/test_pubmed_query.py` (two tests marked `# TODO delete`)
- [ ] Review `get_drug_competitors()` in Open Targets client (marked `# TODO needs rework`)
