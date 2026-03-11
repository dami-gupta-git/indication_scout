## Test Layout

```
tests/
├── conftest.py              # shared fixtures (sample_drug, sample_indication)
├── unit/                    # no network, no external deps
│   └── test_<module>.py
└── integration/             # hits real external APIs
    ├── conftest.py          # client fixtures (open_targets_client, pubmed_client, etc.)
    └── test_<source>.py
```

## Test Style
Write tests as plain functions (def test_...), not inside classes. Do not use class Test... grouping.

## CRITICAL - Testing & Verification
- When writing tests, make sure that unit tests and integration tests are created correctly. Integration tests involve external APIs, databases, or network calls. Just constructing instances and asserting on their attributes/string representation are unit tests.
- Run tests with `pytest` before and after non-trivial changes.
- Tests should be organized into subdirectories that reflect the file being tested.
- When testing a specific function, when the test code is exactly the same, but only the input-output is different, prefer 
to use pytest.mark.parametrize, instead of separate same tests. However, at a time there should be no more than 5 input-output entries at a time.
- For code changes that affect LLM behavior, add or update tests at the smallest reasonable unit (e.g. prompt builders, service-layer helpers) instead of relying only on end-to-end tests.
- Before suggesting a refactor, first:
  - Identify the current behavior and entry points it affects (CLI, Streamlit, services).
  - Describe how to verify that behavior is preserved (tests, sample inputs, or fixtures).
- Do NOT run integration tests automatically.
- Do NOT run tests that hit external APIs or databases.
- After refactoring:
  - Ensure all existing unit tests pass.
  - Run only the corresponding integration tests only to ensure they pass.
  - Add targeted regression tests when fixing bugs or changing behavior.
- If test coverage is missing in a changed area, propose minimal, focused tests that validate the new or modified behavior.
- If expected values for an integration test are not already known (from existing tests, docs, or data you've provided), you MUST stop and ask for them. 
Writing the test with placeholder/weak assertions and asking later is not acceptable.

## CRITICAL -  Test Assertions
- Check ALL fields for the object being tested, not just a subset.
- Assertions must verify actual values, not just types or existence:
  - BAD: `assert result is not None`, `assert isinstance(result, Drug)`, `assert len(items) > 0`
  - GOOD: `assert result.name == "metformin"`, `assert result.phase == 4`, `assert len(items) == 3`
- When testing a list, extract a specific item and test all its fields:
  ```
  # BAD
  assert len(trials) > 0
  assert all(isinstance(t, Trial) for t in trials)

  # GOOD
  assert len(trials) == 5
  trial = trials[0]
  assert trial.nct_id == "NCT12345678"
  assert trial.phase == "Phase 3"
  assert trial.status == "Completed"
  assert trial.enrollment == 500
  ```
- Use pytest.mark.parametrize
 ```
# BAD
 
 @pytest.mark.asyncio
async def test_semaglutide_drug_data(open_targets_client):
    drug = await open_targets_client.get_drug("semaglutide")
    assert drug.name == "SEMAGLUTIDE"
    assert drug.chembl_id == "CHEMBL2108724"
    ....
    
 @pytest.mark.asyncio    
async def test_trastuzumab_adverse_event(open_targets_client):
    drug = await open_targets_client.get_drug("")
    assert drug.name == "trastuzumab"
    assert drug.chembl_id == "CHEMBL426"
    
# GOOD    
 @pytest.mark.asyncio
 @pytest.mark.parametrize(
    "drug_name, chembl_id",
    [
        (
            "semaglutide","CHEMBL2108724,
        ),
        (
            "trastuzumab","CHEMBL426,
        ),
async def test_drug_data(open_targets_client, drug, chembl_id):
    """Test fetching drug data """
    drug = await open_targets_client.get_drug(drug)
    assert drug.name == drug
    assert drug.chembl_id == chembl_id
    
        
 ```
