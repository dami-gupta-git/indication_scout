# Testing

The suite has four kinds of test. Unit and integration tests cover behaviour as it is written;
the regression layers pin behaviour as it currently *is*, so a change that alters output is
visible before it reaches a report.

| Kind | Location | Network | Run with |
| --- | --- | --- | --- |
| Unit | `tests/unit/` | none | `pytest tests/unit/` |
| Integration | `tests/integration/` | real APIs | `pytest tests/integration/` |
| Contract (Layer 0) | `tests/regression/layer0_contracts/` | none (replayed) | `pytest -m contract` |
| Evidence gate (Layer 1) | `tests/regression/layer1_deterministic/` | none | `pytest tests/regression/layer1_deterministic/` |
| Structural specs (Layer 2) | `tests/regression/layer2_structural/` | none, but needs generated reports | `pytest -m regression_layer2` |
| Full-pipeline replay | `tests/regression/pipeline_replay/` | none (replayed), needs Postgres | `pytest -m regression` |

A plain `pytest` run excludes the two regression markers (`regression`, `regression_layer2`); the
contract tests are not excluded and run with everything else. In CI they run on every push as
part of `make test-regression`, alongside the evidence-gate and report-diff tests.

Tests mirror the source tree: a test for a module under `services/` belongs in
`tests/<unit|integration>/services/`, and so on. Conventions for writing them — assertions on
real values rather than types, parametrization limits, no `print()` — are in `skills/testing.md`.

## Data-source contract tests

Each data source client has one recorded response per method. The test calls the client, and
asserts every field of the parsed model against what the recording contains. A change to a
client, a parse helper, or a Pydantic model fails here and names the source that broke, without
running an agent, an LLM, or the database.

Covered today: Open Targets (drug, target), ClinicalTrials.gov (single trial), PubMed (search,
abstracts), Europe PMC (citation counts), ChEMBL (molecule, ATC), openFDA (label indications,
label safety).

```bash
pytest -m contract                 # all of them, about a second
pytest -m contract -k chembl       # one client
```

Each test runs against an empty cache directory, so the call always goes through the client's
parse path rather than being served from a warm file cache.

### Re-recording

Re-record when a client legitimately changes what it requests or how it parses:

```bash
SCOUT_CASSETTE_MODE=record pytest -m contract -k chembl
```

Recording hits the real API. It must run with the test constants file
(`CONSTANTS_FILE=.env.constants.test`, which `tests/conftest.py` sets), because the production
constants use a different Open Targets page size and replay then finds no matching request.

Recording rewrites the assertions' source of truth, so read the diff: a count that moved is
either the upstream data changing or the change under test.

### Checking for upstream drift

```bash
SCOUT_CASSETTE_MODE=live pytest -m contract
```

This bypasses the recordings and hits the real APIs, which is how a renamed or dropped upstream
field surfaces. Expect noise from values that move on their own — citation counts and
relevance-sorted PubMed results will differ from what was recorded.

### Adding one

Add the call to the recording script pattern, record its cassette under
`tests/regression/layer0_contracts/cassettes/<name>.yaml`, then write the assertions from the
recorded values. One cassette per method keeps a re-recording from churning unrelated files;
two tests may share a cassette when they issue the same request (the two openFDA tests do).

## Regression layers

`tests/regression/README.md` is the authoritative document for Layers 1 and 2, the spec
assertion types, the full-pipeline replay, and how to pin a new drug.
