# API Clients Guide

This document describes the data source clients used in IndicationScout.

## Overview

All clients inherit from `BaseClient`, which provides:
- Async HTTP requests with retry and exponential backoff
- Session management (connection pooling)
- Consistent error handling via `DataSourceError`

## BaseClient

```python
from indication_scout.data_sources.base_client import BaseClient, DataSourceError
```

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `timeout` | 30.0 | Request timeout in seconds |
| `max_retries` | 3 | Number of retry attempts for failed requests |

### Error Handling

All clients raise `DataSourceError` on failure:

```python
class DataSourceError(Exception):
    source: str        # e.g., "open_targets", "clinical_trials", "pubmed"
    status_code: int   # HTTP status code (if applicable)
```

Retryable errors (429, 5xx) are automatically retried with exponential backoff.

---

## OpenTargetsClient

GraphQL client for the Open Targets Platform.

```python
from indication_scout.data_sources.open_targets import OpenTargetsClient
```

### Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `get_drug(chembl_id)` | Fetch drug data by ChEMBL ID | `DrugData` |
| `get_rich_drug_data(chembl_id)` | Fetch drug + all target data in parallel | `RichDrugData` |
| `get_target_data(target_id)` | Fetch target data by Ensembl ID | `TargetData` |
| `get_drug_indications(chembl_id)` | All indications for a drug | `list[Indication]` |
| `get_drug_competitors(chembl_id, min_stage="PHASE_3")` | Raw competitor data: diseases with drugs at or above `min_stage`, plus the drug's own indications list | `CompetitorRawData` (TypedDict with `diseases: dict[str, set[str]]` and `drug_indications: list[str]`). The top-N slicing, LLM dedup, and caching happen in `RetrievalService.get_drug_competitors()`, not in the client. |
| `get_drug_target_competitors(chembl_id)` | All drugs acting on each of the drug's targets, keyed by target symbol | `dict[str, list[DrugSummary]]` |
| `get_disease_drugs(disease_id)` | Get drugs for a disease, deduplicated by drug_id | `list[DrugSummary]` |
| `get_disease_synonyms(disease_name)` | Exact, related, narrow, and broad synonyms | `DiseaseSynonyms` |

> **Note:** Open Targets uses ChEMBL IDs (e.g. `CHEMBL1201583`) as the sole drug identifier. Resolve a free-text drug name to a ChEMBL ID upstream before calling these methods.

### Convenience Accessors

These methods call `get_target_data()` and return a specific slice:

| Method | Returns |
|--------|---------|
| `get_target_data_associations(target_id)` | `list[Association]` (filtered by `settings.open_targets_association_min_score`) |
| `get_target_data_pathways(target_id)` | `list[Pathway]` |
| `get_target_data_interactions(target_id)` | `list[Interaction]` |
| `get_target_data_drug_summaries(target_id)` | `list[DrugSummary]` |
| `get_target_data_tissue_expression(target_id)` | `list[TissueExpression]` |
| `get_target_data_mouse_phenotypes(target_id)` | `list[MousePhenotype]` |
| `get_target_data_safety_liabilities(target_id)` | `list[SafetyLiability]` |
| `get_target_data_genetic_constraints(target_id)` | `list[GeneticConstraint]` |

### Caching

Results are cached to disk (`_cache/` directory) with a 5-day TTL.

### Example

```python
async with OpenTargetsClient() as client:
    # Get drug data (semaglutide ChEMBL ID)
    drug = await client.get_drug("CHEMBL1201583")
    print(f"{drug.chembl_id}: {len(drug.indications)} indications")

    # Get target data
    target = await client.get_target_data("ENSG00000112164")  # GLP1R
    print(f"{target.symbol}: {len(target.associations)} disease associations")
```

### Error Behavior

- **Nonexistent drug/target**: Raises `DataSourceError` with "No drug found" or "No target found"
- **Empty string**: Raises `DataSourceError`
- **Invalid format**: Raises `DataSourceError`

---

## ClinicalTrialsClient

REST client for ClinicalTrials.gov API v2.

```python
from indication_scout.data_sources.clinical_trials import ClinicalTrialsClient
```

### Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `get_trial(nct_id)` | Fetch a single trial by NCT ID | `Trial` |
| `search_trials(drug, indication, ...)` | Search trials by drug and optional indication | `list[Trial]` |
| `detect_whitespace(drug, indication, ...)` | Check if drug-indication pair is unexplored | `WhitespaceResult` |
| `get_landscape(indication, top_n=50, ...)` | Competitive landscape for an indication | `IndicationLandscape` |
| `get_terminated(drug, indication, ...)` | Trial-outcome evidence split into four scopes | `TrialOutcomes` |

### Parameters

**search_trials**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `drug` | str | Drug/intervention name |
| `indication` | str \| None | Disease/indication |
| `date_before` | date \| None | Temporal holdout cutoff |
| `phase_filter` | str \| None | e.g., "PHASE3", "(PHASE2 OR PHASE3)" |
| `sort` | str \| None | CT.gov sort spec, e.g. `"EnrollmentCount:desc"`, `"StartDate:desc"` |
| `target_mesh_id` | str \| None | MeSH descriptor (D-number). When set, results are post-filtered to trials whose `mesh_conditions` or `mesh_ancestors` contain this D-number. Resolve via `services.disease_helper.resolve_mesh_id`. |

**Note on `target_mesh_id`**: ClinicalTrials.gov's Essie engine is recall-first
— a `query.cond` for "hypertension" returns glaucoma/portal/pulmonary hypertension
trials too. `target_mesh_id` post-filters by the canonical MeSH D-number to
keep only trials tagged for the intended disease. `detect_whitespace`,
`get_landscape`, and `get_terminated` accept the same parameter.

### Example

```python
from indication_scout.services.disease_helper import resolve_mesh_id

async with ClinicalTrialsClient() as client:
    mesh_id = await resolve_mesh_id("hypertension")  # "D006973"

    # Search for trials
    trials = await client.search_trials(
        drug="semaglutide",
        indication="hypertension",
        phase_filter="PHASE3",
        sort="EnrollmentCount:desc",
        target_mesh_id=mesh_id,
    )

    # Check whitespace
    result = await client.detect_whitespace(
        "tirzepatide", "Huntington disease",
        target_mesh_id=await resolve_mesh_id("Huntington disease"),
    )
    if result.is_whitespace:
        print(f"No trials found. {len(result.indication_drugs)} other drugs in this space.")

    # Competitive landscape
    landscape = await client.get_landscape("gastroparesis", top_n=10)
    for competitor in landscape.competitors:
        print(f"{competitor.sponsor}: {competitor.drug_name} ({competitor.max_phase})")

    # Trial-outcome evidence (four scopes)
    outcomes = await client.get_terminated(
        "semaglutide", "hypertension", target_mesh_id=mesh_id,
    )
    print(len(outcomes.drug_wide), len(outcomes.indication_wide))
    print(len(outcomes.pair_specific), len(outcomes.pair_completed))
```

### Error Behavior

- **Nonexistent drug/indication**: Returns empty results (not an error)
- **Empty string**: Returns empty or unfiltered results
- **Unresolvable MeSH term**: `resolve_mesh_id` returns `None`; pass `target_mesh_id=None` to skip post-filtering (recall-first results)
- **API errors**: Raises `DataSourceError`

---

## PubMedClient

REST client for NCBI PubMed E-utilities.

```python
from indication_scout.data_sources.pubmed import PubMedClient
```

### Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `search(query, max_results, date_before)` | Search for PMIDs (cached) | `list[str]` |
| `get_count(query, date_before)` | Count results without fetching | `int` |
| `fetch_abstracts(pmids, batch_size)` | Fetch abstract details by PMID | `list[PubmedAbstract]` |

### Example

```python
async with PubMedClient() as client:
    # Search and fetch
    pmids = await client.search("semaglutide diabetes", max_results=10)
    abstracts = await client.fetch_abstracts(pmids)

    for abstract in abstracts:
        logger.info("[%s] %s", abstract.pmid, abstract.title)
        logger.info("  Authors: %s", ", ".join(abstract.authors[:3]))
        logger.info("  Journal: %s", abstract.journal)

    # Quick count
    count = await client.get_count("GLP-1 receptor agonist")
    logger.info("Found %d articles", count)
```

### Error Behavior

- **Nonexistent search term**: Returns empty list (not an error)
- **Invalid PMID**: Returns empty list (article not found)
- **Empty query**: Returns empty list
- **XML parse error**: Raises `DataSourceError`

---

## ChEMBLClient

REST client for the ChEMBL compound database.

```python
from indication_scout.data_sources.chembl import ChEMBLClient
```

### Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `get_molecule(chembl_id)` | Fetch molecule properties by ChEMBL ID | `MoleculeData` |
| `get_atc_description(atc_code)` | Fetch ATC classification hierarchy for an ATC code (cached) | `ATCDescription` |

### Example

```python
async with ChEMBLClient() as client:
    molecule = await client.get_molecule("CHEMBL1118")
    logger.info("%s: %s", molecule.molecule_chembl_id, molecule.molecule_type)
    logger.info("Max phase: %s, Oral: %s", molecule.max_phase, molecule.oral)

    atc = await client.get_atc_description("A10BA02")
    logger.info("Level 3: %s, Level 4: %s", atc.level3_description, atc.level4_description)
```

### Error Behavior

- **Nonexistent ChEMBL ID**: Raises `DataSourceError` (HTTP 404)
- **Nonexistent ATC code**: Raises `DataSourceError`
- **Malformed response**: Raises `DataSourceError`

See [chembl.md](chembl.md) for full field-level documentation and agent usage.

---

## Common Patterns

### Using as Context Manager

All clients support async context managers for automatic cleanup:

```python
async with OpenTargetsClient() as client:
    result = await client.get_drug("metformin")
# Session automatically closed
```

### Manual Cleanup

```python
client = ClinicalTrialsClient()
try:
    result = await client.search_trials(drug="aspirin")
finally:
    await client.close()
```

### Parallel Requests

```python
import asyncio

async with OpenTargetsClient() as client:
    drugs = ["metformin", "semaglutide", "tirzepatide"]
    results = await asyncio.gather(*[client.get_drug(d) for d in drugs])
```

### Error Handling

```python
from indication_scout.data_sources.base_client import DataSourceError

async with OpenTargetsClient() as client:
    try:
        drug = await client.get_drug("not_a_real_drug")
    except DataSourceError as e:
        print(f"Error from {e.source}: {e}")
        if e.status_code:
            print(f"HTTP status: {e.status_code}")
```
