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
| `get_drug(name)` | Fetch drug data by name | `DrugData` |
| `get_rich_drug_data(name)` | Fetch drug + all target data in parallel | `RichDrugData` |
| `get_target_data(target_id)` | Fetch target data by Ensembl ID | `TargetData` |
| `get_drug_indications(name)` | All indications for a drug | `list[Indication]` |
| `get_drug_competitors(name)` | Top 10 diseases with phase 3+ competitor drugs | `dict[str, set[str]]` |
| `get_drug_target_competitors(name)` | All drugs acting on each of the drug's targets | `dict[str, list[DrugSummary]]` |
| `get_disease_drugs(disease_id)` | Get drugs for a disease, deduplicated by drug_id | `list[DrugSummary]` |
| `get_disease_synonyms(disease_name)` | Exact, related, narrow, and broad synonyms | `DiseaseSynonyms` |

### Convenience Accessors

These methods call `get_target_data()` and return a specific slice:

| Method | Returns |
|--------|---------|
| `get_target_data_associations(target_id, min_score)` | `list[Association]` |
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
    # Get drug data
    drug = await client.get_drug("semaglutide")
    print(f"{drug.name}: {len(drug.indications)} indications")

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
| `search_trials(drug, condition, ...)` | Search trials by drug and/or condition | `list[Trial]` |
| `detect_whitespace(drug, condition)` | Check if drug-condition pair is unexplored | `WhitespaceResult` |
| `get_landscape(condition, top_n)` | Competitive landscape for a condition | `ConditionLandscape` |
| `get_terminated(query, max_results)` | Terminated/withdrawn/suspended trials | `list[TerminatedTrial]` |

### Parameters

**search_trials**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `drug` | str | Drug/intervention name |
| `condition` | str \| None | Disease/condition |
| `date_before` | date \| None | Temporal holdout cutoff |
| `phase_filter` | str \| None | e.g., "PHASE3", "(PHASE2 OR PHASE3)" |
| `max_results` | int | Maximum trials to return (default: 200) |

### Example

```python
async with ClinicalTrialsClient() as client:
    # Search for trials
    trials = await client.search_trials(
        drug="semaglutide",
        condition="diabetes",
        phase_filter="PHASE3",
        max_results=50,
    )

    # Check whitespace
    result = await client.detect_whitespace("tirzepatide", "Huntington disease")
    if result.is_whitespace:
        print(f"No trials found. {len(result.condition_drugs)} other drugs in this space.")

    # Competitive landscape
    landscape = await client.get_landscape("gastroparesis", top_n=10)
    for competitor in landscape.competitors:
        print(f"{competitor.sponsor}: {competitor.drug_name} ({competitor.max_phase})")
```

### Error Behavior

- **Nonexistent drug/condition**: Returns empty results (not an error)
- **Empty string**: Returns empty or unfiltered results
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
