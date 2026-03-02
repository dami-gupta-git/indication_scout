# Open Targets Client — Data Contracts

**API:** Open Targets Platform GraphQL API (`https://api.platform.opentargets.org/api/v4/graphql`)
**Auth:** None required
**Cache TTL:** 5 days

The client provides three primary entry points that fetch complete data in a single call, plus convenience accessors for specific data types.

---

## Primary Entry Points

### `get_rich_drug_data(drug_name) → RichDrugData`

Fetches drug data and all associated target data in parallel. Calls `get_drug()` to resolve the drug, then calls `get_target_data()` concurrently for every target in `DrugData.targets`.

**In:** `"semaglutide"` (a human-friendly drug name)

**Out:** `RichDrugData` containing:
- `drug` — `DrugData` (see below)
- `targets` — `list[TargetData]`, one per target in `drug.targets`, fetched in parallel

**Agent use:** Provides everything Open Targets knows about a drug and all its targets in a single object, for agents that need both drug-level and target-level data without making multiple calls.

---

### `get_drug(drug_name) → DrugData`

Resolves a drug name to its ChEMBL ID and fetches all drug data in a single call.

**In:** `"semaglutide"` (a human-friendly drug name)

**Out:** `DrugData` containing:
- `chembl_id` — the canonical drug identifier (e.g. `"CHEMBL3137309"`)
- `name` — normalized name
- `synonyms` — alternative names (Ozempic, Wegovy, Rybelsus)
- `trade_names` — brand names
- `drug_type` — small molecule, antibody, peptide, etc.
- `is_approved` — bool
- `max_clinical_phase` — highest phase reached (0–4)
- `year_first_approved` — int or None
- `targets` — list of `DrugTarget`:
  - `target_id` — Ensembl gene ID (e.g. `"ENSG00000112164"`)
  - `target_symbol` — gene symbol (e.g. `"GLP1R"`)
  - `mechanism_of_action` — string (e.g. `"Glucagon-like peptide 1 receptor agonist"`)
  - `action_type` — e.g. `"AGONIST"`, `"INHIBITOR"`
- `indications` — list of `Indication`:
  - `disease_id` — EFO identifier
  - `disease_name` — human-readable
  - `max_phase` — highest phase for this indication (0–4)
  - `references` — supporting references
- `warnings` — list of `DrugWarning`:
  - `warning_type` — `"Black Box Warning"` or `"Withdrawn"`
  - `description` — adverse effect description
  - `toxicity_class` — classification of toxicity type
  - `country` — where the warning was issued
  - `year` — when issued
  - `efo_id` — disease term for the warning class
- `adverse_events` — list of `AdverseEvent`:
  - `name` — the adverse event term (e.g. `"pancreatitis"`)
  - `meddra_code` — MedDRA identifier
  - `count` — number of FAERS reports
  - `log_likelihood_ratio` — statistical significance
- `adverse_events_critical_value` — threshold for significance

**Properties:**
- `approved_disease_ids` — set of disease IDs with phase 4 / approved status
- `investigated_disease_ids` — set of all disease IDs being actively investigated

---

### `get_target_data(target_id) → TargetData`

Fetches all target data in a single GraphQL call. Paginates associations if >500.

**In:** `"ENSG00000112164"` (Ensembl gene ID)

**Out:** `TargetData` containing:
- `target_id` — Ensembl ID
- `symbol` — gene symbol (e.g. `"GLP1R"`)
- `name` — approved gene name
- `associations` — list of `Association`
- `pathways` — list of `Pathway`
- `interactions` — list of `Interaction`
- `drug_summaries` — list of `DrugSummary`
- `expressions` — list of `TissueExpression`
- `mouse_phenotypes` — list of `MousePhenotype`
- `safety_liabilities` — list of `SafetyLiability`
- `genetic_constraint` — list of `GeneticConstraint`

---

## Convenience Accessors

These methods call `get_target_data()` internally and return specific slices:

| Method | Returns |
|--------|---------|
| `get_target_data_associations(target_id, min_score=0.1)` | `list[Association]` filtered by score |
| `get_target_data_pathways(target_id)` | `list[Pathway]` |
| `get_target_data_interactions(target_id)` | `list[Interaction]` |
| `get_target_data_drug_summaries(target_id)` | `list[DrugSummary]` |
| `get_target_data_tissue_expression(target_id)` | `list[TissueExpression]` |
| `get_target_data_mouse_phenotypes(target_id)` | `list[MousePhenotype]` |
| `get_target_data_safety_liabilities(target_id)` | `list[SafetyLiability]` |
| `get_target_data_genetic_constraints(target_id)` | `list[GeneticConstraint]` |

Additional methods:
| Method | Returns |
|--------|---------|
| `get_drug_indications(drug_name)` | `list[Indication]` — all indications for a drug (calls `get_drug` internally) |
| `get_drug_competitors(name)` | `dict[str, set[str]]` — top 10 diseases with phase 3+ competitor drugs, grouped by disease name |
| `get_drug_target_competitors(drug_name)` | `dict[str, list[DrugSummary]]` — all drugs acting on each of the drug's targets, keyed by target symbol |
| `get_disease_drugs(disease_id)` | `list[DrugSummary]` — all drugs for a disease, deduplicated by drug_id |
| `get_disease_synonyms(disease_name)` | `DiseaseSynonyms` — exact, related, narrow, and broad synonyms for a disease |

---

## Data Models

### Association

Target-disease association with per-datatype evidence breakdown.

```python
class Association(BaseModel):
    disease_id: str               # EFO identifier (e.g. "EFO_0003847")
    disease_name: str             # Human-readable (e.g. "non-alcoholic steatohepatitis")
    overall_score: float          # 0-1, headline association score
    datatype_scores: dict[str, float]  # Per-datatype breakdown:
        # - genetic_association: GWAS, gene burden
        # - somatic_mutation: cancer driver evidence
        # - known_drug: existing drug evidence
        # - affected_pathway: Reactome pathway evidence
        # - literature: text mining co-mentions
        # - animal_model: mouse phenotype data
        # - rna_expression: differential expression
    therapeutic_areas: list[str]  # Top-level categories
```

**Agent use:** High overall score + zero `known_drug` score = potential whitespace. The `datatype_scores` breakdown tells the Literature Agent where to look and the Critique Agent what evidence is missing.

---

### Pathway

Reactome pathway the target participates in.

```python
class Pathway(BaseModel):
    pathway_id: str          # Reactome ID
    pathway_name: str        # e.g. "Incretin synthesis, secretion, and inactivation"
    top_level_pathway: str   # Parent category (e.g. "Signal transduction")
```

**Agent use:** If GLP1R participates in "Regulation of lipid metabolism" and NASH is a metabolic disease, that's a mechanistic link.

---

### Interaction

Protein-protein interaction partner.

```python
class Interaction(BaseModel):
    interacting_target_id: str       # Ensembl ID of partner
    interacting_target_symbol: str   # Gene symbol
    interaction_score: float         # Confidence 0-1 (None for Reactome/Signor)
    source_database: str             # IntAct, STRING, Signor, Reactome
    biological_role: str             # e.g. "enzyme target"
    evidence_count: int              # Supporting experiments
    interaction_type: str | None     # "physical", "functional", "signalling", "enzymatic"
```

**Agent use:** If GLP1R interacts strongly with proteins implicated in fibrosis, that strengthens the NASH hypothesis even if direct evidence is thin.

---

### TissueExpression

Expression data for a single tissue.

```python
class TissueExpression(BaseModel):
    tissue_id: str                   # UBERON ID
    tissue_name: str                 # e.g. "liver", "pancreas"
    tissue_anatomical_system: str    # e.g. "digestive system"
    rna: RNAExpression
    protein: ProteinExpression

class RNAExpression(BaseModel):
    value: float     # TPM (transcripts per million)
    quantile: int    # Relative level across tissues
    unit: str        # "TPM"

class ProteinExpression(BaseModel):
    level: int                           # 0-3
    reliability: bool                    # Measurement reliability
    cell_types: list[CellTypeExpression]

class CellTypeExpression(BaseModel):
    name: str        # Cell type name
    level: int       # Expression level
    reliability: bool
```

**Agent use:** For NASH evaluation, it matters whether GLP1R is expressed in liver tissue. High expression strengthens the hypothesis; absence is a legitimate concern.

---

### MousePhenotype

Phenotype observed in mouse models.

```python
class MousePhenotype(BaseModel):
    phenotype_id: str              # MP ontology ID (e.g. "MP:0001556")
    phenotype_label: str           # e.g. "increased circulating glucose level"
    phenotype_categories: list[str]  # Top-level MP categories
    biological_models: list[BiologicalModel]

class BiologicalModel(BaseModel):
    allelic_composition: str    # e.g. "Glp1r<tm1Ddr>/Glp1r<tm1Ddr>"
    genetic_background: str     # e.g. "involves: 129S1/Sv * C57BL/6"
    literature: list[str]       # PMIDs
    model_id: str               # MGI identifier
```

**Agent use:** If GLP1R knockout mice show liver-related phenotypes, that's direct animal model evidence. PMIDs bridge to PubMed for full paper retrieval.

---

### SafetyLiability

Known target safety effect from Open Targets.

```python
class SafetyLiability(BaseModel):
    event: str | None          # Safety event description
    event_id: str | None       # Event identifier
    effects: list[SafetyEffect]
    datasource: str | None     # Source of the data
    literature: str | None     # Literature reference
    url: str | None

class SafetyEffect(BaseModel):
    direction: str             # Effect direction
    dosing: str | None         # Dosing conditions
```

---

### GeneticConstraint

GnomAD loss-of-function intolerance score.

```python
class GeneticConstraint(BaseModel):
    constraint_type: str       # "syn" (synonymous), "mis" (missense), "lof" (loss-of-function)
    oe: float | None           # Observed/expected ratio (lower = more constrained)
    oe_lower: float | None     # Lower CI bound
    oe_upper: float | None     # Upper CI bound
    score: float | None        # Constraint score
    upper_bin: int | None      # 0 = most constrained, 5 = least
```

**Agent use:** A highly constrained gene (low OE, low bin) suggests inhibiting it could have dangerous effects. The Critique Agent can flag: "GLP1R is moderately constrained (LOF OE = 0.35)."

---

### AdverseEvent

Significant adverse event from FAERS for a drug.

```python
class AdverseEvent(BaseModel):
    name: str                      # Adverse event term (e.g. "pancreatitis")
    meddra_code: str | None        # MedDRA identifier
    count: int                     # FAERS report count
    log_likelihood_ratio: float    # Statistical significance (higher = more significant)
```

**Agent use:** FAERS adverse events help identify class-level safety concerns.

---

### DrugSummary

A drug known to act on a target, with indication and phase info.

```python
class DrugSummary(BaseModel):
    drug_id: str                   # ChEMBL ID
    drug_name: str
    disease_id: str                # Indication EFO ID
    disease_name: str
    phase: float                   # Clinical phase (0-4)
    status: str | None             # e.g. "Active", "Completed", "Terminated"
    mechanism_of_action: str
    clinical_trial_ids: list[str]  # NCT IDs (bridge to ClinicalTrials.gov)
```

**Agent use:** If three GLP-1 agonists are in Phase 3 for NASH, that's both validation (biology works) and a competitive concern.

---

### DrugData

Everything Open Targets knows about a drug.

```python
class DrugData(BaseModel):
    chembl_id: str = ""
    name: str = ""
    synonyms: list[str] = []
    trade_names: list[str] = []
    drug_type: str = ""
    is_approved: bool | None = None
    max_clinical_phase: float | None = None
    year_first_approved: int | None = None
    warnings: list[DrugWarning] = []
    indications: list[Indication] = []
    targets: list[DrugTarget] = []
    adverse_events: list[AdverseEvent] = []
    adverse_events_critical_value: float | None = None
    atc_classifications: list[str] = []

    @property
    def approved_disease_ids(self) -> set[str]
    @property
    def investigated_disease_ids(self) -> set[str]
```

---

### RichDrugData

DrugData combined with full TargetData for each of its targets.

```python
class RichDrugData(BaseModel):
    drug: DrugData | None = None
    targets: list[TargetData] = []
```

**Agent use:** Returned by `get_rich_drug_data()`. Provides everything Open Targets knows about a drug and all its targets in a single object, for agents that need both drug-level and target-level data without making multiple calls.

---

### DiseaseSynonyms

Synonyms for a disease from Open Targets, grouped by relation type.

```python
class DiseaseSynonyms(BaseModel):
    disease_id: str                # EFO/MONDO identifier
    disease_name: str              # Canonical disease name
    parent_names: list[str] = []   # Parent disease names in the ontology
    exact: list[str] = []          # Exact synonyms (hasExactSynonym)
    related: list[str] = []        # Related synonyms (hasRelatedSynonym)
    narrow: list[str] = []         # Narrow synonyms (hasNarrowSynonym)
    broad: list[str] = []          # Broad synonyms (hasBroadSynonym)

    @property
    def all_synonyms(self) -> list[str]  # exact + related + parent_names combined
```

**Agent use:** Used to expand disease search terms when querying other data sources (e.g., PubMed, ClinicalTrials.gov) that may use different terminology for the same condition.

---

### TargetData

Everything Open Targets knows about a target.

```python
class TargetData(BaseModel):
    target_id: str
    symbol: str
    name: str
    associations: list[Association] = []
    pathways: list[Pathway] = []
    interactions: list[Interaction] = []
    drug_summaries: list[DrugSummary] = []
    expressions: list[TissueExpression] = []
    mouse_phenotypes: list[MousePhenotype] = []
    safety_liabilities: list[SafetyLiability] = []
    genetic_constraint: list[GeneticConstraint] = []
```

---

## Data Flow

```
get_drug("semaglutide")
  │
  ├─→ DrugData.targets ──→ target_ids
  │                          │
  │                          └──→ get_target_data(target_id)
  │                                  │
  │                                  ├── associations   ──→ Biology Agent
  │                                  ├── pathways       ──→ Biology Agent
  │                                  ├── interactions   ──→ Biology Agent
  │                                  ├── expressions    ──→ Biology Agent
  │                                  ├── mouse_phenotypes ──→ Biology Agent
  │                                  ├── drug_summaries ──→ Landscape Agent
  │                                  ├── safety_liabilities ──→ Critique Agent
  │                                  └── genetic_constraint ──→ Critique Agent
  │
  ├─→ DrugData.indications ──→ Landscape Agent + Supervisor
  │
  ├─→ DrugData.approved_disease_ids ──→ Supervisor (filters out known uses)
  │
  ├─→ DrugData.warnings ──→ Critique Agent
  │
  └─→ DrugData.adverse_events ──→ Critique Agent
```

---

## Agent-to-Method Mapping

| Agent | Methods consumed | What it gets |
|-------|-----------------|--------------|
| **Biology** | `get_target_data` | Disease associations, pathways, PPIs, tissue expression, mouse phenotypes |
| **Landscape** | `get_target_data`, `get_drug_indications`, `get_disease_drugs` | Known drugs for target, full drug indications pipeline, disease drug landscape |
| **Critique** | `get_target_data`, `get_drug` | Association gaps, phenotype gaps, safety liabilities, genetic constraint, drug warnings, adverse events |
| **Supervisor** | `get_drug` (entry point) | Drug resolution, approved indications for filtering, dispatches target_ids to other agents |

---

## Caching

All public methods use a **disk-based file cache** in the `_cache/` directory with a 5-day TTL (configurable via `CACHE_TTL` in `constants.py`). Cache entries are JSON files keyed by a SHA-256 hash of the namespace and parameters. Expired entries are evicted on read.

Cached namespaces: `drug` (keyed by `chembl_id`), `target` (keyed by `target_id`), `disease_drugs` (keyed by `disease_id`), `disease_synonyms` (keyed by `disease_id`).

The convenience accessors all call `get_target_data` internally, so accessing multiple data types for the same target only hits the API once. Similarly, `get_rich_drug_data` benefits from the cache: if target data was previously fetched, the parallel `get_target_data` calls return cached results.

---

## Example Usage

```python
client = OpenTargetsClient()

# Get drug + all target data in one call
rich = await client.get_rich_drug_data("semaglutide")
print(f"Drug: {rich.drug.name}, Targets fetched: {len(rich.targets)}")

# Or get drug data alone
drug = await client.get_drug("semaglutide")
print(f"ChEMBL ID: {drug.chembl_id}")
print(f"Targets: {[t.target_symbol for t in drug.targets]}")

# Get target data for each target
for target in drug.targets:
    target_data = await client.get_target_data(target.target_id)

    # Or use convenience accessors
    associations = await client.get_target_data_associations(target.target_id, min_score=0.5)
    expressions = await client.get_target_data_tissue_expression(target.target_id)
    phenotypes = await client.get_target_data_mouse_phenotypes(target.target_id)
    safety = await client.get_target_data_safety_liabilities(target.target_id)
    constraints = await client.get_target_data_genetic_constraints(target.target_id)

# Drug-specific accessors
indications = await client.get_drug_indications("semaglutide")
competitors = await client.get_drug_competitors("semaglutide")
target_competitors = await client.get_drug_target_competitors("semaglutide")

# Disease-specific accessors
disease_drugs = await client.get_disease_drugs("EFO_0003847")  # NASH
synonyms = await client.get_disease_synonyms("non-alcoholic steatohepatitis")
```
