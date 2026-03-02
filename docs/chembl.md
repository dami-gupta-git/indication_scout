# ChEMBL Client — Data Contracts

Two methods. Returns molecule-level drug properties and ATC classification hierarchies from the ChEMBL REST API.

**API:** ChEMBL REST API (`https://www.ebi.ac.uk/chembl/api/data`)
**Auth:** None required
**Rate limit:** No published limit; treat as a public scholarly API (avoid hammering)
**Versioning:** URL-less versioning — the base URL reflects the current stable release

---

## 1. `get_molecule(chembl_id) → MoleculeData`

Fetch molecule properties by ChEMBL compound ID.

**In:**
- `chembl_id` — canonical ChEMBL compound identifier (e.g. `"CHEMBL1118"`)

**API call:** `GET /molecule/{chembl_id}.json`

**Out:**
- `molecule_chembl_id` — the canonical identifier echoed back (e.g. `"CHEMBL1118"`)
- `molecule_type` — compound classification: `"Small molecule"`, `"Protein"`, `"Antibody"`, `"Oligonucleotide"`, `"Oligosaccharide"`, `"Cell"`, `"Unknown"`
- `max_phase` — highest clinical trial phase reached: `"4"` (approved), `"3"`, `"2"`, `"1"`, `"0.5"` (early Phase 1), `null` (preclinical / no trials)
- `atc_classifications` — list of WHO ATC codes (e.g. `["A10BA02"]`). Empty list if none assigned
- `black_box_warning` — `1` if the drug carries an FDA black box warning, `0` otherwise
- `first_approval` — year of first regulatory approval as int (e.g. `1994`), `null` if not approved
- `oral` — `true` if the drug has an oral formulation

**Response field mapping:**
```
molecule_chembl_id   → molecule_chembl_id
molecule_type        → molecule_type
max_phase            → max_phase
atc_classifications  → atc_classifications
black_box_warning    → black_box_warning
first_approval       → first_approval
oral                 → oral
```

**Error behavior:**
- **Molecule not found:** Raises `DataSourceError` (HTTP 404 propagated from `_rest_get`)
- **Malformed response:** Raises `DataSourceError` with message `"Unexpected response shape for '{chembl_id}'"`
- **Unexpected exception:** Wrapped and re-raised as `DataSourceError`

---

## Data quality notes

**`max_phase` is a string, not an int.** The API returns it as a string (`"4"`, `"3"`, etc.) or `null`. Do not compare as an integer without explicit casting.

**ATC codes are sparse.** Many approved drugs have no ATC code assigned in ChEMBL, especially biologics and recently approved compounds. An empty `atc_classifications` list does not mean the drug lacks a therapeutic class.

**`first_approval` reflects any regulatory body.** This is not necessarily FDA approval — it could be EMA, PMDA, or another agency, whichever came first.

**`molecule_type` matters for mechanism agents.** A `"Small molecule"` has very different developability properties than an `"Antibody"`. The mechanism agent should use this to qualify repurposing feasibility.

---

## 2. `get_atc_description(atc_code) -> ATCDescription`

Fetch the full ATC classification hierarchy for a single ATC code.

**In:**
- `atc_code` -- a WHO ATC code (e.g. `"A10BA02"`)

**API call:** `GET /atc_class/{atc_code}.json`

**Out:** `ATCDescription` with 10 fields:
- `level1` -- Anatomical main group code (e.g. `"A"`)
- `level1_description` -- (e.g. `"ALIMENTARY TRACT AND METABOLISM"`)
- `level2` -- Therapeutic subgroup code (e.g. `"A10"`)
- `level2_description` -- (e.g. `"DRUGS USED IN DIABETES"`)
- `level3` -- Pharmacological subgroup code (e.g. `"A10B"`)
- `level3_description` -- (e.g. `"BLOOD GLUCOSE LOWERING DRUGS, EXCL. INSULINS"`)
- `level4` -- Chemical subgroup code (e.g. `"A10BA"`)
- `level4_description` -- (e.g. `"Biguanides"`)
- `level5` -- Chemical substance code (e.g. `"A10BA02"`)
- `who_name` -- WHO International Nonproprietary Name (e.g. `"metformin"`)

**Caching:** Results are cached under namespace `"atc_description"` with 5-day TTL.

**Error behavior:**
- **ATC code not found:** Raises `DataSourceError`
- **Malformed response:** Raises `DataSourceError` with message `"Unexpected response shape for ATC code '{atc_code}'"`
- **Unexpected exception:** Wrapped and re-raised as `DataSourceError`

---

## Pydantic models

```python
class MoleculeData(BaseModel):
    molecule_chembl_id: str = ""
    molecule_type: str = ""
    max_phase: str | None = None
    atc_classifications: list[str] = []
    black_box_warning: int | None = None
    first_approval: int | None = None
    oral: bool | None = None

class ATCDescription(BaseModel):
    level1: str = ""
    level1_description: str = ""
    level2: str = ""
    level2_description: str = ""
    level3: str = ""
    level3_description: str = ""
    level4: str = ""
    level4_description: str = ""
    level5: str = ""
    who_name: str = ""
```

---

## Data flow from ChEMBL into the pipeline

```
Retrieval Service (build_drug_profile)
  |
  +-- get_atc_description("A10BA02")    # for each ATC code on the drug
  |     -> level3_description, level4_description used in DrugProfile.atc_descriptions
  |
Mechanism Agent (stub)
  |
  +-- get_molecule("CHEMBL1118")        # metformin
        -> molecule_type, max_phase, atc_classifications, black_box_warning
```

---

## Agent-to-method mapping

| Consumer | Method | What it gets |
|----------|--------|--------------|
| **Retrieval service** | `get_atc_description` | Level 3 and level 4 ATC descriptions for `DrugProfile.atc_descriptions` |
| **Mechanism agent** (stub) | `get_molecule` | Molecule type, clinical maturity (`max_phase`), ATC class, safety flag (`black_box_warning`) |