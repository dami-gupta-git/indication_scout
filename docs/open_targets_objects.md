# Open Targets — Data Model Reference

18 Pydantic models total. Two primary objects (`DrugData`, `TargetData`) returned by the two primary client methods. Everything else is nested inside them or returned standalone.

---

## Primary objects

| Model | Returned by |
|---|---|
| `DrugData` | `get_drug(drug_name)` |
| `TargetData` | `get_target_data(target_id)` |

**`DrugData`** — drug-centric. You query by drug name and get everything about that drug: its ChEMBL ID, approval status, indications (diseases it's been trialed for), black box warnings, adverse events, and a list of `DrugTarget` objects (its molecular targets + MoA).

**`TargetData`** — target-centric. You query by target (e.g. GLP1R) and get everything about that protein: disease associations, pathways, tissue expression, mouse phenotypes, safety liabilities, and a list of `DrugSummary` objects (all drugs that hit this target).

**`DrugSummary`** — a lightweight, read-only view of a drug *from the target's perspective*. It lives inside `TargetData.drug_summaries` and contains just the fields relevant to that target relationship: drug name, disease, clinical phase, status, MoA, and trial IDs. It's not queried directly — it's populated as part of a `get_target()` call.

The shared key between them is the ChEMBL ID (`DrugData.chembl_id` == `DrugSummary.drug_id`). `RichDrugData` combines a `DrugData` with its full `TargetData` list for agents that need both sides at once.

---

## Models nested inside `DrugData`

| Model | Field | Description |
|---|---|---|
| `DrugTarget` | `targets` | Lightweight target reference (id, symbol, MoA, action type) |
| `Indication` | `indications` | Disease + max phase for this drug |
| `DrugWarning` | `warnings` | Black box warnings and withdrawals |
| `AdverseEvent` | `adverse_events` | FAERS signals (also used inside `DrugData`) |

---

## Models nested inside `TargetData`

| Model | Field | Description |
|---|---|---|
| `Association` | `associations` | Target-disease links with per-datatype evidence scores |
| `Pathway` | `pathways` | Reactome pathways the target participates in |
| `Interaction` | `interactions` | Protein-protein interaction partners |
| `DrugSummary` | `drug_summaries` | Drugs acting on this target, with indication and phase |
| `TissueExpression` | `expressions` | RNA + protein expression per tissue |
| `MousePhenotype` | `mouse_phenotypes` | Knockout/knock-in mouse phenotypes |
| `SafetyLiability` | `safety_liabilities` | Known target safety concerns |
| `GeneticConstraint` | `genetic_constraint` | gnomAD LOF intolerance scores |

---



## Sub-models (nested inside the above)

| Model | Lives inside | Description |
|---|---|---|
| `RNAExpression` | `TissueExpression.rna` | TPM value, quantile, unit |
| `ProteinExpression` | `TissueExpression.protein` | Level, reliability, cell types |
| `CellTypeExpression` | `ProteinExpression.cell_types` | Per-cell-type protein expression |
| `BiologicalModel` | `MousePhenotype.biological_models` | Allelic composition, genetic background, PMIDs |
| `SafetyEffect` | `SafetyLiability.effects` | Direction and dosing conditions |

---

## Composite objects

| Model | Returned by | Description |
|---|---|---|
| `RichDrugData` | `get_rich_drug_data(drug_name)` | `DrugData` + `list[TargetData]` for all of the drug's targets, fetched in parallel |

`RichDrugData` fields:
- `drug: DrugData` — full drug node
- `targets: list[TargetData]` — one `TargetData` per `DrugTarget` in `drug.targets`, in the same order

---

## Standalone (returned directly by methods)

| Model | Returned by | Description |
|---|---|---|
| `DiseaseSynonyms` | `get_disease_synonyms(disease_name)` | Exact, related, narrow, broad synonyms + parent names |

---

*Note: `DiseaseDrug` was removed — it was superseded by `DrugSummary`, which `get_disease_drugs()` now returns.*

---

## Key relationships

- `DrugTarget.target_id` → pass to `get_target_data()` to get the full `TargetData`
- `DrugSummary.drug_id` (ChEMBL ID) ↔ `DrugData.chembl_id` — shared key between the drug-centric and target-centric views
- `DrugData.approved_disease_ids` — set of disease IDs with phase ≥ 4 (derived from `indications`)
- `DrugData.investigated_disease_ids` — set of all disease IDs in `indications` (any phase)
- `DiseaseSynonyms.all_synonyms` — property returning `exact + related + parent_names` combined
