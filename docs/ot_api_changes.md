# Open Targets API Breaking Changes

Date discovered: 2026-03-25

The Open Targets Platform GraphQL API changed significantly. This document maps every
affected type — old fields vs new fields — to guide model and query updates.

---

## Drug object

**GraphQL type:** `Drug`

| Old field | Old type | New field | New type | Notes |
|---|---|---|---|---|
| `isApproved` | `Boolean` | — | — | Removed |
| `maximumClinicalTrialPhase` | `Float` | `maximumClinicalStage` | `String` | Enum: `APPROVAL`, `PHASE_3`, `PHASE_2_3`, `PHASE_2`, `PHASE_1_2`, `PHASE_1`, `EARLY_PHASE_1`, `IND`, `PRECLINICAL`, `PREAPPROVAL`, `UNKNOWN` |
| `yearOfFirstApproval` | `Int` | — | — | Removed, no replacement |
| `id`, `name`, `synonyms`, `tradeNames`, `drugType` | — | unchanged | — | |
| `mechanismsOfAction`, `drugWarnings`, `adverseEvents` | — | unchanged | — | |
| `indications` | `clinicalIndicationsFromDrugImp` | unchanged container | — | Rows changed — see below |

**Our model:** `DrugData`

| Old field | New field |
|---|---|
| `is_approved: bool \| None` | drop — derivable from `maximum_clinical_stage == "APPROVAL"` |
| `max_clinical_phase: float \| None` | `maximum_clinical_stage: str \| None` |
| `year_first_approved: int \| None` | drop — no replacement |

---

## Drug.indications rows

**GraphQL type:** `ClinicalIndicationFromDrug`

| Old field | Old type | New field | New type | Notes |
|---|---|---|---|---|
| `maxPhaseForIndication` | `Float` | `maxClinicalStage` | `String` | Same enum as above |
| `disease { id name }` | — | unchanged | — | |
| `references { source ids }` | — | `clinicalReports { ... }` | `ClinicalReport` | Completely different structure — rich trial-level data |
| — | — | `id` | `String` | New — hash ID for the indication row |
| `approvedIndications` | `[String]` | — | — | Removed from parent container |

**Our model:** `Indication`

| Old field | New field |
|---|---|
| `max_phase: float \| None` | `max_clinical_stage: str \| None` |
| `references: list[dict]` | `clinical_reports: list[ClinicalReport]` — new model (see below) |
| — | `id: str` — new hash ID |

---

## ClinicalReport (new model)

**GraphQL type:** `ClinicalReport`

Replaces the old `references { source ids }` on `Indication`. Rich trial/evidence record.

**Required fields:**

| Field | Type | Notes |
|---|---|---|
| `id` | `String` | Trial/report ID (e.g. NCT number) |
| `source` | `String` | e.g. `AACT`, `FDA` |
| `clinicalStage` | `String` | Same enum as `maxClinicalStage` |
| `hasExpertReview` | `Boolean` | |

**Optional scalar fields:**

| Field | Type |
|---|---|
| `type` | `ClinicalReportType` enum: `CURATED_RESOURCE`, `DRUG_LABEL`, `CLINICAL_TRIAL`, `REGULATORY_AGENCY` |
| `title` | `String` |
| `url` | `String` |
| `year` | `Int` |
| `phaseFromSource` | `String` |
| `trialPhase` | `String` |
| `trialOfficialTitle` | `String` |
| `trialOverallStatus` | `String` |
| `trialStartDate` | `String` |
| `trialWhyStopped` | `String` |
| `trialStudyType` | `String` |
| `trialPrimaryPurpose` | `String` |
| `trialDescription` | `String` |
| `trialNumberOfArms` | `Int` |

**List fields:**

| Field | Item type | Notes |
|---|---|---|
| `drugs` | `ClinRepDrugListItem { drugFromSource: str, drug: { id, name } }` | Drugs involved |
| `diseases` | `ClinicalDiseaseListItem { diseaseFromSource: str, disease: { id, name } \| None }` | Diseases; `disease` can be null |
| `sideEffects` | `ClinicalDiseaseListItem { diseaseFromSource: str, disease: { id, name } \| None }` | Same structure as diseases |
| `countries` | `String` | List of country strings |
| `qualityControls` | `String` | List of strings |
| `trialStopReasonCategories` | `String` | List of strings |
| `trialLiterature` | `String` | List of strings |

---

## Target.knownDrugs → Target.drugAndClinicalCandidates

**GraphQL type:** `ClinicalTargetFromTarget` (was flat `KnownDrug` rows)

Old `knownDrugs` row (flat):

| Field | Type |
|---|---|
| `drugId` | `String` |
| `prefName` | `String` |
| `diseaseId` | `String` |
| `label` | `String` |
| `phase` | `Float` |
| `status` | `String` |
| `mechanismOfAction` | `String` |
| `ctIds` | `[String]` |

New `drugAndClinicalCandidates` row (nested):

| Field | Type | Notes |
|---|---|---|
| `id` | `String` | Hash ID |
| `maxClinicalStage` | `String` | Same enum as above |
| `drug { id, name, drugType, mechanismsOfAction { rows { mechanismOfAction, actionType } } }` | `Drug` | Drug is now a nested object |
| `diseases [ { diseaseFromSource, disease { id, name } } ]` | `[ClinicalDiseaseListItem]` | Multiple diseases per row; `disease` can be null |
| `clinicalReports` | — | Rich trial-level data |
| `status` | — | Removed |
| `ctIds` | — | Removed |

**Our model:** `DrugSummary` — needs significant redesign:

| Old field | Status |
|---|---|
| `drug_id: str` | → `drug.id` |
| `drug_name: str` | → `drug.name` |
| `disease_id: str` | → `diseases[].disease.id` (now a list) |
| `disease_name: str` | → `diseases[].disease.name` (now a list) |
| `phase: float \| None` | → `max_clinical_stage: str \| None` |
| `status: str \| None` | drop — removed |
| `mechanism_of_action: str` | → `drug.mechanismsOfAction.rows[0].mechanismOfAction` |
| `clinical_trial_ids: list[str]` | drop — `ctIds` removed |

---

## Disease.knownDrugs → Disease.drugAndClinicalCandidates

**GraphQL type:** `ClinicalIndicationFromDisease`

| Field | Type | Notes |
|---|---|---|
| `id` | `String` | Hash ID |
| `maxClinicalStage` | `String` | Same enum |
| `drug { id, name, ... }` | `Drug` | Nested drug object |
| `clinicalReports` | — | Rich trial-level data |

Old `knownDrugs` fields `drugId`, `prefName`, `targetId`, `approvedSymbol`, `diseaseId`, `label`, `phase`, `status`, `mechanismOfAction`, `ctIds` are all gone.
