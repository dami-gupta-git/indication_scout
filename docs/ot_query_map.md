# Open Targets GraphQL Query Map

Field map for each query after the API migration. Reflects the current state of
queries in `data_sources/open_targets.py`.

---

## SEARCH_QUERY

```
search(drug)
└── hits [ { id, entity } ]
```

## DISEASE_SEARCH_QUERY

```
search(disease)
└── hits [ { id, entity } ]
```

## DRUG_QUERY

```
drug
├── id, name, synonyms, tradeNames, drugType
├── maximumClinicalStage
├── mechanismsOfAction.rows[]
│   ├── mechanismOfAction, actionType
│   └── targets[] { id, approvedSymbol }
├── indications.rows[]
│   ├── id, maxClinicalStage
│   ├── disease { id, name }
│   └── clinicalReports[]
│       ├── id, source, clinicalStage, hasExpertReview
│       ├── title, type, trialOverallStatus, trialLiterature
│       ├── drugs[] { drugFromSource, drug { id, name } }
│       └── diseases[] { diseaseFromSource, disease { id, name } }
├── drugWarnings[]
│   └── warningType, description, toxicityClass, country, year, efoId, efoTerm
└── adverseEvents
    ├── criticalValue
    └── rows[] { name, meddraCode, count, logLR }
```

## TARGET_QUERY

```
target
├── id, approvedSymbol, approvedName
├── associatedDiseases.rows[]
│   ├── disease { id, name, therapeuticAreas[] { id, name } }
│   ├── score
│   └── datatypeScores[] { id, score }
├── pathways[] { pathwayId, pathway, topLevelTerm }
├── interactions.rows[]
│   ├── intB, intBBiologicalRole, score, sourceDatabase, count
│   └── targetB { id, approvedSymbol }
├── drugAndClinicalCandidates.rows[]
│   ├── id, maxClinicalStage
│   ├── drug
│   │   ├── id, name, drugType
│   │   └── mechanismsOfAction.rows[]
│   │       ├── mechanismOfAction, actionType
│   │       └── targets[] { id, approvedSymbol }
│   ├── diseases[] { diseaseFromSource, disease { id, name } }
│   └── clinicalReports[]
│       ├── id, source, clinicalStage, hasExpertReview
│       ├── title, type, trialOverallStatus, trialLiterature
│       ├── drugs[] { drugFromSource, drug { id, name } }
│       └── diseases[] { diseaseFromSource, disease { id, name } }
├── expressions[]
│   ├── tissue { id, label, anatomicalSystems }
│   ├── rna { value, unit, level }
│   └── protein { level, reliability, cellType[] { name, level, reliability } }
├── mousePhenotypes[]
│   ├── modelPhenotypeId, modelPhenotypeLabel
│   ├── modelPhenotypeClasses[] { id, label }
│   └── biologicalModels[] { allelicComposition, geneticBackground, id, literature }
├── safetyLiabilities[]
│   ├── event, eventId, datasource, literature, url
│   └── effects[] { direction, dosing }
└── geneticConstraint[]
    └── constraintType, score, exp, obs, oe, oeLower, oeUpper, upperBin, upperBin6
```

## ASSOCIATIONS_PAGE_QUERY

```
target
└── associatedDiseases.rows[]
    ├── disease { id, name, therapeuticAreas[] { id, name } }
    ├── score
    └── datatypeScores[] { id, score }
```

## DISEASE_DRUGS_QUERY

```
disease
└── drugAndClinicalCandidates.rows[]
    ├── id, maxClinicalStage
    ├── drug
    │   ├── id, name, drugType
    │   └── mechanismsOfAction.rows[]
    │       ├── mechanismOfAction, actionType
    │       └── targets[] { id, approvedSymbol }
    └── clinicalReports[]
        ├── id, source, clinicalStage, hasExpertReview
        ├── title, type, trialOverallStatus, trialLiterature
        ├── drugs[] { drugFromSource, drug { id, name } }
        └── diseases[] { diseaseFromSource, disease { id, name } }
```

## DISEASE_SYNONYMS_QUERY

```
disease
├── id, name
├── parents[] { name }
└── synonyms[] { relation, terms }
```
