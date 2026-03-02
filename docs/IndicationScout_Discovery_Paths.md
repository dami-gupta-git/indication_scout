# IndicationScout: Drug Repurposing Discovery Paths

## Overview

IndicationScout identifies drug repurposing candidates using existing API integrations (Open Targets, PubMed, ClinicalTrials.gov) through multiple discovery paths.

---

## Discovery Paths

### Path 1: Drug Class Analogy (from Open Targets)

Query Open Targets for other drugs sharing the same mechanism/target, collect their indications, then subtract the query drug's current indications.

**Flow:**
1. Get your drug's targets (already available via `mechanismsOfAction`)
2. For each target, query `knownDrugs`
3. Collect all (other_drug, indication) pairs
4. Group by indication
5. Subtract your drug's current indications
6. Rank by: how many different drugs treat this indication via your targets

**Key Insight:** This path uses clinical evidence (what drugs are actually approved for), not computed association scores. A candidate that shows up through 3 different drugs on 2 of your targets is stronger than one that shows up once.

### Path 2: Adverse Events Signal (from Open Targets)

Open Targets includes FAERS adverse event data on drug nodes. Use an LLM to map adverse events to therapeutic opportunities (e.g., "causes drowsiness" → potential sleep aid).

### Path 3: Literature-Driven Discovery (from PubMed RAG)

Search PubMed for "[drug name] repurposing" or "[drug name] new indication" and use RAG to extract candidate diseases from the literature. This path catches discoveries that structured data misses.

---

## Multi-Path Architecture

The discovery paths are designed to stack:

- **Path 1 (target-disease associations):** Biology-level signal
- **Path 2 (drug class analogy):** Clinical-level signal
- **Path 3 (literature):** Research-level signal

A candidate that appears in multiple paths is a strong repurposing candidate.

---

## Candidate Ranking

Discovery paths cast a wide net. Downstream agents narrow results:

```
Discovery (wide)  →  Literature Agent (filter)  →  Clinical Trial Agent (validate)
100+ candidates      "Is there published           "Is anyone actually testing this?"
from knownDrugs      evidence for this?"
                     Keeps ~20 with PubMed         Flags ~10 with active trials
                     support
```

### Ranking Signals

| Signal | Interpretation |
|--------|----------------|
| Multiple drugs in class treat this disease | Higher confidence |
| PubMed abstracts exist for drug + disease | Evidence exists |
| Clinical trials exist for drug + disease | Someone's testing it |
| NOT already the drug's current indication | It's actually new |

---

## Ground Truth: Validated Repurposing Cases

These 15 well-documented cases can be used to test discovery logic:

| Drug | Original Indication | Repurposed Indication | Target Gene | Why OT Should Find It |
|------|---------------------|----------------------|-------------|----------------------|
| Thalidomide | Sedative | Multiple myeloma | CRBN, TNF | TNF → myeloma association |
| Sildenafil | Angina | Erectile dysfunction, PAH | PDE5A | PDE5A → pulmonary hypertension |
| Imatinib | CML | GIST | KIT, ABL1, PDGFRA | KIT → GIST strong association |
| Rituximab | Lymphoma | Rheumatoid arthritis | MS4A1 (CD20) | CD20 → RA via autoimmune |
| Metformin | Type 2 Diabetes | PCOS | PRKAB1 (AMPK) | Metabolic pathway overlap |
| Duloxetine | Depression | Chronic pain, fibromyalgia | SLC6A4, SLC6A2 | Norepinephrine → pain pathways |
| Bupropion | Depression | Smoking cessation | SLC6A3 | Dopamine reuptake → addiction |
| Raloxifene | Osteoporosis | Breast cancer prevention | ESR1 | ESR1 → breast cancer strong |
| Colchicine | Gout | CV prevention post-MI | TUBB | Inflammation → CV disease |
| Baricitinib | Rheumatoid arthritis | Alopecia areata, atopic dermatitis | JAK1, JAK2 | JAK → autoimmune broad |
| Topiramate | Epilepsy | Migraine | GRIA1, GABRA1 | Neuro targets → migraine |
| Finasteride | BPH | Hair loss | SRD5A2 | 5-alpha reductase → androgenic |
| Empagliflozin | Type 2 Diabetes | Heart failure | SLC5A2 | SGLT2 → cardiac associations |
| Semaglutide | Type 2 Diabetes | Obesity | GLP1R | GLP1R → metabolic broad |
| Tamoxifen | Breast cancer treatment | Breast cancer prevention | ESR1 | Same target, different indication stage |

---

## Validation Examples

### Test 1: Bupropion (CHEMBL894)

- **Target:** SLC6A3 (ENSG00000075120)
- **Query:** `knownDrugs` on ENSG00000075120
- **Expected drugs:** methylphenidate, modafinil, solriamfetol
- **Expected candidates:** ADHD, narcolepsy (confirmed—bupropion has been studied for ADHD)

### Test 2: Imatinib (CHEMBL941)

- **Targets:** ABL1, KIT, PDGFRA
- **Query:** `knownDrugs` on KIT (ENSG00000157404)
- **Expected drugs:** sunitinib, regorafenib, ripretinib
- **Expected candidates:** GIST (imatinib's known repurposing success)

### Test 3: Sildenafil (CHEMBL1233)

- **Target:** PDE5A (ENSG00000138735)
- **Query:** `knownDrugs` on PDE5A
- **Expected drugs:** tadalafil, vardenafil
- **Expected candidates:** Pulmonary arterial hypertension

### Test 4: Baricitinib (CHEMBL3707348)

- **Targets:** JAK1, JAK2
- **Query:** `knownDrugs` on JAK1 (ENSG00000162434)
- **Expected drugs:** tofacitinib, ruxolitinib, upadacitinib
- **Expected candidates:** Broad list across RA, alopecia, atopic dermatitis, myelofibrosis, polycythemia vera

---

## Prerequisites

The following APIs are assumed to be implemented:

- Open Targets (drug data, target data, `knownDrugs` queries)
- PubMed (search and RAG pipeline)
- ClinicalTrials.gov (trial lookup)
