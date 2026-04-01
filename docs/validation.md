# Validation

## Validated Test Drugs

The agent pipeline will be evaluated against 10 ground-truth repurposings with known outcomes:

| Drug | Original Indication | Known Repurposing | Repurposing Approved | Primary Path |
|------|--------------------|--------------------|----------------------|--------------|
| Baricitinib | Rheumatoid arthritis | Alopecia areata, atopic dermatitis | June 2022 | Path 2 |
| Imatinib | CML | GIST | February 2002 | Path 2 |
| Sildenafil | Erectile dysfunction | Pulmonary arterial hypertension | 2005 | Path 2 |
| Bupropion | Depression | Smoking cessation, ADHD | May 1997 | Path 2 |
| Thalidomide | Sedative | Multiple myeloma | 2006 | Path 2 |
| Rituximab | Lymphoma | Rheumatoid arthritis | 2006 | Path 2 |
| Duloxetine | Depression | Chronic pain, fibromyalgia | June 2008 | Path 2 |
| Metformin | Type 2 diabetes | PCOS | Off-label | Path 2 |
| Colchicine | Gout | Cardiovascular prevention | 2020 | Path 2 |
| Empagliflozin | Type 2 diabetes | Heart failure | February 2022 | Path 2 |

Success metrics: **Recall@10** (known repurposing in top-10 candidates) and **MRR** (mean reciprocal rank).

---

## Prospective Validation Candidates

Active repurposing candidates the pipeline should surface. These are not yet approved for the repurposed indication.

| Drug | Original Indication | Candidate Repurposing | Mechanism Angle | Evidence | Trial |
|------|--------------------|-----------------------|-----------------|----------|-------|
| Metformin | Type 2 diabetes | Ovarian cancer | Cancer stem cell targeting | Phase II: median OS 57.9 months, confirmed impact on cancer stem cells (JCI Insight) | Phase III in development |
| Semaglutide | Type 2 diabetes / obesity | NASH/MASH | GLP-1 receptor agonist | Strong Phase II data, not yet approved for liver indication | — |
| Mebendazole | Antiparasitic | Colorectal cancer | Tubulin inhibition | Phase II/III: 25% tumor regression in refractory CRC (The Medical Advisor) | NCT03925662 |
| Disulfiram | Alcoholism | Glioblastoma | Copper-dependent cytotoxicity | Positive Phase II in recurrent GBM (The Medical Advisor) | NCT03034135 |
| Itraconazole | Antifungal | Non-small cell lung cancer | Hedgehog pathway inhibition | Positive Phase II: 1-year PFS 45% vs 32% historical (The Medical Advisor) | NCT03664115 |
