# Metformin — miss analysis

Source: `snapshots/metformin_2026-09-03_22-06-30.md`, run with
`OPEN_TARGETS_COMPETITOR_PREFETCH_MAX=40`, `LITERATURE_TOP_K=15`,
`SUPERVISOR_CANDIDATE_CAP=15`, `SUPERVISOR_INVESTIGATION_CAP=3`,
`MECHANISM_TOP_CANDIDATES=5`, `CLINICAL_TRIALS_LANDSCAPE_MAX_TRIALS=50`.

The report lists 18 diseases under **Diseases Considered** — 15 from the competitor path plus 3
promoted by the mechanism agent — and deep-dives 3: coronary artery disorder, polycystic ovary
syndrome and metabolic dysfunction-associated steatotic liver disease.

Ranks below come from the cached Open Targets sibling ranking (`cache/competitors_raw/`,
`CHEMBL1431`, `min_stage=PHASE_3`) and from the post-merge list in `cache/competitors_merged/`.

## The ranking carries no signal for this drug

Every one of the 40 diseases in metformin's sibling ranking has exactly one competitor drug, and in
all 40 cases that drug is **metformin itself**.

Metformin's Open Targets targets are the 51 subunits of mitochondrial complex I plus GPD2. No other
drug in Open Targets acts on those targets, so the competitor search returns no rivals at all. What
the pipeline calls a competitor ranking is metformin's own Open Targets disease list.

Two consequences:

The sort key is a count of rival drugs, so all 40 diseases tie at 1. Python's sort is stable, so the
order that survives is the order the targets happened to be iterated in — arbitrary with respect to
anything clinical. Which three diseases reach the deep dive is chance. Coronary artery disorder,
PCOS and MASLD are all defensible candidates, but nothing in the pipeline selected them on merit.

`get_drug_competitors` in `data_sources/open_targets.py` does not exclude the drug being analysed
from its own competitor list. Every count is inflated by one, and for a drug with no true rivals the
list becomes entirely self-referential. This affects every drug: baricitinib's alopecia areata entry
counts baricitinib among its 8.

## Where candidates are lost

| Stage | Setting | At run time |
| --- | --- | --- |
| Sibling diseases prefetched from Open Targets | `OPEN_TARGETS_COMPETITOR_PREFETCH_MAX` | 40 |
| Kept after the LLM merges duplicate disease names | `LITERATURE_TOP_K` | 15 |
| Kept in the supervisor's final ranked list | `SUPERVISOR_CANDIDATE_CAP` | 15 |
| Mechanism candidates appended on top | `MECHANISM_TOP_CANDIDATES` | 5 |
| Deep-dived by the supervisor | `SUPERVISOR_INVESTIGATION_CAP` | 3 |
| Completed trials fetched per pair | `CLINICAL_TRIALS_LANDSCAPE_MAX_TRIALS` | 50 |

## Errors

| Error | Reason | Setting to change |
| --- | --- | --- |
| The 15 uninvestigated candidates are reported as "Evidence gate exclusions" | The gate is defined in `prompts/supervisor.txt` as zero trials AND no relevant literature. These 15 were never investigated, so they have zero trials and zero PMIDs by construction. The label tells the reader they were checked and found empty. For prediabetes, obesity, breast cancer and gestational diabetes that is the opposite of what the evidence shows. | None — prompt and reporting bug. The footer must exclude candidates that were never investigated |
| Prediabetes never investigated | Sibling rank 12, post-merge position 9, outside the deep-dive cap of 3. Metformin for diabetes prevention is its best-evidenced repurposing use. | `SUPERVISOR_INVESTIGATION_CAP` 3 → 16 |
| Gestational diabetes, breast cancer, miscarriage, preeclampsia, sepsis never investigated | Post-merge positions 12–15 plus one mechanism promotion, all outside the cap of 3. | `SUPERVISOR_INVESTIGATION_CAP` 3 → 16 |
| Type 1 diabetes (adjunct), atherosclerosis, prostate cancer, colorectal cancer, chronic kidney disease, endometrial cancer, hidradenitis suppurativa never listed | Sibling ranks 19–39, inside the prefetch of 40 but dropped by the cuts to 15. | `LITERATURE_TOP_K` 15 → 60 and `SUPERVISOR_CANDIDATE_CAP` 15 → 60 |
| Aging / healthspan never listed | Sibling rank 37, dropped by the same cuts to 15. The TAME trial makes this metformin's most-discussed repurposing hypothesis. | `LITERATURE_TOP_K` 15 → 60 and `SUPERVISOR_CANDIDATE_CAP` 15 → 60 |
| Diabetes mellitus appears as a repurposing candidate | It is an approved indication for metformin and was correctly stripped from the competitor list. It re-entered through the mechanism path, which adds candidates to the allowlist in `_merge_and_dedup_impl` with no approved-indication filter. | None — the approved strip must be applied to mechanism candidates too |
| The candidate list is filled with overlapping terms: coronary artery disorder, cardiovascular disorder and heart failure; obesity and metabolic syndrome; prediabetes syndrome and insulin resistance | The hierarchical super/subtype dedup is commented out in `_merge_and_dedup_impl`, disabled because it was collapsing actionable subtypes such as PCOS into broad parents for exactly this kind of broadly-acting drug. Only exact-name merging runs. | None — needs the equivalence-group approach the disabled code notes |
| PCOS trial evidence drawn from a partial sample: 48 relevant of the first 50 fetched, out of 111 completed trials on record | The completed-trial fetch is capped per pair. | `CLINICAL_TRIALS_LANDSCAPE_MAX_TRIALS` 50 → 120 for a drug with this much trial volume; costs proportionally more fetching |
| Boxed warning list wrong: metformin's label carries one boxed warning, for lactic acidosis. The report claims "multiple black-box warnings (respiratory, cardiac, vascular, and metabolic toxicity)". | Adverse-event report categories mixed with actual label warnings. The same fault appears in the baricitinib report. | None — data handling bug |

## What the report got right

The three deep-dived candidates are all real metformin repurposing stories, and the literature
sections are accurate and well-cited, including the mixed and contradicting evidence for PCOS and
MASLD rather than only the supportive trials. The trial contamination filtering is thorough — 17
excluded for coronary artery disorder and 26 for MASLD, each with a stated rule — and correctly
catches trials where metformin is background therapy rather than the studied agent. Terminated
trials are correctly attributed to recruitment and funding rather than safety.

## Settings worth changing

`.env.constants` was raised after this run (uncommitted) to `LITERATURE_TOP_K=60`,
`SUPERVISOR_CANDIDATE_CAP=60`, `SUPERVISOR_INVESTIGATION_CAP=16`,
`OPEN_TARGETS_COMPETITOR_PREFETCH_MAX=60`. Against this report those changes recover prediabetes,
gestational diabetes, breast cancer, miscarriage, preeclampsia and sepsis into the deep dive, and
bring type 1 diabetes, atherosclerosis, the cancers, chronic kidney disease, hidradenitis and aging
into the candidate list. Aging stays outside the deep dive at sibling rank 37.

`CLINICAL_TRIALS_LANDSCAPE_MAX_TRIALS` is unchanged at 50 and still truncates the PCOS trial record.

## What the settings do not fix

For metformin the ranking is a 40-way tie decided by iteration order, so raising the caps buys
coverage without buying selection. Nothing improves the odds that the deep dive spends its budget
well. The self-competitor bug, the approved-indication leak through the mechanism path, the
evidence-gate mislabelling and the boxed-warning mix-up are all code or prompt defects with no
setting attached.
