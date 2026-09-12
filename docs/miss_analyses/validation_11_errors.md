# Validation 11 — miss analysis

Source: `results/holdout_validation/validation_results_11.md`, run with
`SUPERVISOR_CANDIDATE_CAP=15`, `SUPERVISOR_INVESTIGATION_CAP=3`,
`MECHANISM_ASSOCIATIONS_PER_TARGET=30`.

In that file `1` means the target indication reached the deep dive, `0` means it was in the merged
candidate list but ranked below the investigation cap, and `-1` means it never entered the candidate
list. There are 8 hits, 17 rows at `0` and 10 rows at `-1`.

Ranks below come from `scripts/validation/probe_candidates.py`, which reports the Open Targets
sibling ranking before truncation. The ranking is not date-filtered: `date_before` only suppresses
the approved-indication strip, so a sibling drug that reached phase 3 after the cutoff still counts.

## Where candidates are lost

The competitor path narrows three times, each governed by a setting in `.env.constants`:

| Stage | Setting | Now |
| --- | --- | --- |
| Sibling diseases prefetched from Open Targets | `OPEN_TARGETS_COMPETITOR_PREFETCH_MAX` | 40 |
| Kept after the LLM merges duplicate disease names | `LITERATURE_TOP_K` | 15 |
| Deep-dived by the supervisor | `SUPERVISOR_INVESTIGATION_CAP` | 3 |

The middle cut is the binding constraint. `get_drug_competitors` in `services/retrieval.py` trims
the merged list with `LITERATURE_TOP_K`. Despite the name, that setting is used nowhere else — it
does not affect literature retrieval, so raising it only widens the competitor list.

`SUPERVISOR_CANDIDATE_CAP` (15) trims the supervisor's final ranked list, which is where the
competitor list and the mechanism candidates have already been merged.

## Score -1 — never in the candidate list

| Drug | Indication | Reason | Setting to change |
| --- | --- | --- | --- |
| everolimus | advanced renal cell carcinoma | Renal cell carcinoma ranks #0 and "kidney cancer" is in the merged list. The matcher is instructed to reject a broader parent, so it returned no match. | None — not a retrieval miss; the row is probably mislabelled |
| everolimus | progressive pancreatic neuroendocrine tumors | "Neuroendocrine neoplasm" ranks #25, inside the prefetch of 40 but dropped by the cut to 15. Pancreatic neuroendocrine tumor itself ranks #238. | `LITERATURE_TOP_K` 15 → 30, and `SUPERVISOR_CANDIDATE_CAP` to match |
| rituximab | pemphigus vulgaris | Pemphigus ranks #58 of 228, just outside the prefetch of 40. | `OPEN_TARGETS_COMPETITOR_PREFETCH_MAX` 40 → 60, `LITERATURE_TOP_K` 15 → 60, `SUPERVISOR_CANDIDATE_CAP` 15 → 60 |
| imatinib | dermatofibrosarcoma protuberans | Ranks #163 of 387 with 2 competitor drugs. | `OPEN_TARGETS_COMPETITOR_PREFETCH_MAX` 40 → 170 with `LITERATURE_TOP_K` and `SUPERVISOR_CANDIDATE_CAP` to match; not applied |
| colchicine | familial mediterranean fever | Ranks #465 of 540 with 1 competitor drug. Colchicine's Open Targets targets are tubulin, so the ranking is dominated by oncology. | None — rank is far beyond any workable prefetch |
| thalidomide | erythema nodosum leprosum | Absent from the sibling ranking entirely. | None |
| duloxetine | diabetic peripheral neuropathic pain | No competitor or mechanism hit. | None |
| colchicine | atherosclerotic cardiovascular disease | No Open Targets link to the anti-inflammatory mechanism. | None |
| propranolol | proliferating infantile hemangioma | Serendipitous clinical discovery, no Open Targets signal. | None |
| everolimus | subependymal giant cell astrocytoma | Everolimus's only Open Targets target is FKBP1A, and astrocytoma is absent from its top 400 associations. The mechanism path never held it. The "rank 22" in the original note is not a position in this ranking. | None |

## Score 0 — in the list, below the investigation cap

All 17 rows have the same cause: the disease was in the merged list but ranked below 3.

| Rank | Rows |
| --- | --- |
| 4 | empagliflozin / chronic kidney disease, everolimus / HR+ HER2- breast cancer |
| 5 | duloxetine / fibromyalgia |
| 6 | tadalafil / benign prostatic hyperplasia, topiramate / migraine prophylaxis |
| 7 | baricitinib / alopecia areata |
| 8 | bupropion / smoking cessation, topiramate / alcohol dependence |
| 10 | duloxetine / generalized anxiety disorder, methotrexate / psoriasis |
| 11 | semaglutide / cardiovascular risk reduction |
| 12 | imatinib / myelodysplastic syndrome, minoxidil / androgenetic alopecia |
| 14 | rituximab / rheumatoid arthritis |
| 16 | imatinib / myeloproliferative neoplasm, imatinib / hypereosinophilic syndrome, imatinib / aggressive systemic mastocytosis |

All 17 are governed by `SUPERVISOR_INVESTIGATION_CAP`. Setting it to 8 recovers 8 rows, 12 recovers
13, and 16 recovers all 17. Each recovered row costs one literature and one clinical-trials fan-out.
The three rows at rank 16 also sit outside `SUPERVISOR_CANDIDATE_CAP` (15) and need that raised to
20 as well.

## Settings worth changing

These values are now set in `.env.constants`.

| Setting | Was | Now | What it changes |
| --- | --- | --- | --- |
| `SUPERVISOR_INVESTIGATION_CAP` | 3 | 16 | Deep-dives all 17 rows that scored 0. Costs 16 literature and 16 clinical-trials fan-outs per run instead of 3. |
| `OPEN_TARGETS_COMPETITOR_PREFETCH_MAX` | 40 | 60 | Brings pemphigus (rank 58) inside the prefetch. Beyond 60 the sibling evidence is one or two drugs per disease. |
| `LITERATURE_TOP_K` | 15 | 60 | Keeps the whole prefetch through the merge, so neuroendocrine neoplasm (rank 25) and pemphigus survive. |
| `SUPERVISOR_CANDIDATE_CAP` | 15 | 60 | Without this the wider competitor list is trimmed straight back to 15 and the other two changes do nothing. |

The two recovered `-1` rows move to `0`, not to `1`: pemphigus and neuroendocrine neoplasm reach the
candidate list at positions well past 16, so they are surfaced but not deep-dived. Reaching `1` would
need an investigation cap near 60.

The mechanism candidate path has its own two settings, also now raised:

| Setting | Was | Now | What it changes |
| --- | --- | --- | --- |
| `MECHANISM_ASSOCIATIONS_PER_TARGET` | 15 | 30 | Associations kept per target before pooling. Validation 11 was run at 30 while the config said 15; this aligns them. |
| `MECHANISM_TOP_CANDIDATES` | 5 | 25 | The final trim after direction and approval filtering. |

`scripts/validation/probe_mechanism.py` now reports where a disease sits in both cuts. It shows the
everolimus / SEGA justification for these values does not hold: the mechanism path never had
astrocytoma. It also shows the effect of the top-candidates change varies by drug. For everolimus
only 6 rows survive the direction filter, so the limit does not bind at either 5 or 25. For imatinib
193 rows survive, so 25 surfaces twenty more candidates than 5 did.

`MECHANISM_SIGNAL_THRESHOLD` and `MECHANISM_ASSOCIATIONS_CAP` were left alone — they apply only
inside a tool the mechanism agent calls during its reasoning loop, not to the candidates that reach
the supervisor.

Not applied: imatinib / DFSP needs a prefetch around 170. Six of the ten `-1` rows are not
reachable from Open Targets at any setting.

## Matcher inconsistency

The matcher in `scripts/validation/gen_seed_candidate_recall.py` is told not to match a broader
parent, but the run accepted "heart failure", "leukemia", "alopecia" and "renal insufficiency" as
matches while rejecting "kidney cancer" for advanced renal cell carcinoma. Some labels in the file
are unreliable in both directions as a result.
