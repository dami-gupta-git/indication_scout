# Contamination

"Contamination" is IndicationScout's umbrella term for retrieval noise that must be excluded before
counting evidence toward a drug×disease repurposing candidate. It is not comorbidity per se — it
covers any hit (a clinical trial, an FDA-label mapping, or a PubMed abstract) that superficially
matches a search but is actually about the wrong drug, a wrong/distinct disease, an already
FDA-approved indication, or the drug being used for a comorbidity rather than the target disease.
Contamination is computed independently at three levels — per-NCT in clinical trials, per-candidate
disease against the FDA approval mapping, and per-PMID in literature synthesis — and in every case
flows downstream into filtering (excluded from signals/evidence counts) and into user-facing
"N hidden/excluded" disclosure notes in the final report.

## 1. Clinical trials: NCT-level contamination

**Definition** (`src/indication_scout/prompts/clinical_trials.txt:22-92`): a trial is "contaminated"
(vs. "relevant") when, for the drug×indication pair under investigation, the trial actually studies:

- **TEST 0** — a different drug (this drug only a comparator/background arm),
  `clinical_trials.txt:37-54`
- **TEST 1** — an already-**approved** indication of the drug, or a narrower subtype of one
  (already-approved evidence, not repurposing), `clinical_trials.txt:58-69`
- **TEST 2** — a genuinely **distinct disease** or wrong therapeutic intent (e.g. drug used for a
  comorbidity in patients who happen to have the target disease), `clinical_trials.txt:80-92`

**Computed by**: `finalize_analysis` tool in
`src/indication_scout/agents/clinical_trials/clinical_trials_tools.py:577-650`. The LLM emits one
`{"nct": ..., "verdict": "relevant"|"contaminated"}` per shown trial; the tool splits these into
`relevant_ncts` / `contaminated_ncts` (`clinical_trials_tools.py:635-647`) and rejects
incomplete/unknown verdict sets (`clinical_trials_tools.py:597-633`).

**Wired into the output artifact**:
`src/indication_scout/agents/clinical_trials/clinical_trials_agent.py:206-213` —
`ClinicalTrialsOutput.relevant_nct_ids` / `contaminated_nct_ids` populated from
`finalize.relevant_ncts` / `finalize.contaminated_ncts`.

**Downstream use**:

- **Deterministic trial signals**: `derive_trial_signals()` in
  `src/indication_scout/agents/_trial_signals.py:125-169` filters completed/terminated trials to the
  relevant set (`_filter_relevant`, line 116-122) and excludes contaminated NCTs (best-effort) from
  the active-Phase-3 search-set read (`search_excluded`, lines 154-163). Called at
  `clinical_trials_agent.py:221-226`.
- **Report rendering**: `src/indication_scout/report/format_report.py:141-145,207-211,240` filters
  `contaminated_nct_ids` out of the example-trials table shown to the user, and surfaces a "N hidden
  as a different indication" disclosure note.
- **Dev-stage judgment**: `src/indication_scout/services/dev_stage.py:9,412` — only
  relevance/contamination-filtered trials are sent to the LLM dev-stage judge.

## 2. FDA approval-relationship contamination (candidate-level)

**Definition**: a candidate disease itself is labeled `"contaminated"` (one of
`ApprovalLabel = Literal["approved","combination_only","contaminated","none"]`,
`src/indication_scout/services/approval_check.py:50`) when it's a real, distinct repurposing target
but a registry/literature search for it would pull in the drug's *already-approved* trials/evidence —
e.g. broad umbrella term vs. an approved narrower subtype, or a sibling disease sharing a search term
with an approved indication (`src/indication_scout/prompts/extract_fda_approval_single.txt:22-34`).

**Computed by**: `get_fda_approved_disease_mapping()` in
`src/indication_scout/services/approval_check.py:454-501` — curated short-circuit against
`CURATED_FDA_CONTAMINATED_CANDIDATES` (`src/indication_scout/constants.py:429-438`) first, else LLM
classification.

**Downstream use**: `supervisor_output.py:100-104` (`approval_relationship` field), consumed by
`supervisor_tools.py:259,388,743-866` to demote/annotate candidates and count
`contaminated_nct_ids` for the report note (`supervisor_tools.py:863-866`); rendered in
`format_report.py:161` and mapped to user-facing prose in `judge_interpretive.py:125-157` (internal
label never shown verbatim to the user).

## 3. Literature/PubMed evidence contamination (per-PMID)

**Definition** (`src/indication_scout/prompts/synthesize.txt:16-37`): an abstract is "contaminated"
(vs. supporting/contradicting/mixed) when it's about a different drug, this drug but a different
disease, this drug studied for a different therapeutic intent (comorbidity) in patients with the
target disease, or the drug's already-approved sub-indication.

**Computed by**: `synthesize()` flow in `src/indication_scout/services/retrieval.py:887-956`. The
LLM's `verdicts` map classifies each input PMID; any PMID it omits defaults to `"contaminated"`
(conservative — `retrieval.py:888-889,912`). `relevant_pmids` = anything not contaminated
(`_RELEVANT_VERDICTS = {supporting, contradicting, mixed, neutral}`, line 899);
`contaminated_pmids` built at `retrieval.py:940-942,956`.

**Downstream use**: `model_evidence_summary.py:34-55` stores `contaminated_pmids` on the
evidence-summary model; excluded abstracts are dropped from evidence grading and renderers show an
"N excluded" note.
