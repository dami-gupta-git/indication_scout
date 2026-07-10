# Harness tests

Standalone, run-by-hand harnesses used to prototype and gate prompt/agent changes before wiring
them into the pipeline. Each is a `python …_harness.py` script (not collected by pytest). Most call
real LLMs and/or live data sources; some are pure/deterministic. Run N times where drift, not
ignorance, is the failure mode.

## Layout

- `data/` — test fixtures read by the harnesses (`sild_htn_trials.json`).
- `prompts/` — candidate prompt files (`adverse_event_synthesis_prompt.txt`, `fact_judge_prompt.txt`).
- `output/` — captured run artifacts (`ae_sweep_out.txt`).

## Harnesses

### Synthesis / literature grading
- **adverse_event_synthesis_harness.py** — sweeps the seed corpus running current vs candidate
  `synthesize.txt` (adds rule #5: a bare adverse-event-list disease mention is not evidence for/against
  repurposing), reporting only pairs whose verdict diverges. Catches the intended fix and regressions.
- **animal_only_synthesis_harness.py** — B3 gate: can the synthesize call also emit a reliable
  `is_animal_only` flag from the abstracts it already reads (all-animal vs ≥1 human vs no-evidence)?
  Pulls real abstracts from pgvector.
- **literature_strength_harness.py** — isolated call: grades DRUG-SPECIFIC literature strength without
  inflating to "strong" on class-level (other-drug) RCTs. `evidence_basis="class_level"` vs `drug_specific`.
  Real abstracts by PMID.
- **approval_aware_literature_harness.py** — extension to `judge_literature_strength`: papers on an
  APPROVED sub-indication of a broad candidate must not count toward its strength (`evidence_basis="approved"`).

### Clinical-trials relevance / summary
- **trial_relevance_harness.py** — signal-ablation for per-trial relevant-vs-contaminated tagging
  (sildenafil × systemic HTN): mesh-only baseline vs title+interventions+summary. Reads `data/sild_htn_trials.json`.
- **trial_relevance_intent_harness.py** — TEST 2 therapeutic-intent clause: a trial that names the
  candidate only as the study POPULATION (not treatment target) is contamination.
- **trial_relevance_approved_subtype_harness.py** — TEST 1 approved-subtype clause after the multi-condition
  refactor: a trial contaminates if ANY listed condition is the approved subtype (semaglutide × NAFLD anchor).
- **approval_aware_trials_harness.py** — CT relevance gate: a trial about an APPROVED sub-indication of
  a broad candidate is CONTAMINATION, not roll-up evidence. Generalizes the `CURATED_CONTAMINATED_NCTS` hardcode.
- **ct_agent_harness.py** — sends drug × disease into the REAL clinical_trials agent and dumps the raw
  message history plus the assembled `ClinicalTrialsOutput` decision.
- **ct_summary_harness.py** — when the CT summary is FED the resolved dev stage, does the prose stop
  contradicting it (no "no completed Phase 3" when a Phase 3 exists) and judge closure correctly?

### Development stage
- **dev_stage_judgment_harness.py** — can the LLM judge the dev-stage tier from raw phase+status alone
  (Phase 4 ≠ progression, Phase 2/3 counts as Phase 3, unknown/withdrawn ≠ completed)?
- **active_programs_render_harness.py** — pure/deterministic: does `_render_active_programs` describe
  each trial's status honestly (planned ≠ active, unknown ≠ inactive)? No LLM, no network.

### Approval relationship
- **approval_relationship_harness.py** — proposed upstream 4-way classifier (approved / combination_only
  / contaminated / none); only "approved" drops a candidate. Kills the "demoted a real candidate" bugs.
- **approval_input_harness.py** — runs the interpretive call with production-shaped approval inputs;
  checks no phase-tier understatement and no false "approved for this indication" claim.

### Ranking / critic
- **critic_reorder_harness.py** — feeds the ranking critic a deliberately WRONG order (disproven
  candidate first) and asserts it demotes the tested-and-failed/closed candidate to last.
- **critic_reorder_animal_withdrawn_harness.py** — CURRENT vs ENRICHED FACT strings: shows the critic
  can't demote a murine-only/withdrawn-trial candidate until withdrawn-count + animal-only tag are plumbed in.
- **critic_rewrite_harness.py** — A2 critic-rewrite step: rewrites a false "no Phase 3" claim when a
  completed Phase 3 is on record, leaves a true "no regulatory program" claim untouched.
- **ranking_judgment_harness.py** — feeds the REAL `supervisor.txt` + synthetic per-candidate label
  blocks and asks only for the ranked order; isolates whether the prompt alone ranks sensibly (no critic).

### Blurb / interpretive fields
- **interpretive_fields_harness.py** — isolated call handed the authoritative facts, asked only for
  constraint/key_risk/assessment; asserts they don't contradict the given stage.
- **staged_blurb_harness.py** — end-to-end staged blurb chain (judge_dev_stage → judge_interpretive →
  assemble) on the T1DM shape that broke monolithic runs; proves an internally-consistent card.
- **watch_nct_harness.py** — asserts each candidate's `watch` line cites ONLY its own NCTs (fix for an
  NCT leaking across candidates).

### Fact judging
- **fact_judge_harness.py** — feeds a model ONLY the enriched registry facts (title, stop-reason,
  status, phase, literature) to test whether fact-enrichment alone judges the known cases correctly.
  Reads `prompts/fact_judge_prompt.txt`.
