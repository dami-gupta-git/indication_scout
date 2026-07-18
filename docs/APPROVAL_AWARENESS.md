# Approval Awareness — Design & Implementation

This document explains the **approval-aware** machinery in IndicationScout: how the system
distinguishes a genuine drug-repurposing signal from evidence that merely restates a drug's
**already-approved** use. It is the single onboarding reference for everything built across this
work — the labeling, the gates, the ranking, the rendering, and the tests.

> TL;DR. A repurposing report must not present "the drug already works for an approved sub-form of
> this disease" as new evidence. We label the drug↔disease relationship once, upstream; then every
> downstream stage (trials, literature, dev-stage, ranking, rendering) consumes that label so the
> approved part is excluded and only the genuine repurposing signal is surfaced and ranked.

### Summary — Approval Awareness (docs/APPROVAL_AWARENESS.md)

Documents how the system avoids presenting a drug's already-approved use as if it were a new repurposing discovery.

The problem: a repurposing candidate is a broad disease (e.g. "chronic kidney disease"), but the drug may already be FDA-approved for a narrower slice of it (e.g. "diabetic kidney disease"). Trials/papers about that already-approved slice aren't real repurposing evidence — but a naive pipeline counts them anyway, inflating candidates that have zero genuine new signal.

The fix: classify every candidate disease against the drug's FDA label, once, upstream, into one of four labels:

approved — same disease or a narrower subtype/variant of what's approved → dropped entirely. Test: "would prescribing the drug for this candidate be within the existing FDA label?" If yes (it's just a finer-grained version of the approved condition, e.g. diabetic kidney disease under approved CKD), it's not new — drop it.
none — a genuinely distinct disease, even if related (e.g. it can cause the approved condition, like polycystic kidney disease causing CKD, but isn't the same diagnosis) → kept and ranked as real evidence.
combination_only — approved only as part of a combination product → demoted, not dropped.
contaminated — a real repurposing target, but its trial/paper counts are polluted by data actually about the approved sibling/subtype → kept and ranked, but counts are flagged as unreliable.
Why this matters: this same "is it the approved thing or a subtype (exclude) vs a distinct-but-related thing (keep)" logic is applied consistently at every downstream stage — trial relevance, literature relevance, development-stage tiering, and final ranking — so the report doesn't contradict itself (e.g. summary text and trial counts disagreeing about whether something is approved).

Engineering pattern: only a small set of clinical-safety-critical rules are hardcoded in Python (e.g. "if evidence isn't drug-specific, cap the strength score"); everything else is left to LLM judgment, but fed these upstream labels so it can't re-derive or contradict them.

---

## 1. The problem

A repurposing candidate is a BROAD disease (e.g. "NAFLD", "mood disorder", "leukemia"). The drug
may already be **FDA-approved for a narrower part of it** (NASH under NAFLD; MDD/SAD under mood
disorder; Ph+ CML under leukemia). Trials and papers about that approved part are NOT repurposing
evidence — but a naive pipeline counts them, which:

- inflates the candidate's apparent maturity ("completed Phase 3!" — but for the approved subtype),
- floats zero-repurposing-evidence candidates to the top of the ranking,
- contradicts itself across surfaces (summary says one thing, section body another).

The guiding rule for the whole system:

> **Accuracy over coverage.** It is acceptable to miss a real candidate. It is NOT acceptable to
> surface an analysis that wasn't grounded in the upstream data. When unsure, exclude.

---

## 2. The four relationship labels

A label-grounded LLM call (`services/approval_check.py :: get_fda_approved_disease_mapping`)
classifies each candidate against the drug's FDA label into ONE of four labels:

| label              | meaning                                                                 | effect                                  |
|--------------------|-------------------------------------------------------------------------|-----------------------------------------|
| `approved`         | same disease, a synonym, or a narrower CHILD of an approved indication   | **DROP** upstream (never a candidate)   |
| `combination_only` | labeled only as part of a combination product                           | demote                                  |
| `contaminated`     | a real repurposing target, but trial/registry counts are polluted by an approved sibling/child | **KEEP + rank**; trial counts suspect   |
| `none`             | sibling / related / broader-with-uncovered-population / unrelated        | **KEEP**, rank normally                 |

Only `approved` removes a candidate. `contaminated` and `none` are both kept and ranked.

### The on-label test (how `approved` vs `none` is decided)

> Would prescribing the drug for the candidate be **on-label** — i.e. do the candidate's patients
> already fall within the approved population?

- **Yes → `approved`.** A clinically-named SUBTYPE or CAUSE-VARIANT of a broad approval is on-label
  (e.g. diabetic kidney disease ⊂ approved CKD → `approved`, drop).
- **No → `none`.** A DISTINCT disease that merely *causes* the approved condition is NOT on-label —
  keep it (e.g. polycystic kidney disease or glomerulonephritis, which cause CKD but are distinct
  diseases). The model's distinct-disease judgment here is usually more correct than a naive
  "it's related, demote it" heuristic.

The prompt lives in `prompts/extract_fda_approval_single.txt`. Curated short-circuits (for known
cases) live in `constants.py` (`CURATED_FDA_APPROVED_CANDIDATES`,
`CURATED_FDA_CONTAMINATED_CANDIDATES`, `CURATED_FDA_COMBINATION_ONLY_CANDIDATES`,
`CURATED_FDA_REJECTED_CANDIDATES`).

### Why upstream, once

Earlier, the supervisor LLM re-derived demotion in free-text prose every run. That produced the
**typed↔prose split** (a candidate demoted in the summary text while its typed field was empty) and
flip-flopping (T1DM/NAFLD demoted one run, kept the next). Fix: classify ONCE, upstream, as typed
data; the supervisor is a pure CONSUMER of the label, never a re-judge.

---

## 3. Directional exclusion — the rule everything shares

The single most important rule, applied identically at the trial and literature layers:

> Exclude evidence about the approved indication **or NARROWER** (a sub-indication of the
> candidate). KEEP evidence about a **BROADER approved parent** of the candidate.

- candidate **NAFLD**, approved **NASH** (narrower) → a NASH trial/paper is **excluded** (it's the
  approved subtype).
- candidate **DKD**, approved **CKD** (broader parent) → a CKD trial/paper is **KEPT** — it is the
  candidate's own evidence (the candidate is the narrower disease, CKD evidence rolls up to it).

### Severity / stage qualifier stripping

A severity/stage/biomarker qualifier on the approval does NOT create a separable disease. If
approved is "MASH **with moderate-to-advanced fibrosis**", then a bare "NASH" / "MASH" /
"steatohepatitis" trial IS the approved sub-indication → excluded. ("NASH" is the approved disease,
not a broad parent of it.)

### Minority-biomarker breadth (the opposite case)

A bare broad disease is NOT the approved subset when the approval covers only a small biomarker
subset. Approved "EGFR-mutated NSCLC" (~10-15% of NSCLC) — an all-comers NSCLC trial is genuinely
broader → KEPT. (Distinguish from a severity grade, which is not a minority biomarker.)

### Sibling ≠ subtype

"Type 1 diabetes" is a SIBLING of approved "type 2 diabetes", not a narrower form → KEPT (falls
through to "relevant").

---

## 4. The pipeline, stage by stage

```
candidate list ──> approval labeling (4-way, upstream, typed)
                        │  drops `approved`; carries contaminated/none/combination_only
                        ▼
   ┌────────────────────────────────────────────────────────────┐
   │  per candidate (approved_indications threaded down):         │
   │                                                              │
   │   clinical-trials gate ──> per-trial relevant/contaminated   │
   │        │                    (TEST 1/2/3, ordered)            │
   │        ▼                                                      │
   │   dev_stage (LLM tier over RELEVANT trials only)             │
   │   active_programs (DETERMINISTIC, from relevant trials)      │
   │                                                              │
   │   literature gate ──> per-PMID relevant/contaminated,        │
   │        then strength/direction/basis over the relevant set   │
   └────────────────────────────────────────────────────────────┘
                        │
                        ▼
   supervisor RANKING (LLM judgment over all labels)
                        │
                        ▼
   critique_ranking (FACT-CHECK only — Phase-3 contradiction repair)
                        │
                        ▼
   render: MD report + JSON payload + React (one source per field)
```

### 4a. Clinical-trials relevance gate

`prompts/clinical_trials.txt` — classify EVERY shown trial (search, completed, terminated). Active
trials live only in the search scope, so they MUST be classified or a contaminated active trial
leaks into the "active development" signal. Apply IN ORDER, first match wins:

- **TEST 1 — approved-subtype → CONTAMINATION.** Any condition the trial studies is an approved
  indication or narrower. Includes the severity-qualifier rule. **Multi-condition rule:** a trial
  contaminates if ANY listed condition is the approved subtype — a co-listed sibling/other disease
  does NOT rescue it (e.g. "type 2 diabetes with NASH" → contaminated, the T2DM framing is
  irrelevant). *Does NOT* contaminate: a sibling, a minority-biomarker-broad disease, or anything
  when approved=(none).
- **TEST 2 — distinct disease or wrong drug → CONTAMINATION.** A distinct disease sharing a parent
  term (systemic Hypertension query pulling in PAH — PAH is separate), or the studied drug is not
  this drug (it's only a comparator/PK probe).
- **TEST 3 — otherwise → RELEVANT.** Studies this drug for this indication or a narrower
  non-approved subtype (rolls up as real evidence).

Judge from DRUGS / TITLE / SUMMARY, not disease name alone. When unsure, prefer contaminated.

The output is a per-NCT verdict map; downstream signals (`relevant_nct_ids`,
`contaminated_nct_ids`) are computed over it. Stage and counts are graded on the RELEVANT set only.

### 4b. Development stage (`services/dev_stage.py`)

Two outputs from the relevant trial set:

1. **tier** — an LLM judgment (`_STAGE_PROMPT`) into one of 8 tiers: `phase3_terminated_for_cause`
   > `completed_phase3` > `active_phase3` > `phase3_unknown_status` > `completed_phase2` >
   `exploratory_phase4_only` > `early_phase` > `untested`. The LLM returns ONLY `{tier, reason}`.

   A **deterministic floor** (`_enforce_tier_floor`) backstops the LLM (it mis-tiers on long, noisy
   trial lists, especially around the "Phase 2/Phase 3" ambiguity):
   - trials exist → never `untested` (floor to `early_phase`);
   - a completed pure Phase-3-band trial exists → at least `completed_phase3`;
   - an active pure Phase-3-band trial exists → at least `active_phase3`;
   - `phase3_terminated_for_cause` (a deliberate closure judgment) is never overridden.

2. **active_programs** — **fully deterministic** (`_render_active_programs`), NOT the LLM. "What is
   still moving": filter to active statuses, list pure Phase 3 and Phase 2/3 as separate groups,
   each with a count that EQUALS its listed NCT ids; else non-pivotal-active; else "None active".
   Counting/listing NCTs is mechanical — letting the LLM do it produced miscounts ("5 Phase 3"
   while listing 4) and false "None active".

   **Status-format gotcha:** CT.gov returns underscored uppercase statuses
   (`NOT_YET_RECRUITING`, `ACTIVE_NOT_RECRUITING`). Always match via
   `_trial_signals._is_active` / `_normalize_status` — never a hand-rolled space-lowercase set
   (that silently drops not-yet-recruiting trials).

   The dev_stage cache stores the **tier only**; active_programs is re-rendered every call so a
   render-logic change takes effect with no cache bust.

### 4c. Literature (`services/retrieval.py :: synthesize`)

A SINGLE LLM call reads the abstracts once and emits one internally-consistent EvidenceSummary:
per-PMID `verdicts`, `evidence_basis`, `strength`, `is_observational`, `summary`, `key_findings`.
(This merged the former two-pass `synthesize` + `judge_literature_strength`, which let prose drift
from the typed fields. `literature_strength.py` is deleted.) The `verdicts` map labels each PMID
contaminated | supporting | contradicting | mixed; `supporting_pmids` / `contradicting_pmids` /
`relevant_pmids` / `study_count` / overall `direction` are then built DETERMINISTICALLY in code
from that map (supporting = supporting+mixed; contradicting = contradicting+mixed; relevant =
non-contaminated) — the literature analogue of the per-trial gate.

`evidence_basis` is the key field:

- `drug_specific` — real this-drug-this-disease evidence.
- `class_level` — the disease-relevant RCTs are for OTHER drugs in the class.
- `approved` — the only relevant evidence is for an approved sub-indication.
- `none` — no relevant abstracts retrieved.

**Per-PMID direction sub-call (`_judge_pmid_directions`, the authority for direction).** A small
isolated `query_small_llm` call over the RELEVANT abstracts asks ONE narrow question per abstract:
"did THIS drug benefit THIS disease in THIS abstract? (supporting | contradicting | mixed)". Its
verdict OVERRIDES synthesize's direction for any relevant PMID. This replaced earlier
phrase-matching guards, which could not attribute a benefit to the right drug-arm — e.g. metformin
× hepatic steatosis, where a *comparator's* benefit ("ipragliflozin significantly improved …
metformin showed minimal change") or a side metabolic-marker improvement read as "supporting" for
metformin. Direction is a semantic judgment; the narrow framing handles attribution that regex
cannot. (See `docs/future.md` history — the regex guard was retired in favor of this.)

**The strength cap (the one clinical-safety invariant kept in deterministic code):**
`evidence_basis != "drug_specific" → strength = none, direction = none`. This prevents the
class-level-RCT inflation bug (e.g. GLP-1 class RCTs for Parkinson's reading as "strong" for
semaglutide). The prompt EMITS the basis + raw strength; one post-call line ENFORCES the cap.

The directional-exclusion and severity rules from §3 apply per-abstract here too.

### 4d. Ranking (`prompts/supervisor.txt`)

Ranking is the supervisor LLM's **judgment over all the labels it sees**, not a fixed tier ladder.
Per candidate it has: dev_stage tier + relevant-trial counts; literature strength / direction /
design / **evidence_basis**; the closure verdict; any adverse signal. (The `evidence_basis` is
surfaced explicitly in the literature header for exactly this reason.)

Guidance the LLM weighs (not rigid):

- A completed pivotal stage with a quantified readout beats recruiting/rationale-only.
- Genuine drug-specific supportive literature usually outweighs a higher trial stage with NO
  drug-specific literature behind it.
- `direction=contradicts` (moderate/strong) is a DISPROVEN hypothesis → near the bottom; a CLOSED
  candidate ranks below live ones.
- **Weak-grounding distinction** (two different `strength=none` cases, ranked differently):
  - basis `approved` / `class_level` — abstracts exist but don't study this drug as repurposing →
    poorly grounded; should NOT outrank a genuine drug-specific supportive candidate.
  - basis `none` (no abstracts yet) — a real completed trial of THIS drug, publications haven't
    caught up → a legitimate early signal; do NOT bury it.

This is deliberately LLM judgment with all labels visible, rather than hard demotion rules — the
labels (especially `evidence_basis`) give it what it needs to decide. Validated by
`scratch/ranking_judgment_harness.py`.

### 4e. critique_ranking — FACT-CHECK only

After the supervisor drafts ranked blurbs, `critique_ranking` runs a fresh-context critic that
**repairs factual contradictions** against each candidate's authoritative Phase-3 FACT (so no blurb
claims "no Phase 3" when a relevant Phase 3 is on record). It does **NOT** reorder or re-rank —
ordering is the supervisor's judgment. It is still mandatory before `finalize_supervisor` (a
call-order gate), now purely as the fact-check pass.

### 4f. Rendering & propagation

The resolved facts have ONE source each and reach every surface:

- `signals.active_programs` (deterministic) → blurb (finalize overwrite), CT-section prose
  (`clinical_trials_summary`), interpretive fields (`judge_interpretive`), the MD report table
  (`format_report.py`), the JSON payload, and React (`frontend/src/types.ts`,
  `overview/blurb.ts`, `overview/SummaryBlurbCard.tsx`).
- dev_stage tier → the authoritative `stage` line (overwrites whatever the LLM drafted).
- The internal word "contaminated" never appears in user-facing report prose — it is translated to
  plain language (e.g. "evidence is for an already-approved sub-indication").
- `watch` blurb field: a one-line prompt rule requires any cited NCT to belong to THIS candidate
  (a T1D NCT once leaked into Parkinson's watch line).

---

## 5. Where each invariant is enforced (code vs prompt)

Most relationship judgments are the LLM's (with the labels visible). A SMALL set of
clinical-accuracy invariants are deterministic in code because the LLM cannot be trusted to honor
them on every run:

| invariant                                                    | enforced in                              |
|--------------------------------------------------------------|------------------------------------------|
| 4-way approval label (typed, once, upstream)                 | `approval_check.py` (+ curated sets)     |
| strength cap (basis ≠ drug_specific → strength/direction none)| `retrieval.py` (post-synthesize)         |
| dev_stage tier floor (completed/active Phase 3; never untested-with-trials) | `dev_stage.py :: _enforce_tier_floor` |
| active_programs (count == listed ids; what's moving)         | `dev_stage.py :: _render_active_programs`|
| trial-status normalization (CT.gov underscored forms)        | `_trial_signals._is_active`              |

Everything else — relevance verdicts, tier choice, ranking order, weak-grounding — is LLM judgment
fed the labels.

---

## 6. Test coverage

Marker: `@pytest.mark.approval_aware` (run with `pytest -m approval_aware`). Most are integration
tests that hit live Anthropic + live CT.gov; anchors are guarded with `if <nct> in shown` / skip so
CT.gov drift doesn't flake them. Deterministic logic (floors, renders, parsing) is unit-tested.

By stage:

- **Approval labeling** — `tests/integration/services/test_approval_check.py`: semaglutide T2DM=approved,
  NAFLD=contaminated, unrelated=none; label extraction from Ozempic/Wegovy.
- **Trial gate** — `tests/integration/agents/clinical_trials/test_approval_aware_relevance_e2e.py`:
  sildenafil PAH list-driven contamination; bupropion approved-SAD via rule; semaglutide
  multi-condition "T2DM with NASH" contaminated; imatinib Ph+ CML contaminated / CLL relevant.
  Plus `test_sild_htn_relevance_e2e.py` (full hand-labeled systemic-HTN vs PAH split).
- **Literature** — `tests/integration/services/test_literature_strength.py`: class-level Parkinson
  not strong; drug-specific T1DM not none; bupropion MDD approved-not-strong vs bipolar
  drug-specific; empagliflozin CKD parent-kept. `_assert_self_consistent` guards the
  no-orphan-PMID / no supporting∩contradicting invariants (the merge).
- **dev_stage** — `tests/integration/services/test_dev_stage.py`: tier crux cases incl. the
  Phase-2/3 trap, noisy-set-with-completed-Phase-3 (imatinib Leukemia shape), Not-Applicable→early
  (bupropion Mood Disorder), semaglutide T1D active-Phase-3 count==listed-ids.
  `tests/unit/services/test_dev_stage.py`: the floors, the deterministic active_programs render
  (count, Phase-2/3 split, CT.gov underscored status, non-pivotal, none-active), tier parsing.
- **Ranking** — `tests/integration/agents/supervisor/test_supervisor_ranking.py`: drug-specific
  beats approved-basis; contradicts/closed sink to bottom; real semaglutide T1DM>NAFLD>Parkinson.
- **Supervisor (full agent)** — `tests/integration/agents/supervisor/test_supervisor_agent.py`:
  semaglutide sibling-kept + contaminated labels; sildenafil systemic-HTN kept + PAH tagged.

Validation harnesses (not pytest; run manually under `scratch/`): `approval_relationship_harness.py`,
`approval_aware_trials_harness.py`, `approval_aware_literature_harness.py`,
`trial_relevance_test1_harness.py`, `ranking_judgment_harness.py`, `watch_nct_harness.py`,
`dev_stage_judgment_harness.py`.

---

## 7. Key files

| concern               | file                                                       |
|-----------------------|------------------------------------------------------------|
| 4-way label           | `services/approval_check.py`, `prompts/extract_fda_approval_single.txt` |
| curated label sets    | `constants.py`                                             |
| trial gate            | `prompts/clinical_trials.txt`, `agents/clinical_trials/`   |
| trial signals/status  | `agents/_trial_signals.py`                                 |
| dev stage + floor + active_programs | `services/dev_stage.py`                      |
| literature merge + strength cap | `services/retrieval.py`, `prompts/synthesize.txt`, `models/model_evidence_summary.py` |
| ranking + watch + critique | `prompts/supervisor.txt`, `agents/supervisor/supervisor_tools.py` |
| rendering             | `report/format_report.py`, `frontend/src/overview/`        |

For the running log of decisions and gotchas behind all of this, see `for_me/findings.md`
(gitignored) — this doc is the durable summary; findings.md is the blow-by-blow.
