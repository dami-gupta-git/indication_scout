# GLOSSARY — Reading an IndicationScout Report

This document explains **what the tool does** and **what every field, verdict, and label in the
report means** — including the exact rule that produces each one. It is not about setup, architecture,
or the agents. It answers questions like: what is "Closed signal", how is Evidence "strong" decided,
what does "moderate, mixed, RCT-backed" mean, and what is each number on the Clinical Trials page.

A note used throughout: some fields are **deterministic** (computed in code from typed data — the same
inputs always give the same value) and some are **LLM-authored** (a language model writes them from the
deterministic facts it is handed). This matters because LLM-authored fields (notably the verdict tags)
are not drawn from a fixed list — the examples below are the ones the model is steered toward, not an
enforced set.

---

## 1. What the application does

You give it **one drug name** (`scout find -d "metformin"`, or type it in the web UI). It produces a
**drug-repurposing report**: a ranked set of *other* diseases the drug might be repurposed for, each
characterized by the current state of its hypothesis — trial activity, recent literature, mechanistic
plausibility, and regulatory status.

Candidate diseases are seeded **only from OpenTargets target–disease associations**. The tool looks up
the drug's molecular target(s), then pulls the diseases those targets are associated with. This is why
the banner reads:

> Candidates come from OpenTargets target–disease associations only; off-target indications
> (e.g. duloxetine for pain) won't appear.

An indication that isn't mediated by the drug's known target(s) — an "off-target" effect — never enters
the candidate list. That is a deliberate scope limit, not a bug: the tool errs toward omission over
surfacing an ungrounded candidate.

The report is shown across four tabs: **Overview**, **Mechanism**, **Clinical Trials**, **Literature**.

---

## 2. Overview — the stat tiles

The four top-band numbers:

| Tile | What it counts |
|---|---|
| **Candidate diseases** | Every disease surfaced from OpenTargets, whether or not it was fully investigated. |
| **Investigated** | How many candidates received full literature + clinical-trials analysis. Capped at **3** (`settings.supervisor_investigation_cap`, set via the `SUPERVISOR_INVESTIGATION_CAP` env var), so this is usually `min(3, Candidate diseases)`. |
| **Total trials** | Sum, across investigated diseases, of each disease's all-status trial count. |
| **Total studies** | Sum, across investigated diseases, of each disease's graded PubMed abstract count. |

So "17 candidate diseases / 3 investigated" means 17 were surfaced but only the top 3 got the deep
literature + trials workup. The other 14 appear as candidates but have no detail card.

---

## 3. Overview — the Disease Scorecard

One row per genuine candidate disease (demoted / approval-only entries are dropped). Columns:

| Column | Meaning |
|---|---|
| **Rank** | 1-based position chosen by the supervisor, capped at `settings.supervisor_candidate_cap` (15 by default). Matches the summary-card numbering. |
| **Verdict** | The one-tag interpretive assessment of the hypothesis — see §4. |
| **Evidence** | Literature evidence **strength** — a quantity/quality grade (`strong` / `moderate` / `weak` / `none`), independent of whether the evidence supports or contradicts the drug — see §5. |
| **Trials** | Exact all-status count of trials matching this drug × this disease. |
| **Competitors** | Number of *distinct sponsor + drug programs* running trials in that disease area (other drugs competing in the indication, not the analyzed drug). |

**Competitors, precisely:** the tool fetches trials for the indication, keeps only Drug/Biological
interventions (vaccines excluded), and groups them by `sponsor | drug_name`. Each distinct
sponsor+drug pair is one competitor, carrying its own max phase, trial count, and enrolment. The count
is that list's length.

**Trials, precisely:** the all-status pair count from ClinicalTrials.gov's `countTotal` for the drug ×
MeSH-indication pair. (It counts trials by status once — completed and terminated trials are counted
separately elsewhere to avoid double-counting.)

---

## 4. Verdict / "Assessment" tags — the interpretive one-liner

**These are LLM-authored, not rule-based.** A single interpretive judge is handed the deterministic
facts (stage, active programs, trial count, literature summary, relationship, approved-indication
status) and asked to pick a short verdict tag that fits. The tags below are the **examples the model is
steered toward** — it can emit others, and there is no enforced list.

| Tag | What it's meant to convey |
|---|---|
| **Live but bottlenecked** | Development is ongoing but something is holding it back (evidence/regulatory/commercial). |
| **Maturing, awaiting readout** | Late-stage work is underway and the field is waiting on results. |
| **Tested, status unconfirmed** | Trials exist but their current status is unknown — the *neutral* tag preferred over a decline tag when status is unclear. |
| **Stalled, regulatory gap** | Progress has stopped against a regulatory hurdle. |
| **Untested at scale** | Little or no substantial trial evidence. |
| **Closed signal** | The hypothesis appears foreclosed — e.g. late-stage work completed but no pivotal program remains and evidence has turned against it. |

Soft rules baked into the judge's prompt (these constrain the choice, they don't hard-assign a tag):
- A **nonzero trial count** means the hypothesis *was* studied — the judge must not call it untested or
  abandoned.
- When **trials exist but status is unknown**, prefer a neutral tag (e.g. "Tested, status unconfirmed")
  over a decline tag.
- **"None active"** means no trial is *currently recruiting* — state that as the status; do not infer a
  cause that wasn't given.
- The tag must **not name a phase tier** (that's the Stage field's job).

> Note: "Closed signal" also appears as a *footer heading* grouping candidates the clinical-trials
> sub-agent judged closed. That footer is a separate use from the per-disease verdict tag above.

**Constraint** and **Key risk** on each detail card are authored by the same judge:
- **Constraint** — what holds the hypothesis back (regulatory / commercial / evidence gap), or the
  current status if nothing clearly does. The judge is told not to invent a blocker.
- **Key risk** — the single biggest risk to the hypothesis, in one phase-free line.

---

## 5. Evidence strength — strong / moderate / weak / none

Grades the literature by **quantity and quality only, independent of direction**. A trial body that
*disproves* the hypothesis can still be "strong". An LLM grades it, then a deterministic cap can
override the grade.

| Grade | Rule |
|---|---|
| **strong** | Multiple drug-specific clinical studies (RCTs, large cohorts) for *this* drug in *this* disease — whether they show efficacy or failure. Never "strong" unless the evidence is drug-specific. |
| **moderate** | Small drug-specific clinical studies, case series, or strong drug-specific preclinical data. |
| **weak** | Drug-specific case reports only, or drug-specific in-vitro / animal data only. |
| **none** | No drug-specific evidence at all. |

**Deterministic cap (overrides the LLM):** if the evidence isn't drug-specific — i.e. it's class-level,
for an already-approved sub-indication, or absent — strength is forced to **none** regardless of the
LLM grade. (What "already-approved sub-indication" means, and how such evidence is stripped out across
the whole report, is §9.)

How it renders on the detail card:
- Class-level evidence → "class-level signal (no direct evidence for this drug)".
- Already-approved sub-indication → "evidence is for an already-approved sub-indication (not
  repurposing)".
- Otherwise → the strength word, plus the direction if known.

---

## 6. Literature one-liner — "moderate, mixed, RCT-backed / controlled"

The card's Literature line is built deterministically from three pieces: **strength**, **direction**,
**design**.

**Strength** — the word from §5 (`strong` / `moderate` / `weak` / `none`).

**Direction** — set deterministically from how the individual abstracts split:

| Direction | Rule |
|---|---|
| **supports** | Only supporting abstracts. |
| **contradicts** | Only contradicting/mixed abstracts, none supporting. |
| **mixed** | At least one supporting *and* at least one contradicting/mixed abstract. |
| **none** | No relevant abstracts (or evidence isn't drug-specific). |

**Design** — derived from typed flags on the evidence:

| Design phrase | Rule |
|---|---|
| **RCT-backed / controlled** | At least one relevant drug-specific RCT or controlled trial. |
| **observational** | Relevant evidence is observational. |
| **animal/in-vitro only** | Only animal / in-vitro evidence (takes precedence). |
| **undetermined design** | Study design couldn't be determined. |

So **"moderate, mixed, RCT-backed / controlled"** reads as: *moderate* amount/quality of evidence, that
is *mixed* in direction, drawn from *controlled trials*. (When evidence is class-level or
approved-indication, this line short-circuits to the corresponding note instead.)

---

## 7. Disease detail card — Stage & Active programs

### Stage (Development stage)

A deterministic phrase from a fixed tier vocabulary. The *tier* is chosen by an LLM but constrained by
a deterministic floor, so it can never contradict the trial record.

| Tier | Phrase shown |
|---|---|
| phase3_terminated_for_cause | "Phase 3 terminated for cause (safety/efficacy stop)" |
| completed_phase3 | "Phase 3 completed for this indication" |
| active_phase3 | "Active Phase 3 development on record for this indication" (appends the active NCT ids) |
| phase3_unknown_status | "Phase 3 on record, status unknown" |
| completed_phase2 | "Phase 2 completed for this indication, no Phase 3" |
| exploratory_phase4_only | "Phase 4 exploratory only (post-approval off-label study; no dedicated development program for this indication)" |
| early_phase | "Early-phase only, no completed pivotal readout" |
| untested | "No registry (ClinicalTrials.gov) trials on record for this indication" |

Deterministic floors that override the LLM's tier pick:
- If **any trial exists**, the tier can never be "untested".
- A **completed pure Phase-3 trial** forces at least `completed_phase3`.
- An **active pure Phase-3 trial** forces at least `active_phase3`.
- `phase3_terminated_for_cause` is honored **only** if a Phase-3 trial has a genuine safety/efficacy
  stop reason; otherwise it's demoted.
- A `completed_phase3` backed only by a combined Phase 2/3 (no standalone Phase 3) renders as
  "Phase 2/Phase 3 completed for this indication (no standalone Phase 3)".

### Active programs (pivotal vs non-pivotal)

Fully deterministic. Looks only at currently-active trials:
- **Pivotal active/planned** (Phase 3 or Phase 2/3) → names them with counts and NCT ids, e.g.
  "2 Phase 3 active (NCT…)".
- **Only earlier-phase or post-approval trials active** → "No pivotal program active; N non-pivotal
  active (NCT…)".
- **Nothing active** → "None active", or "None active; N on record with unknown status (NCT…)" when
  unknown-status trials exist.

The card's closing **Assessment** = the verdict tag from §4. The 2-sentence prose blurb is LLM-authored
from the same facts (and must surface a failure/disproof when the literature direction is "contradicts").

---

## 8. Clinical Trials tab

This tab is driven by the currently focused disease.

### KPI tiles

| Tile | Meaning |
|---|---|
| **Total trials** | All-status trial count for this drug × disease pair. |
| **Recruiting** | Trials currently enrolling participants (CT.gov status `RECRUITING`). |
| **Active (not recruiting)** | Trials ongoing but no longer enrolling (`ACTIVE_NOT_RECRUITING`). |

### Status breakdown (donut)

Built from four count buckets (each a separate CT.gov query). Zero-count buckets are hidden.

| Status | Meaning |
|---|---|
| **RECRUITING** | Currently enrolling. |
| **ACTIVE_NOT_RECRUITING** | Ongoing, enrolment closed. |
| **WITHDRAWN** | Stopped before enrolling anyone. |
| **UNKNOWN** | CT.gov auto-assigns this when a record hasn't been updated in ~2 years. The trial *ran* — the outcome is just unknowable from status. **Not** the same as "never happened". |

Note: COMPLETED and TERMINATED are deliberately **not** in this breakdown — they're reported in their
own sections so counts aren't doubled.

### Completed trials, by phase

A funnel + filterable table of all COMPLETED trials for the pair, grouped by registry phase. Phase
order, earliest to latest:

`Early Phase 1 → Phase 1 → Phase 1/Phase 2 → Phase 2 → Phase 2/Phase 3 → Phase 3 → Phase 3/Phase 4 → Phase 4 → Not Applicable`

- **Early Phase 1** — pre-Phase-1 / exploratory; sorts first.
- **Phase 1–4** (and combined labels like "Phase 1/Phase 2") — standard registry phase labels.
- **Not Applicable** — CT.gov's own value for studies where the drug-phase concept doesn't apply (device,
  behavioral, observational studies). It's a verbatim registry label, sorted last. (Distinct from an
  empty phase, which the UI relabels "Unknown".)

### Filters

- **Phase** filter — options are the distinct phases present; shared with the funnel, so clicking a
  funnel segment filters the table.
- **Status** filter — distinct statuses present in the shown trials.
- The competitor table has its own separate **Max phase** filter.

### Other sections

- **Terminated trials** — trials stopped early, each showing phase and `why_stopped` text.
- **Competitive landscape** — the competitor table (Drug / Sponsor / Max phase / Trials), one row per
  sponsor+drug program (see §3).
- **Excluded trials** — trials the agent judged to belong to a *different* indication are removed from
  the tables but still counted in the verbatim total-count headers.

---

## 9. Approval relationship & contamination — why some evidence is stripped out

This is the machinery that keeps the report from presenting *"the drug already works for an approved
part of this disease"* as if it were new repurposing evidence. It is why a candidate can show a high
trial count yet a low evidence grade, and why some trials appear in the count headers but not in the
tables.

### The problem it solves

A candidate disease is usually **broad** (e.g. "NAFLD"). The drug may already be FDA-approved for a
**narrower part** of it (e.g. "NASH" ⊂ NAFLD). Trials and papers about that approved narrow part are
**not** repurposing evidence — but a naive count would include them, making a candidate look more mature
than its genuine repurposing signal warrants. "Contamination" is the term for that already-approved
evidence leaking into a candidate's counts.

The governing rule is **accuracy over coverage**: it is acceptable to miss a real candidate; it is not
acceptable to surface evidence that is really about the approved use. When unsure, exclude.

### The approval relationship label (one per candidate, decided once)

Every candidate is classified once, upstream, against the drug's FDA label into exactly one of four
**approval-relationship** labels. This is typed data — computed before any per-disease analysis and
then handed down — not something re-decided per surface.

| Label | Meaning | Effect on the report |
|---|---|---|
| **approved** | The candidate is the same disease, a synonym, or a **narrower child** of an approved indication. | **Dropped entirely** — never appears as a candidate. |
| **combination_only** | Approved only as part of a combination product. | Kept but demoted. |
| **contaminated** | A genuine repurposing target, **but** its trial/registry counts are polluted by an approved sibling or child. | **Kept and ranked**; its trial/paper counts are treated as suspect and filtered (below). |
| **none** | A sibling, a broader indication with an uncovered population, or unrelated. | Kept and ranked normally. |

Only **approved** removes a candidate. The deciding test is *"would prescribing the drug for this
candidate be on-label — do its patients already fall inside the approved population?"* Yes → `approved`
(dropped); No → `none` (kept). So a clinically-named subtype of a broad approval is dropped, but a
*distinct* disease that merely causes the approved condition is kept.

### The directional rule (shared by trials and literature)

Once a candidate is kept, individual trials and papers are filtered by one rule:

> Exclude evidence about the approved indication **or anything narrower**. **Keep** evidence about a
> **broader approved parent** of the candidate.

- candidate **NAFLD**, approved **NASH** (narrower) → a NASH trial/paper is **excluded** (it's the
  approved subtype).
- candidate **DKD**, approved **CKD** (broader parent) → a CKD trial/paper is **kept** — it rolls up as
  the candidate's own evidence.

Two refinements: a severity/stage qualifier does **not** create a separable disease (approved "NASH with
moderate fibrosis" still means a bare "NASH" trial is excluded); but a **minority-biomarker** approval
does (approved "EGFR-mutated NSCLC", ~10–15% of NSCLC, means an all-comers NSCLC trial is genuinely
broader → kept). A **sibling** is never a subtype ("type 1 diabetes" vs approved "type 2 diabetes" →
kept).

### Where contamination shows up in the report

- **Trials tab — "Excluded trials"** (§8): trials judged to be about the approved part (or a distinct
  indication) are removed from the tables. They are **still counted in the verbatim total-count
  headers** — the header is the raw registry total, the tables show only the relevant subset.
- **Stage & Active programs** (§7): computed over the **relevant** (non-contaminated) trials only, so a
  contaminating approved-indication Phase 3 does not inflate the stage.
- **Evidence strength** (§5): if the only relevant literature is for the approved sub-indication, the
  deterministic cap forces strength to **none**, and the card reads *"evidence is for an already-approved
  sub-indication (not repurposing)"*.

The internal word "contaminated" is never shown in report prose — it is always translated to plain
language like the phrase above.

> **Full reference:** the design, the exact prompt tests (TEST 1/2/3), the per-PMID literature gate, and
> where each invariant is enforced in code live in [`docs/APPROVAL_AWARENESS.md`](docs/APPROVAL_AWARENESS.md).

---

## Deterministic vs LLM-authored — quick reference

| Field | Source |
|---|---|
| Stage phrase, Active programs, Trials/Recruiting/Active counts, Status breakdown, Competitors | **Deterministic** |
| Literature direction ("mixed"/"contradicts") and design ("RCT-backed / controlled") | **Deterministic** (from typed flags) |
| Evidence strength | LLM grade, then **deterministic cap** to "none" if not drug-specific |
| Stage *tier* selection | LLM, constrained by **deterministic floor** |
| Verdict / Assessment tag, Constraint, Key risk, prose blurb | **LLM-authored** free text |
