# Report description

Where every block in the React UI and the downloadable Markdown report comes
from — which Pydantic field, which agent, and (for supervisor-authored prose)
which block of `supervisor.txt` instructs it.

Both rendering paths share the same `SupervisorOutput` object but consume it
differently:

- **React UI** — `frontend/src/`, renders the JSON interactively. The frontend
  fetches the `SupervisorOutput` JSON directly (see §0) and reads its fields; it
  does **not** parse the Markdown report.
- **Markdown report** — `src/indication_scout/report/format_report.py`, produces
  the `.md` file the user downloads (and the snapshots in `snapshots/`).

`SupervisorOutput` itself is built by
`src/indication_scout/agents/supervisor/supervisor_agent.py::run_supervisor_agent`.

---

## 0. Data flow (React ↔ API)

The React app polls the FastAPI backend for the JSON result, then renders from
the JSON. The Markdown report is fetched separately, only on download.

| Step | Endpoint (`api/routes/analyses.py`) | Returns |
|---|---|---|
| Start a run | `POST /api/analyses` | `{job_id}` |
| Poll for result | `GET /api/analyses/{job_id}` (every ~1.5s, `useAnalysis.ts`) | `AnalysisStatusResponse` — `status` + `result: SupervisorOutput \| null` |
| Load a built-in example | `GET /api/examples/{drug}` | same shape |
| Download report | `GET /api/analyses/{job_id}/report.md` (`analyses.py:93-103`) | Markdown string from `format_report(result)` |

API client: `frontend/src/api.ts`. Polling hook: `frontend/src/useAnalysis.ts`.
TypeScript mirror of the Pydantic shape: `frontend/src/types.ts`.

The React UI consumes the JSON object directly — no intermediate Markdown
parsing. `format_report` (see §2) is only invoked server-side for the download.

---

## 1. React UI — `frontend/src/`

### 1.1 App shell & header band — `App.tsx`

| UI element | Pydantic field | Source |
|---|---|---|
| Title `{drug}` (header / sidebar) | `result.drug_name` | Drug name passed to `run_supervisor_agent` |
| Scope disclaimer caption | none | Static copy |
| **Candidate diseases** KPI | `result.candidate_diseases.length` | Merged allowlist (deterministic): competitor scan (OpenTargets, drugs sharing same target) + mechanism-promoted diseases. Built in `agents/supervisor/supervisor_tools.py::find_candidates` + `merge_and_dedup`. Not LLM-authored. |
| **Investigated** KPI | `result.disease_findings.length` | Subset of the allowlist the supervisor LLM chose to call `analyze_literature` / `analyze_clinical_trials` on |
| **Total trials** KPI | sum of `finding.clinical_trials.search.total_count` over `disease_findings` | clinical_trials agent (`search_trials` tool → ClinicalTrials.gov via `ClinicalTrialsClient`) |
| **Total studies** KPI | sum of `finding.literature.evidence_summary.study_count` over `disease_findings` | literature agent (`synthesize` tool → judgment over PubMed abstracts retrieved by `PubMedClient` + ranked by `RetrievalService`) |
| Tab nav (Overview / Mechanism / Clinical Trials / Literature) | none | Static labels; keyboard-accessible |
| Sidebar **focus disease** radio group | `finding.disease` per `disease_findings` | Local UI state; filters the Clinical Trials and Literature tabs to one disease. "All" resets. |
| Download report button | `getReportMarkdown(jobId)` → `/report.md` | `format_report(result)` (see §2) |

Pre-result states: `LandingHero.tsx` (empty state: tagline, example-drug chips)
and `LoadingState.tsx` (in-flight pipeline progress, KPI skeletons). Both are
static apart from the drug name prop.

### 1.2 Overview tab — `tabs/OverviewTab.tsx`

| UI element | Pydantic field | Source |
|---|---|---|
| **Disease scorecard** grid (`grid/ComparisonGrid.tsx`) | per-disease, see §1.2.1 | one row per genuine candidate |
| **Disease detail** ranked cards (`overview/SummaryBlurbCard.tsx`) | `finding.blurb` per `top_diseases`, see §1.2.2 | supervisor LLM blurbs |
| Summary footer (Demoted / Closed signals / Evidence gate exclusions) | tail of `result.summary` | Extracted from the summary string by `summaryFooter.ts`; rendered via `components/Markdown.tsx` |
| **Candidate diseases** list with investigated / not-investigated markers | `result.candidate_diseases` vs `{f.disease for f in disease_findings}` | Merged allowlist; the ✓ flag is computed locally |

Only candidates with a structured blurb appear as scorecard rows / detail cards.
`blurb.ts::hasStructuredBlurb` gates this (true if any of stage, literature,
blocker, active_programs, key_risk, verdict, watch is non-empty), filtering out
demoted / prose-only entries.

#### 1.2.1 Disease scorecard columns — `grid/ComparisonGrid.tsx` (rows built in `gridRows.ts`)

| Column | Pydantic field | Source |
|---|---|---|
| **Rank** | order within `result.top_diseases` (1-based, genuine candidates only) | supervisor ranking |
| **Disease** | `finding.disease` | merged allowlist (clickable → sets focus disease) |
| **Verdict** | `finding.blurb.verdict` | supervisor LLM (`VerdictTag` badge) |
| **Evidence** | `finding.literature.evidence_summary.strength` | literature agent (`StrengthBadge`) |
| **Trials** | `finding.clinical_trials.search.total_count` | clinical_trials agent |
| **Competitors** | `finding.clinical_trials.landscape.competitors.length` | clinical_trials agent |
| **Recruiting** | `finding.clinical_trials.search.by_status["RECRUITING"]` | clinical_trials agent |

Sortable; collapses to stacked cards on mobile.

#### 1.2.2 Disease detail card — `overview/SummaryBlurbCard.tsx`

All fields are `CandidateBlurb` (supervisor LLM via `finalize_supervisor`). A row
renders only when its field is non-empty.

| Card element | Pydantic field | `supervisor.txt` block (`# WRITING THE BLURBS`, `supervisor.txt:321`) |
|---|---|---|
| Card title `{rank}. {disease}` | rank + `finding.disease` | — |
| **Verdict** badge | `blurb.verdict` | verdict guidance under `# WRITING THE BLURBS` |
| **Evidence** badge | `finding.literature.evidence_summary.strength` | — (literature agent) |
| **Stage** | `blurb.stage` | `# WRITING THE BLURBS` |
| **Literature** | `blurb.literature` | `# WRITING THE BLURBS` |
| **Constraint** | `blurb.blocker` | `# WRITING THE BLURBS` |
| **Active programs** | `blurb.active_programs` | `# WRITING THE BLURBS` |
| **Key risk** | `blurb.key_risk` | `# WRITING THE BLURBS` |
| **Watch** callout | `blurb.watch` | NCT id and/or expected timing if known; empty if no scheduled readout — do not invent timing |
| Prose paragraph | `blurb.prose` | exactly 2 sentences of interpretive synthesis |

### 1.3 Mechanism tab — `tabs/MechanismTab.tsx`

All fields come from `result.mechanism` (`MechanismOutput`) — **mechanism agent**
(`agents/mechanism/mechanism_agent.py`), which reads OpenTargets via
`mechanism_tools`.

| UI element | Pydantic field | Source |
|---|---|---|
| **Mechanistic analysis** summary | `mechanism.summary` | mechanism agent's final LLM message (`components/Markdown.tsx`) |
| **Network graph** (`graph/MechanismGraph.tsx`, react-flow) | `mechanism` + `disease_findings`, see below | derived in `graphData.ts` |
| **Molecular targets** list | `mechanism.drug_targets` keys (gene symbols) | OpenTargets `drug.linkedTargets` |
| **Mechanisms of action** table | `mechanism.mechanisms_of_action` — cols `mechanism_of_action`, `action_type`, `target_symbols` | OpenTargets `drug.mechanismsOfAction` |
| **Repurposing candidates** cards | `mechanism.candidates` — `target_symbol`, `action_type`, `disease_name`, `disease_description`, `target_function` | OpenTargets `target.associatedDiseases` evidence, filtered to non-approved indications and POSITIVE-direction matches by `mechanism_tools` |

Graph (`graphData.ts`): drug node = `drug_name`; target nodes = `drug_targets`
keys that have a candidate edge; disease nodes = intersection of
`candidates[].disease_name` and `disease_findings[].disease` (grounded +
investigated only). Drug→target edges labelled with `action_type`; target→disease
edges from `candidates`. Disease node colour by `finding.source`
(competitor / mechanism / both), size by `clinical_trials.search.total_count`.
Clicking a disease node sets the focus disease.

### 1.4 Clinical Trials tab — `tabs/ClinicalTrialsTab.tsx` (focus disease only)

All fields come from `finding.clinical_trials` (`ClinicalTrialsOutput`) for the
focused disease — **clinical_trials agent**
(`agents/clinical_trials/clinical_trials_agent.py`).

| UI element | Pydantic field | Source |
|---|---|---|
| `Source: {tag}` caption | `finding.source` (`"competitor" \| "mechanism" \| "both"`) | Allowlist tag, not an agent |
| **Total trials** / **Recruiting** / **Active (not recruiting)** KPIs | `ct.search.total_count`, `ct.search.by_status["RECRUITING"]`, `ct.search.by_status["ACTIVE_NOT_RECRUITING"]` (`SearchTrialsResult`) | clinical_trials agent → `search_trials` |
| Summary paragraph | `ct.summary` | clinical_trials agent's final LLM message |
| **Status breakdown** donut (`charts/StatusDonut.tsx`) | `ct.search.by_status` (slices, largest first, zeros dropped — `chartData.ts`) | same |
| **Phase funnel** (`charts/PhaseFunnel.tsx`) | `ct.completed.trials[].phase` counts, ordered early→late (`chartData.ts`) | clicking a bar filters the completed table |
| **Completed trials** table (`tables/CompletedTrialsTable.tsx`) | `ct.completed.trials` minus `ct.contaminated_nct_ids` (max 25 shown) — NCT, Title, Phase, Status | clinical_trials agent → `get_completed_trials`; sortable, phase/status filter chips |
| **Terminated trials** cards | `ct.terminated.trials` minus contaminated (first 15) — NCT, title, phase, `why_stopped` | clinical_trials agent → `get_terminated_trials` |
| **Competitive landscape** table (`tables/CompetitorsTable.tsx`) | `ct.landscape.competitors` (max 25) — Drug, Sponsor, Max phase, Trials | clinical_trials agent → `get_indication_landscape`; sortable |
| **Excluded trials** disclosure | `ct.completed`/`ct.terminated` trials in `ct.contaminated_nct_ids` | trials the agent judged a different indication/drug; hidden from tables but still counted in `total_count` (`trialFilter.ts`) |

FDA approval (`ct.approval`) is rendered in the Markdown report (§2.4.3) but is
**not** shown as a block in the React UI.

### 1.5 Literature tab — `tabs/LiteratureTab.tsx` (focus disease only)

All fields come from `finding.literature.evidence_summary` (`EvidenceSummary`) —
**literature agent** (`agents/literature/literature_agent.py`).

| UI element | Pydantic field | Source |
|---|---|---|
| `Source: {tag}` caption | `finding.source` | Allowlist tag, not an agent |
| **Evidence strength** KPI | `lit.strength` (`"none" \| "weak" \| "moderate" \| "strong"`) | literature agent → `synthesize` |
| **Study count** KPI | `lit.study_count` | same |
| **Summary** body | `lit.summary` | same (`components/Markdown.tsx`) |
| **Key findings** bullets | `lit.key_findings` | same |
| **Supporting PMIDs** linked list | `lit.supporting_pmids` | literature agent's selected PMIDs (subset of `finding.literature.pmids`); upstream via `PubMedClient` + `RetrievalService` (BioLORD-2023 embeddings) |

Shared components: `components/Badge.tsx` (`StrengthBadge`, `VerdictTag`,
`SourceTag`), `components/links.tsx` (`NctLink`, `PmidLink`),
`components/Markdown.tsx` (react-markdown for all agent prose).

---

## 2. Markdown report — `format_report.py`

The Markdown file follows a fixed layout assembled by `format_report(output)`
(`format_report.py:404-505`). References below cite the sample
`snapshots/imatinib_2026-06-15_00-59-47.md`.

### 2.1 Header — `format_report.py:409-417`

| Markdown element | Pydantic field | Source |
|---|---|---|
| `# IndicationScout Report: {drug}` | `output.drug_name` (title-cased) | Drug name passed to `run_supervisor_agent` |
| `_Generated {timestamp} UTC_` | none | `datetime.utcnow()` at render time |
| `_Not for clinical use; for research purposes only_` | none | static boilerplate |

### 2.2 Summary section — `format_report.py:429-454`

The block is `output.summary` (LLM-authored by the supervisor) with per-disease
`CandidateBlurb`s spliced in by `_splice_blurbs_into_summary`
(`format_report.py:334-401`). Footer lines (Demoted / Closed signals / Evidence
gate exclusions) pass through unchanged; a `---` separator is inserted before the
footer block. The supervisor LLM produces this content from two prompt sections
in `src/indication_scout/prompts/supervisor.txt`:

- **`# WRITING THE SUMMARY`** — `supervisor.txt:230` → the ranked list and footers
- **`# WRITING THE BLURBS`** — `supervisor.txt:321` → the per-candidate
  `CandidateBlurb` fields (rendered as the table + Watch + prose)

| Markdown element | Pydantic field | Source agent | `supervisor.txt` block |
|---|---|---|---|
| Heading `Ranked repurposing signals for {drug}:` | within `output.summary` | supervisor LLM (`finalize_supervisor`) | `# WRITING THE SUMMARY` |
| Ranked entries `{rank}. {disease}` | within `output.summary`; `output.top_diseases` constrains which diseases get blurbs | supervisor LLM | `# WRITING THE SUMMARY` (line format + tier ordering) |
| Per-candidate 2-column table — row **Stage** | `CandidateBlurb.stage` | supervisor LLM | `# WRITING THE BLURBS` |
| Per-candidate 2-column table — row **Literature** | `CandidateBlurb.literature` | supervisor LLM | `# WRITING THE BLURBS` |
| Per-candidate 2-column table — row **Constraint** | `CandidateBlurb.blocker` (renamed "Constraint" via `_BLURB_TABLE_FIELDS`, `format_report.py:273-280`) | supervisor LLM | `# WRITING THE BLURBS` |
| Per-candidate 2-column table — row **Active programs** | `CandidateBlurb.active_programs` | supervisor LLM | `# WRITING THE BLURBS` |
| Per-candidate 2-column table — row **Key risk** | `CandidateBlurb.key_risk` | supervisor LLM | `# WRITING THE BLURBS` |
| Per-candidate 2-column table — row **Assessment** | `CandidateBlurb.verdict` (renamed "Assessment") | supervisor LLM | `# WRITING THE BLURBS` |
| `**Watch:** ...` line | `CandidateBlurb.watch` | supervisor LLM | `# WRITING THE BLURBS` (NCT id and/or expected timing if known; empty if no scheduled readout — do not invent timing) |
| Italic 2-sentence paragraph | `CandidateBlurb.prose` | supervisor LLM | `# WRITING THE BLURBS` (exactly 2 sentences of interpretive synthesis) |
| `Demoted — approval relationship: ...` footer | within `output.summary` | supervisor LLM | `# APPROVAL RELATIONSHIPS` (`supervisor.txt:151`) — sub-templates `same` / `narrower` / `broader_overlapping` / `broader_distinct` / `combination` |
| `Closed signals: ...` footer (when present) | within `output.summary` | supervisor LLM | `# WRITING THE SUMMARY` |
| `Evidence gate exclusions: ...` footer | within `output.summary` | supervisor LLM | `# WRITING THE SUMMARY` (substantive gate rule: zero trials AND <5 PMIDs, OR zero trials AND strength=none) |
| Footer precedence (each candidate in exactly one footer) | within `output.summary` | supervisor LLM | approval-relationship demotions > closed signals > evidence gate exclusions |
| Separator `---` between blurbs and footer | none | formatter | `_splice_blurbs_into_summary` inserts it when it sees a footer line |
| `_Note: trial counts in this summary reflect ClinicalTrials.gov only…_` | none | formatter (static boilerplate) | `format_report.py` summary section |

Disease-name title-casing inside the supervisor's prose is applied by
`_title_case_known_diseases` (`format_report.py:250-270`) — a deterministic
post-processing step over the LLM's output (longest disease names first to avoid
substring shadowing), not the LLM itself. `_title_case_disease`
(`format_report.py:60-64`) title-cases individual names (drug, candidate list,
finding headers) while preserving acronyms and possessives.

### 2.3 Diseases Considered section — `format_report.py:457-469`

| Markdown element | Pydantic field | Source |
|---|---|---|
| Heading `## Diseases Considered` | none | formatter |
| `_Note: not every disease listed here…_` | none | formatter (static boilerplate) |
| Bullet list of diseases | `output.candidate_diseases` | Merged allowlist: competitor scan (OpenTargets `find_candidates`) + mechanism agent's promoted diseases, after EFO-based dedup + LLM hierarchical merge. Built in `supervisor_tools.py::merge_and_dedup`. Deterministic, not LLM-authored. |

### 2.4 Findings by Disease — per-disease section — `format_report.py:472-504`

For each `finding` in `output.disease_findings`, the formatter emits a fixed
sub-structure.

#### 2.4.1 Section header

| Markdown element | Pydantic field | Source |
|---|---|---|
| `### {Disease} _(source: {tag})_` | `finding.disease`, `finding.source` | `finding.disease` is the canonical name from the merged allowlist. `finding.source` is the allowlist tag (`"competitor"` / `"mechanism"` / `"both"`) — set when the disease was added. Not an agent. |

#### 2.4.2 Literature subsection — `format_report.py:67-108` (`_fmt_literature`)

All fields come from `finding.literature.evidence_summary` (`EvidenceSummary`) —
**literature agent**. The `synthesize` tool writes the structured object; upstream
data comes from `PubMedClient` + `RetrievalService`.

| Markdown element | Pydantic field | Source |
|---|---|---|
| `### Literature — {Disease}` heading | `finding.disease` | formatter |
| `**Evidence strength:** {strength}[, {direction}]` | `evidence_summary.strength` (+ `direction` when ≠ "none") | literature agent (`synthesize`) |
| `**Evidence strength:** class-level signal (no direct evidence…)` | when `evidence_summary.evidence_basis == "class_level"` | literature agent |
| `**Relevant studies:** {n}` | `evidence_summary.study_count` | literature agent |
| Summary paragraph | `evidence_summary.summary` | literature agent — LLM prose |
| `**Key findings:**` bullets | `evidence_summary.key_findings` | literature agent — LLM bullets |
| `**Supporting PMIDs:** [pmid](url), ...` | `evidence_summary.supporting_pmids` | literature agent's selected PMIDs |
| `**Contradicting PMIDs:** [pmid](url), ...` | `evidence_summary.contradicting_pmids` | literature agent's selected PMIDs |
| `_No evidence summary available._` (fallback) | none | formatter, when `evidence_summary` is None |

#### 2.4.3 Clinical Trials subsection — `format_report.py:118-247` (`_fmt_clinical_trials`)

All fields come from `finding.clinical_trials` (`ClinicalTrialsOutput`) —
**clinical_trials agent**.

| Markdown element | Pydantic field | Source |
|---|---|---|
| `### Clinical Trials` heading | none | formatter |
| `**Development stage:** {phrase}` | `ct.signals` (dev-stage phrase) | clinical_trials agent |
| Narrative paragraph(s) | `ct.summary` | clinical_trials agent's final LLM message |
| `_{n} trial(s) excluded as a different indication: {ncts}._` | `ct.contaminated_nct_ids` | clinical_trials agent tagged these as a different indication/drug |
| `**FDA approval:** Approved ({matched_indication})` | `ct.approval` with `label_found=True`, `is_approved=True` | clinical_trials agent → `check_fda_approval` → openFDA labels + LLM indication-match in `services/approval_check.py` |
| `**FDA approval:** Not found on FDA label for this indication` | `ct.approval` with `label_found=True`, `is_approved=False` | same |
| `**FDA approval:** No FDA label found for {drug_names} — status undetermined` | `ct.approval.drug_names_checked` with `label_found=False` | same |
| `**Trial activity:** {n} total trial(s) for this pair` | `ct.search.total_count` (`SearchTrialsResult`) | clinical_trials agent → `search_trials` |
| `- _Whitespace: no trials found…_` (zero-trials hint) | none | formatter, when `total_count == 0` |
| `**Completed trials ({n} total on record):**{count clause}` heading | `ct.completed.total_count` (`CompletedTrialsResult`) | clinical_trials agent → `get_completed_trials` |
| Completed trial bullets `- [NCT…](url) — {title} ({phase}, {status})` | `ct.completed.trials` minus `contaminated_nct_ids`, first 10 (`_TRIAL_RENDER_CAP`) | same; `- _…and {n} more…_` line when truncated |
| `**Terminated trials ({n} total on record):**{count clause}` heading | `ct.terminated.total_count` (`TerminatedTrialsResult`) | clinical_trials agent → `get_terminated_trials` |
| Terminated trial bullets `- [NCT…](url) {title} ({phase})[{category}] — *{why_stopped}*` | `ct.terminated.trials` minus contaminated, first 10 — `[category]` is `_classify_stop_reason(why_stopped)`, deterministic Python in `clinical_trials_tools.py`, NOT LLM | same; category is post-processing |
| `_No clinical trials data available._` (fallback) | none | formatter, when no other lines emitted |

The `{count clause}` on the trial-section headers is `_trial_count_clause`
(`format_report.py:26-57`): it reports how many trials were shown vs. hidden as a
different indication, and whether only a slice of the on-record total was fetched.

**Contaminated-relationship suppression.** When `finding.blurb.approval_relationship`
is in `_CONTAMINATED_RELATIONSHIPS` (`broader_distinct` / `broader_overlapping`),
`_fmt_clinical_trials` omits the completed/terminated trial tables and emits a
short note (`…not listed — trial record contaminated by approved subtype…`)
instead, because the trial record is dominated by the already-approved subtype.
Total counts are still shown.

Note: `ct.landscape` (`IndicationLandscape`) is populated by the clinical_trials
agent but is **not** rendered as a separate block by the Markdown formatter — it
only surfaces indirectly when the agent quotes it inside `ct.summary`. The React
UI renders it explicitly (§1.4 Competitive landscape); the Markdown report does
not.

---

## 3. Agent / data-source summary

| Agent / source | Where it writes into `SupervisorOutput` | Underlying data |
|---|---|---|
| **Supervisor agent** (`supervisor_agent.py`) | `output.summary`, `output.top_diseases`, every `CandidateFindings.blurb` | LLM call via `finalize_supervisor` tool, ranking over what the other agents returned |
| **Mechanism agent** (`mechanism_agent.py`) | `output.mechanism` (`MechanismOutput`) | OpenTargets GraphQL via `OpenTargetsClient`, `RichDrugData` model |
| **Literature agent** (`literature_agent.py`) | each `CandidateFindings.literature` (`LiteratureOutput`), notably `.evidence_summary` (`EvidenceSummary`) | PubMed via `PubMedClient` + semantic retrieval via `RetrievalService` (BioLORD-2023 embeddings) |
| **Clinical trials agent** (`clinical_trials_agent.py`) | each `CandidateFindings.clinical_trials` (`ClinicalTrialsOutput`) — `.search`, `.completed`, `.terminated`, `.landscape`, `.approval`, `.contaminated_nct_ids`, `.summary` | ClinicalTrials.gov via `ClinicalTrialsClient`; FDA approval via `FDAClient` + `services/approval_check.py` |
| **Merged allowlist** (deterministic, `supervisor_tools.py`) | `output.candidate_diseases`, each `CandidateFindings.source` tag | Competitor scan (OpenTargets, drugs sharing the same molecular target) + mechanism-promoted diseases, deduped by EFO ID + LLM hierarchical merge |
| **Format/UI layer** | none — reads `SupervisorOutput` | `format_report.py` (Markdown); `frontend/src/` (React, consumes JSON via `/api/analyses`) |
