# Report description

Where every block in the Streamlit UI and the downloadable Markdown report comes
from — which Pydantic field, which agent, and (for supervisor-authored prose)
which block of `supervisor.txt` instructs it.

The two rendering paths share the same `SupervisorOutput` object but format it
differently:

- **Streamlit UI** — `app.py`, renders interactively.
- **Markdown report** — `src/indication_scout/report/format_report.py`, produces
  the `.md` file the user downloads (and the snapshots in `snapshots/`).

`SupervisorOutput` itself is built by
`src/indication_scout/agents/supervisor/supervisor_agent.py::run_supervisor_agent`.

---

## 1. Streamlit UI — `app.py`

### 1.1 Header band — `app.py:172-191`

| UI element | Pydantic field | Source |
|---|---|---|
| Title `Indication Scout — {drug}` | `analysed_drug` (session_state) | Sidebar text input |
| `_Generated {timestamp}_` | none | Local render time |
| **Candidate diseases** metric | `output.candidate_diseases` | Merged allowlist (deterministic): competitor scan (OpenTargets, drugs sharing same target) + mechanism-promoted diseases. Built in `agents/supervisor/supervisor_tools.py::find_candidates` + `merge_and_dedup`. Not LLM-authored. |
| **Investigated** metric | `len(output.disease_findings)` | Subset of the allowlist that the supervisor LLM chose to call `analyze_literature` / `analyze_clinical_trials` on |
| **Total trials** metric | sum of `finding.clinical_trials.search.total_count` over `disease_findings` | clinical_trials agent (`search_trials` tool → ClinicalTrials.gov via `ClinicalTrialsClient`) |
| **Total studies** metric | sum of `finding.literature.evidence_summary.study_count` over `disease_findings` | literature agent (`synthesize` tool → judgment over PubMed abstracts retrieved by `PubMedClient` + ranked by `RetrievalService`) |

### 1.2 Overview tab — `app.py:205-229`

| UI element | Pydantic field | Source |
|---|---|---|
| **Summary** body | `output.summary` (spliced with `_splice_blurbs_into_summary`) | Supervisor LLM via the `finalize_supervisor` tool. Splice merges per-disease `CandidateBlurb`s into the ranked summary string. |
| **Candidate diseases** list with `✓ investigated` / `_not investigated_` markers | `output.candidate_diseases` vs `{f.disease for f in output.disease_findings}` | Merged allowlist; the ✓ flag is computed locally |
| **Full markdown report** expander | `st.session_state["report_md"]` | `format_report(output)` — the Markdown formatter (see §2) |

### 1.3 Mechanism tab — `app.py:233-276`

All fields come from `output.mechanism` (`MechanismOutput`) — **mechanism agent**
(`agents/mechanism/mechanism_agent.py`). The agent reads OpenTargets via
`mechanism_tools` and produces the structured output.

| UI element | Pydantic field | Source |
|---|---|---|
| **Mechanistic analysis** summary text | `mech.summary` | mechanism agent's final LLM message |
| **Molecular targets** bullet list | `mech.drug_targets` (gene symbol → Ensembl ID) | OpenTargets `drug.linkedTargets` |
| **Mechanisms of action** dataframe | `mech.mechanisms_of_action` (`list[MechanismOfAction]`) — columns: `mechanism_of_action`, `action_type`, `target_symbols` | OpenTargets `drug.mechanismsOfAction` |
| **Repurposing candidates from mechanism** containers | `mech.candidates` (`list[MechanismCandidate]`) — fields: `target_symbol`, `action_type`, `disease_name`, `disease_description`, `target_function` | OpenTargets `target.associatedDiseases` evidence, filtered to non-approved indications and POSITIVE-direction matches by `mechanism_tools` |

### 1.4 Clinical Trials tab — `app.py:280-359`

All fields come from `finding.clinical_trials` (`ClinicalTrialsOutput`) for the
selected disease — **clinical_trials agent**
(`agents/clinical_trials/clinical_trials_agent.py`).

| UI element | Pydantic field | Source |
|---|---|---|
| `Source: {tag}` caption | `finding.source` (`"competitor" \| "mechanism" \| "both"`) | Allowlist tag, not an agent. Set when the disease was added to the merged allowlist. |
| Summary paragraph | `ct.summary` | clinical_trials agent's final LLM message |
| **Total trials** / **Recruiting** / **Active (not recruiting)** metrics | `ct.search.total_count`, `ct.search.by_status["RECRUITING"]`, `ct.search.by_status["ACTIVE_NOT_RECRUITING"]` (`SearchTrialsResult`) | clinical_trials agent → `search_trials` tool → ClinicalTrials.gov |
| **Status breakdown** bar chart | `ct.search.by_status` (sorted desc) | same as above |
| **Completed trials** dataframe (NCT, Title, Phase, Status) | `ct.completed.trials[:25]` (`CompletedTrialsResult`) | clinical_trials agent → `get_completed_trials` tool |
| **Terminated trials** containers (NCT, title, phase, stop category, `why_stopped`) | `ct.terminated.trials[:15]` (`TerminatedTrialsResult`); stop category from `_classify_stop_reason(t.why_stopped)` | clinical_trials agent → `get_terminated_trials` tool; `_classify_stop_reason` is deterministic Python in `clinical_trials_tools.py` |
| **Competitive landscape** dataframe (Drug, Sponsor, Max phase, Trials) | `ct.landscape.competitors[:25]` (`IndicationLandscape`) | clinical_trials agent → `get_indication_landscape` tool |
| (FDA approval block — removed) | previously `ct.approval` (`ApprovalCheck`) | (was) clinical_trials agent → `check_fda_approval` tool → openFDA labels + LLM extraction in `services/approval_check.py`. **Removed from the UI**, still rendered in the Markdown report. |

### 1.5 Literature tab — `app.py:363-397`

All fields come from `finding.literature.evidence_summary` (`EvidenceSummary`)
for the selected disease — **literature agent**
(`agents/literature/literature_agent.py`).

| UI element | Pydantic field | Source |
|---|---|---|
| `Source: {tag}` caption | `finding.source` | Allowlist tag, not an agent |
| **Evidence strength** metric | `lit.strength` (`"none" \| "weak" \| "moderate" \| "strong"`) | literature agent → `synthesize` tool, LLM judgment over retrieved abstracts |
| **Study count** metric | `lit.study_count` | same as above |
| **Summary** body | `lit.summary` | same as above |
| **Key findings** bullets | `lit.key_findings` | same as above |
| **Supporting PMIDs** linked list | `lit.supporting_pmids` | literature agent's selected PMIDs (subset of `finding.literature.pmids`); upstream data via `PubMedClient` + `RetrievalService` (BioLORD-2023 embeddings) |

---

## 2. Markdown report — `format_report.py`

The Markdown file follows a fixed layout assembled by
`format_report(output)`. References below cite the sample
`snapshots/semaglutide_2026-05-14_16-47-04.md`.

### 2.1 Header — `format_report.py:277-283`

| Markdown element | Pydantic field | Source |
|---|---|---|
| `# IndicationScout Report: {drug}` | `output.drug_name` | Drug name passed to `run_supervisor_agent` |
| `_Generated {timestamp}_` | none | `datetime.utcnow()` at render time |

### 2.2 Summary section — `format_report.py:295-320`

The block is `output.summary` (LLM-authored by the supervisor) with per-disease
`CandidateBlurb`s spliced in by `_splice_blurbs_into_summary`. The supervisor
LLM produces this content by following two prompt sections, both in
`src/indication_scout/prompts/supervisor.txt`:

- **`# WRITING THE SUMMARY`** — `supervisor.txt:197-275` → the ranked list and
  footer lines
- **`# WRITING THE BLURBS`** — `supervisor.txt:277-374` → the per-candidate
  `CandidateBlurb` fields (rendered as the 6-row table + Watch + prose)

| Markdown element | Pydantic field | Source agent | `supervisor.txt` block |
|---|---|---|---|
| Heading `Ranked repurposing signals for {drug}:` | within `output.summary` | supervisor LLM (`finalize_supervisor`) | `supervisor.txt:203-210` (FORMAT rule mandating this exact heading; "signals" not "candidates") |
| Ranked entries `{rank}. {disease}` | within `output.summary`; `output.top_diseases` constrains which diseases get blurbs | supervisor LLM | `supervisor.txt:212` (line format), `supervisor.txt:244-257` (absolute tier ordering: Tier 1 = Phase 3+ completed, Tier 2 = Phase 2 completed, Tier 3 = Phase 4 only, Tier 4 = Phase 1 only, Tier 5 = no trials) |
| Per-candidate 2-column table — row **Stage** | `CandidateBlurb.stage` | supervisor LLM | `supervisor.txt:284-289` |
| Per-candidate 2-column table — row **Literature** | `CandidateBlurb.literature` | supervisor LLM | `supervisor.txt:290-296` |
| Per-candidate 2-column table — row **Constraint** | `CandidateBlurb.blocker` (field renamed to "Constraint" in the table; see `_BLURB_TABLE_FIELDS`, `format_report.py:141-148`) | supervisor LLM | `supervisor.txt:297-299` |
| Per-candidate 2-column table — row **Active programs** | `CandidateBlurb.active_programs` | supervisor LLM | `supervisor.txt:300-301` |
| Per-candidate 2-column table — row **Key risk** | `CandidateBlurb.key_risk` | supervisor LLM | `supervisor.txt:302-304` |
| Per-candidate 2-column table — row **Assessment** | `CandidateBlurb.verdict` (renamed to "Assessment" in the table) | supervisor LLM | `supervisor.txt:305-312` |
| `**Watch:** ...` line | `CandidateBlurb.watch` | supervisor LLM | `supervisor.txt:313-315` ("NCT id and/or expected timing if known… Empty string if no scheduled readout — DO NOT invent or estimate timing") |
| Italic 2-sentence paragraph | `CandidateBlurb.prose` | supervisor LLM | `supervisor.txt:320-326` ("EXACTLY 2 sentences of interpretive plain-text synthesis…") |
| `Demoted — approval relationship: ...` footer | within `output.summary` | supervisor LLM | `supervisor.txt:118-195` (the `# APPROVAL RELATIONSHIPS` section). Sub-templates: `same` at `:122-125`; `narrower` at `:127-131`; `broader_overlapping` at `:138-140`; `broader_distinct` at `:144-146`; `combination` at `:171-177`. Footer line format at `:261-264`. |
| `Closed signals: ...` footer (when present) | within `output.summary` | supervisor LLM | `supervisor.txt:266-267` |
| `Evidence gate exclusions: ...` footer | within `output.summary` | supervisor LLM | `supervisor.txt:269-270` (line format), `supervisor.txt:367-373` (substantive gate rule: zero trials AND <5 PMIDs, OR zero trials AND strength=none) |
| Footer precedence (each candidate in exactly one footer) | within `output.summary` | supervisor LLM | `supervisor.txt:272-273` (approval-relationship demotions > closed signals > evidence gate exclusions) |
| Separator `---` between blurbs and footer | none | formatter | `format_report.py:235-243` (inserted by `_splice_blurbs_into_summary` when it sees a footer line) |
| `_Note: trial counts in this summary reflect ClinicalTrials.gov only…_` | none | formatter (static boilerplate) | `format_report.py:312-316` |

Disease-name title-casing inside the supervisor's prose is applied by
`_title_case_known_diseases` (`format_report.py:118-138`) — a deterministic
post-processing step over the LLM's output, not the LLM itself.

### 2.3 Diseases Considered section — `format_report.py:322-335`

| Markdown element | Pydantic field | Source |
|---|---|---|
| Heading `## Diseases Considered` | none | formatter |
| `_Note: not every disease listed here…_` | none | formatter (static boilerplate) |
| Bullet list of diseases | `output.candidate_diseases` | Merged allowlist: competitor scan (OpenTargets `find_candidates`) + mechanism agent's promoted diseases, after EFO-based dedup + LLM hierarchical merge. Built in `supervisor_tools.py::merge_and_dedup`. Deterministic, not LLM-authored. |

### 2.4 Findings by Disease — per-disease section — `format_report.py:338-365`

For each `finding` in `output.disease_findings`, the formatter emits a fixed
sub-structure.

#### 2.4.1 Section header

| Markdown element | Pydantic field | Source |
|---|---|---|
| `## {Disease} _(source: {tag})_` | `finding.disease`, `finding.source` | `finding.disease` is the canonical name from the merged allowlist. `finding.source` is the allowlist tag (`"competitor"` / `"mechanism"` / `"both"`) — set when the disease was added to the allowlist. Not an agent. |

#### 2.4.2 Literature subsection — `format_report.py:27-49` (`_fmt_literature`)

All fields come from `finding.literature.evidence_summary` (`EvidenceSummary`) —
**literature agent**. The literature agent's `synthesize` tool writes the
structured object; upstream data comes from `PubMedClient` + `RetrievalService`.

| Markdown element | Pydantic field | Source |
|---|---|---|
| `### Literature — {Disease}` heading | `finding.disease` | formatter |
| `**Evidence strength:** {strength}` | `evidence_summary.strength` | literature agent (`synthesize`) — `"none" \| "weak" \| "moderate" \| "strong"` |
| `**Relevant studies:** {n}` | `evidence_summary.study_count` | literature agent (`synthesize`) |
| Summary paragraph | `evidence_summary.summary` | literature agent (`synthesize`) — LLM prose |
| `**Key findings:**` bullets | `evidence_summary.key_findings` | literature agent (`synthesize`) — LLM bullets |
| `**Supporting PMIDs:** [pmid](url), ...` | `evidence_summary.supporting_pmids` (rendered as PubMed links) | literature agent's selected PMIDs |
| `_No evidence summary available._` (fallback) | none | formatter, when `evidence_summary` is None |

#### 2.4.3 Clinical Trials subsection — `format_report.py:52-115` (`_fmt_clinical_trials`)

All fields come from `finding.clinical_trials` (`ClinicalTrialsOutput`) —
**clinical_trials agent**.

| Markdown element | Pydantic field | Source |
|---|---|---|
| `### Clinical Trials` heading | none | formatter |
| Narrative paragraph(s) at the top | `ct.summary` | clinical_trials agent's final LLM message |
| `**FDA approval:** Approved ({matched_indication})` | `ct.approval.is_approved`, `ct.approval.matched_indication` (when `ct.approval.label_found` is True and `is_approved` is True) | clinical_trials agent → `check_fda_approval` tool → openFDA labels + LLM indication-match in `services/approval_check.py` |
| `**FDA approval:** Not found on FDA label for this indication` | `ct.approval` with `label_found=True`, `is_approved=False` | same as above |
| `**FDA approval:** No FDA label found for {drug_names_checked} — status undetermined` | `ct.approval.drug_names_checked` (when `label_found=False`) | same as above |
| `**Trial activity:** {n} total trial(s) for this pair` | `ct.search.total_count` (`SearchTrialsResult`) | clinical_trials agent → `search_trials` tool → ClinicalTrials.gov |
| `- _Whitespace: no trials found…_` (zero-trials hint) | none | formatter, when `total_count == 0` |
| `**Completed trials ({n} total):**` heading | `ct.completed.total_count` (`CompletedTrialsResult`) | clinical_trials agent → `get_completed_trials` tool |
| Completed trial bullets `- [NCT…](url) — {title} ({phase}, {status})` | `ct.completed.trials[:10]` — each `Trial` has `nct_id`, `title`, `phase`, `overall_status` | same |
| `**Terminated trials ({n}):**` heading | `ct.terminated.total_count` (`TerminatedTrialsResult`) | clinical_trials agent → `get_terminated_trials` tool |
| Terminated trial bullets `- [NCT…](url) {title} ({phase})[{category}] — *{why_stopped}*` | `ct.terminated.trials[:10]` — each `Trial` has `nct_id`, `title`, `phase`, `why_stopped`. `[category]` is `_classify_stop_reason(why_stopped)` — deterministic Python in `clinical_trials_tools.py`, NOT LLM. | same; category is post-processing |
| `_No clinical trials data available._` (fallback) | none | formatter, when no other lines emitted |

Note: `ct.landscape` (`IndicationLandscape`) is populated by the clinical_trials
agent but is **not** rendered as a separate block by the Markdown formatter. The
landscape only surfaces indirectly when the clinical_trials agent quotes it inside
`ct.summary` (e.g. "9 identified competitors"). The Streamlit UI renders it
explicitly; the Markdown report does not.

---

## 3. Agent / data-source summary

| Agent / source | Where it writes into `SupervisorOutput` | Underlying data |
|---|---|---|
| **Supervisor agent** (`supervisor_agent.py`) | `output.summary`, `output.top_diseases`, every `CandidateFindings.blurb` | LLM call via `finalize_supervisor` tool, ranking over what the other agents returned |
| **Mechanism agent** (`mechanism_agent.py`) | `output.mechanism` (`MechanismOutput`) | OpenTargets GraphQL via `OpenTargetsClient`, `RichDrugData` model |
| **Literature agent** (`literature_agent.py`) | each `CandidateFindings.literature` (`LiteratureOutput`), notably `.evidence_summary` (`EvidenceSummary`) | PubMed via `PubMedClient` + semantic retrieval via `RetrievalService` (BioLORD-2023 embeddings) |
| **Clinical trials agent** (`clinical_trials_agent.py`) | each `CandidateFindings.clinical_trials` (`ClinicalTrialsOutput`) — `.search`, `.completed`, `.terminated`, `.landscape`, `.approval`, `.summary` | ClinicalTrials.gov via `ClinicalTrialsClient`; FDA approval via `FDAClient` + `services/approval_check.py` |
| **Merged allowlist** (deterministic, `supervisor_tools.py`) | `output.candidate_diseases`, each `CandidateFindings.source` tag | Competitor scan (OpenTargets, drugs sharing the same molecular target) + mechanism-promoted diseases, deduped by EFO ID + LLM hierarchical merge |
| **Format/UI layer** | none — reads `SupervisorOutput` | `format_report.py` (Markdown); `app.py` (Streamlit) |
