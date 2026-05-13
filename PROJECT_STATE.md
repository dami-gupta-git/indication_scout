# IndicationScout — Project State Snapshots

Dated snapshots of implementation status, architectural decisions, and known issues.
For detailed findings and patterns, see `docs/findings.md`.

---

## Update (2026-05-10)

### Implementation Status Changes
- **literature_tools.expand_search_terms**: Complete — fixed bypass of production path (was skipping expansion entirely).
- **disease_helper.resolve_mesh_id**: Complete — rewrote to parse MeSH preferred term from esearch `querytranslation` instead of unreliable esummary endpoint. Added retry-on-empty (3 attempts, 2s spacing) and 5-concurrent semaphore.
- **pubmed_query expansion**: Partial — fixed PubMed `<term> AND <multi-word>` parser breakage by resolving disease to MeSH preferred term and quoting verbatim in LLM prompt. Added `PUBMED_SEARCH_SLEEP_SECONDS = 1.0` pre-call sleep and zero-PMID warning log.
- **supervisor FDA approval output**: Complete — removed per-disease "FDA approval: yes/no" fields from summary line formats and report blurbs. Added "combination component" classification to APPROVAL RELATIONSHIPS rules.
- **retrieval.py caching**: Complete — documented file cache vs pgvector roles in `RetrievalService.fetch_and_cache` (cache TTL 5 days, scratch cache strategy for experiments).

### New Patterns / Decisions
- **Query expansion is now the default**. Expansion produces richer prose (regulatory context, competitive nuance, off-label-vs-unmet-need) at cost of older citations and lower strength labels. Verdict-diff metrics under-measure expansion's value; prose quality is the better signal.
- **FDA approval context belongs in prose, not header fields**. Every candidate is unapproved for the candidate disease by definition; approval relationships (same/narrower/broader/combination) belong inline in blurbs.
- **MeSH preferred terms remain lowercase** per project convention.
- **Scratch cache strategy**: Redirect `DEFAULT_CACHE_DIR` to experiment-specific path (e.g., `methods/query_expansion/_scratch_cache`) for duration of run, then revert. No cache bypass needed.
- **Diff metrics are insufficient**. Verdict-flip count and PMID-set overlap under-measure expansion's qualitative contribution. Use prose quality and currency as primary lens.

### Known Issues / Caveats Added
- **PubMed esummary unreliable**: NCBI's `db=mesh` esummary returns empty records for valid UIDs. Workaround: parse `querytranslation` from esearch instead.
- **MeSH resolver intermittent flakiness**: NCBI backend occasionally returns empty `idlist=[]` for valid terms. Mitigated by retry-on-empty (3 attempts, 2s spacing), but underlying flakiness is real.
- **PubMed automatic term mapping silently zeros queries** with bare multi-word phrases on either side of `AND`. Quoting alone insufficient; resolution to MeSH preferred term + verbatim quoting required.
- **Bupropion obesity misrank (fixed)**: Combination products (Contrave) triggered "literature: strong + FDA approval: no" which LLM read as repurposing opportunity. Fixed via APPROVAL RELATIONSHIPS "combination component" rule — obesity now ranks #5 instead of #1.
- **expand_search_terms tool bypassed production path**: Unknown duration before this session. Production was effectively running with no expansion. Fixed.

---

## Update (2026-05-11)

### Implementation Status Changes
- **clinical_trials_tools filters**: Complete — removed 3 client-side quality filters (MeSH primacy, LLM primacy, drug-alias intervention) from `search_trials` / `get_completed` / `get_terminated`. Holdout scrubber (date-based) preserved. Rationale: tools surface CT.gov's view; judgment lives at LLM layer.
- **clinical_trials_tools labels**: Complete — updated content header (removed status counts parenthetical), renderer sub-bullets, and `prompts/clinical_trials.txt` to forbid per-status enumeration.
- **hierarchical LLM dedup**: Partial (Stub) — commented out the pass at `supervisor_tools.py:470-528` (was collapsing PCOS/hepatic steatosis/gestational diabetes into "metabolic disease"). Exact-match dedup steps (EFO-ID, name, OT name-resolve) still run.
- **supervisor candidate investigation**: Complete — replaced "pick 3-5" with deterministic list-order cap of 6 candidates. Fixes non-determinism observed in bupropion runs ~5 hours apart.
- **trial references PubMed fallback**: Complete — `ClinicalTrialsClient._augment_references_via_pubmed` queries PubMed for `NCT...[si]` when CT.gov returns empty `references`. ~70% hit rate on tested NCTs. Wired into `get_completed_trials` and `get_terminated_trials`.
- **trial refs column**: Complete — added `refs:` column to trial tables in `_trial_formatting.py`. Wired into `analyze_clinical_trials` formatter with prompt directives crediting completed trials with linked PMIDs.
- **report section/label renames**: Complete — `Study count` → `Relevant studies`; `Supporting PMIDs` → `Relevant PMIDs (favorable only)`; `## Candidate Diseases` → `## Diseases Considered`; `## Candidate Findings` → `## Findings by Disease`. Removes endorsement undertone.
- **broader_overlapping approval rule**: Complete — demoted from ranked candidate to footer (alongside broader_distinct). Explicit "do NOT invent scope-narrowed analysis" guard. Triggered by semaglutide × NAFLD hallucination.
- **phase grounding and phase-vs-maturity directives**: Complete — added to `prompts/supervisor.txt`. CT.gov phase field is verbatim (Phase 2 UNKNOWN stays Phase 2, not downgraded). Phase 2/Phase 3 design tag is not pivotal-scale maturity; cross-check enrollment (n<200 → Phase 2-scale).

### New Patterns / Decisions
- **Tools surface raw CT.gov data; all filtering is inappropriate.** The supervisor agent reads titles + MeSH per row and judges individually. Client-side quality filters (MeSH primacy, LLM primacy, drug-alias) hide data, preventing the agent from making informed decisions. Only date-based holdout scrubbing remains (real data-leakage guard).
- **Hard rules in LLM prose don't reliably constrain behavior.** The dedup prompt's "HARD RULE: clinical relevance is NOT your concern" was ignored repeatedly (PCOS/hepatic steatosis collapsed to "metabolic disease"). Switch to curated deterministic lookups (planned `DEDUP_GROUPS` table) instead.
- **Cap supervisor investigation at 6 candidates.** Matches observed ceiling in practice (bupropion 7, semaglutide ~6, metformin 5). Costs 30–40% more agent calls, gains full report-to-report determinism.
- **Approval-relationship framing is distinct from approval status.** Broader/narrower/combination context belongs in prose, not header fields. Every candidate is unapproved for the candidate disease by definition.
- **Phase field grounding.** CT.gov phase is a design declaration, not maturity signal. Phase 2/Phase 3 (both listed) is valid; verify enrollment size (n<200 → characterize as Phase 2-scale, not pivotal).
- **Label naming is self-documenting.** "Relevant studies" vs "Relevant PMIDs (favorable only)" makes the gap explicit rather than contradictory-seeming.

### Known Issues / Caveats Added
- **PubMed indexing gap for older trial readouts.** PMIDs 26800231 (Papp 2016 baricitinib×psoriasis) and 15820237 (Wilens 2005 bupropion×ADHD) have no NCT cross-references in any indexed field. NCT-tag fallback has ~70% hit rate; pre-~2018 papers often unlinked. Accepted as known limitation per findings.md.
- **PubMed default sort is date, not relevance.** Older landmark papers sink in results for heavily-studied drug-disease pairs (e.g., `bupropion AND adhd` returns 246 results, Wilens 2005 at rank 145). No fix shipped; candidate: sort=relevance, higher PUBMED_MAX_RESULTS, or title/author fallback.
- **Clinical-trial filter bugs (root-cause identified, filter removed).** (1) `total_count` extrapolation when filtering top-50 sample; (2) `by_status` inconsistency (per-status counts from unfiltered population, total_count decremented from sample). Fixed by removing filtering entirely.
- **Demoted candidates still get full sections.** "Demoted" annotation not added to heading (e.g., "[demoted]"), but user accepts this — section reflects "what was investigated," not "what was endorsed."
- **Active programs blurb still includes partial status counts.** Supervisor LLM reads `by_status` artifact directly despite stripped tool header. Not fixed this session.
- **Prompts not tracked in git.** `*.txt` glob in `.gitignore` (line 72) keeps entire `prompts/` directory untracked. Prompt edits invisible to git history (could not trace broader_overlapping introduction). Deliberate workflow choice per user.

---

## Update (2026-05-11)

### Implementation Status Changes
- **regression testing harness**: Complete — added `src/indication_scout/regression/` (constants, diff, harness modules) with `compare_reports(golden, current) -> list[Diff]` for structure + semantic overlap (Jaccard on candidate/top_diseases, presence of sub-payloads, prose length bounds, tolerance on counts). Free-text never exact-matched.
- **regression test suite**: Complete — `tests/regression/cassette.py` (vcrpy wiring, env-var mode), `test_harness.py` (15 unit tests, all passing), `test_pipeline_regression.py` (marker-gated metformin full-pipeline). vcrpy 7.0.0+ added to dev deps; `regression` and `live` markers in pytest.ini with both excluded by default.
- **CLI regression subcommand**: Complete — `scout diff-report <golden.json> <current.json>` exits non-zero on error-severity diffs.

### New Patterns / Decisions
- **Snapshot boundary: final + per-agent.** `SupervisorOutput` nests specialist outputs under `disease_findings` and `mechanism`; one `model_dump_json` per drug captures everything (no separate wrapper needed).
- **HTTP-layer cassettes, not SDK-level shims.** vcrpy records aiohttp + httpx traffic in one YAML file. A future third LLM entry point cannot bypass HTTP recording; SDK-level shims would be silently circumvented.
- **vcrpy directly (not pytest-recording).** pytest-recording does not support aiohttp; vcrpy 7+ covers both sync and async clients in one cassette.
- **Thresholds in `src/indication_scout/regression/constants.py`** so CLI and tests share them. Starting values (`CANDIDATE_SET_JACCARD_MIN=0.7`, `TOP_DISEASES_JACCARD_MIN=0.6`, `EVIDENCE_COUNT_TOLERANCE=5`) deliberately rough — to be calibrated after 2–3 back-to-back record runs at temperature=0.
- **Drift severity: numeric within tolerance is warn (non-failing); structural problems are error (failing).** Missing required fields, set-divergence below Jaccard min, empty summary, invariant violation → test fails. Free-text length bounded but never compared.
- **DB stays live at replay time.** Cassette stubs external HTTP + LLM; Postgres + pgvector treated as test environment (same as integration suite).
- **Golden-file promotion from scout find.** `test_reports/<drug>_<timestamp>.json` is the same `SupervisorOutput` shape — can be copied and renamed to golden without re-running record mode.

### Known Issues / Caveats Added
- **pytest.ini overrides pyproject.toml markers silently.** Marker + addopts declarations in `pyproject.toml` are inert when `pytest.ini` exists. Moved markers to `pytest.ini`; pyproject.toml block remains as fallback.
- **Harness module import refactoring done mid-session.** Initial design had `tests/regression/harness.py` and CLI tried to import from `tests.*` (namespace pollution). Moved harness + diff into `src/indication_scout/regression/` so both test and CLI can import cleanly.
- **Pre-existing collection errors unrelated to this work.** `for_me/literature/v1_react/test_clinical_trials_agent_bak.py` (outside tests/) and `tests/integration/agents/mechanism/test_mechanism_tools.py` — left untouched.

