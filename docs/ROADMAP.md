# Roadmap

Open ideas and known gaps. Finished plans live in `plans/done/`; open plans in `plans/open/`.

## 1. Integrate with TxGNN to provide drug priors

## 2. Safety/adverse-effect signal is structurally absent from evidence synthesis

### Background
System currently grades evidence-FOR repurposing (efficacy: supports/contradicts/mixed/none) but has
no way to express evidence-AGAINST on safety grounds. `for_me/findings.md` flagged this design gap
(2026-07-10): "weak-but-harmful reads identical to weak-but-benign."

### Investigation (2026-07-17)
Tried adding "safety signal, and adverse effects" to the `semantic_search()` query string
(`retrieval.py` ~line 673-679) to see if pgvector re-ranking would surface safety literature into
reports.

- **Isolated re-rank test** (cosine similarity over cached embeddings, no live pipeline): the phrase
  addition DOES change results. Top-50 overlap with the original query: metformin/PCOS 50/50 (no
  effect), sildenafil/Raynaud 42/50 (8 new CV/hemodynamic-safety papers), rofecoxib/osteoarthritis
  43/50 (7 new CV/renal-toxicity papers) once the cache was backfilled from 94→571 rofecoxib
  abstracts via a full `scout find` run. Effect is drug-dependent: real when the drug has its own
  safety literature (sildenafil, rofecoxib), a wash otherwise (metformin, semaglutide pulled in
  off-topic T2DM reviews instead of NAFLD-specific safety content).
- **Full pipeline test** (`scout find` with the query string edit live) for rofecoxib, bupropion,
  metformin: reports came back **byte-identical** to pre-change baselines (rofecoxib, metformin) or
  showed only a wording shuffle attributable to known LLM non-determinism (bupropion). The
  re-ranking change had zero visible effect on the final report.
- **Traced why:** `SEMANTIC_SEARCH_TOP_K=5` (`.env.constants`) keeps only the top 5 post-rerank
  abstracts before they reach `synthesize()` — for rofecoxib/pain, the safety papers didn't make the
  cut; the top-5 stayed pure efficacy RCTs. `synthesize()`'s cache is keyed on the *sorted PMID set*
  (`retrieval.py:822-823`, deliberate — collapses reorderings to one entry), so even a query-string
  change with unchanged top-5 membership is a guaranteed cache hit, masking any effect further.
- **Root cause confirmed by direct injection** (`inject_safety_synthesize.py`, throwaway script):
  called `synthesize()` directly for rofecoxib×pain with the real efficacy top-5 PLUS 7 real cached
  safety/CV-toxicity PMIDs (e.g. PMID 27719647 "Cardiovascular Toxicity of Cyclooxygenase
  Inhibitors", PMID 15093759 VIGOR trial life-expectancy tradeoff). **All 7 safety papers were
  classified `contaminated` and had zero effect on `strength`/`direction`/`summary`** — identical
  output to the efficacy-only baseline.
- **Prompt-level cause**: `prompts/synthesize.txt` line 25 ("THERAPEUTIC-INTENT MISMATCH... a
  different condition... → contaminated") and the fact that `direction` only has
  supports/contradicts/mixed/none, all graded against efficacy — a paper studying rofecoxib's CV
  toxicity isn't "supporting" (doesn't favor the drug) or "contradicting" per the schema's own
  definition (line 55: contradicts = "fails to work", not "causes harm"), so it structurally falls
  through to contaminated. Not a fixable prompt tweak — `EvidenceSummary` has no field for a safety
  axis orthogonal to efficacy.

### What a real fix would need
- A distinct safety/harm dimension in `EvidenceSummary` (e.g. `safety_signal` +
  `adverse_direction`), classified separately from the efficacy contaminated/supporting/contradicting
  verdict — not folded into the existing direction field.
- A second, dedicated safety-focused `semantic_search` pass (or safety-specific keyword queries added
  to `expand_search_terms`) so safety literature reliably enters the candidate PMID pool in the first
  place — top_k=5 on an efficacy-weighted query will rarely include it.
- Threading the new safety field through to `finalize_supervisor`/blurb-writing so it actually
  surfaces in ranking and report text, per the original 2026-07-10 finding.
- NOT a fix: just appending "safety signal, adverse effects" to the existing query string. Confirmed
  ineffective end-to-end even where it changes intermediate re-ranking.

## 3. Preserve class-level evidence direction

`judge_literature_strength` flattens direction to "none" whenever `evidence_basis != "drug_specific"`
(enforced in `_parse_strength`), so a meaningful class-level negative — e.g. the whole GLP-1 class
failed in Parkinson's (NLY01 negative) — is discarded. Safe (omission over inaccuracy) but
informative for a repurposing tool. Fix additively: a SEPARATE `class_direction` field on
`LiteratureStrength` / `EvidenceSummary`, populated only for `class_level`, rendered as distinct
prose ("class-level signal: GLP-1 class failed in PD") and kept OUT of `es.strength` /
`es.direction` so the supervisor ranking path stays clean.

## 4. Clinical trials query quality

- Expand drug synonyms before querying. ChEMBL/DrugBank already have synonym lists. Pass the top N
  synonyms as an OR query (e.g. `metformin OR glucophage OR dimethylbiguanide`) to `query.intr`.
  This addresses false whitespace signals caused by trials registered under brand names or salt
  forms rather than the INN.
- Dedup competitor entries case-insensitively. `data_sources/clinical_trials.py` keys competitors on
  `f"{t.sponsor}|{drug_name}"` using RAW strings, so casing/whitespace variants of the same drug
  from one sponsor produce duplicate rows — e.g. semaglutide's competitive landscape shows
  "Efimosfermin alfa"/"Efimosfermin Alfa" (both GSK) and "denifanstat"/"Denifanstat" (both Sagimet)
  as separate entries, inflating the count. Fix: normalize the key (lower + strip, and ideally
  collapse internal whitespace) on both sponsor and drug_name; pick a canonical display casing for
  the merged entry.

## 5. Literature agent — adaptive search

The literature tools run a fixed call sequence with no retry logic. Possible additions:

- **Low hit count broadening**: if `fetch_and_cache` returns fewer than ~20 PMIDs, call
  `expand_search_terms` again with a broader disease term (e.g. `"non-alcoholic steatohepatitis"`
  → `"liver disease"`) before proceeding to `semantic_search`.
- **Low similarity retry**: if `semantic_search` returns all similarity scores below ~0.6,
  try `fetch_and_cache` with different queries before calling `synthesize`.

## 6. Smaller open items

- Integration test for `run_rag` in `tests/integration/services/test_retrieval.py`.
- Implement `DrugBankClient.get_drug()` and `get_interactions()` in `data_sources/drugbank.py`.
- Show "Fetching from cache" (not "pulling live evidence") in the loading UI on a seed-report hit.
  Needs a `source: "seed" | "live"` field on `AnalysisStatusResponse`, set early in `_execute`
  (before the spinner sleep), and a frontend copy swap in `LoadingState`.
- Add connection pooling singleton to `db/session.py` (currently creates new engine per call).
- Fix `runners/pubmed_runner.py` to use `logging` instead of `print()`.
- Remove superseded tests in `tests/integration/services/test_pubmed_query.py` (marked `# TODO delete`).

## 7. Duplicated code in `src/indication_scout/` (2026-08-04, `/finddupes` scan)

| Location A | Location B | Overlap | Suggested fix |
|---|---|---|---|
| [clinical_trials.py:777-786](../src/indication_scout/data_sources/clinical_trials.py#L777-L786) (`_normalize_phase` mapping) + [:821-831](../src/indication_scout/data_sources/clinical_trials.py#L821-L831) (`_phase_rank` dict) | [_trial_formatting.py:54-64](../src/indication_scout/agents/_trial_formatting.py#L54-L64) (`_PHASE_RANK`) | Same phase-name → rank ladder hand-copied in 3 places; `_trial_formatting.py`'s comment admits it mirrors `_phase_rank`. | Move to one constant in `constants.py`, import from all 3 sites. |
| [pubmed.py:368-376](../src/indication_scout/data_sources/pubmed.py#L368-L376) (abstract-parts loop, article branch) | [pubmed.py:445-453](../src/indication_scout/data_sources/pubmed.py#L445-L453) (book branch) | Identical AbstractText join/label logic within `_parse_pubmed_xml`. | Extract `_extract_abstract(elem) -> str \| None`. |
| [pubmed.py:390-401](../src/indication_scout/data_sources/pubmed.py#L390-L401) (pub-date build, article) | [pubmed.py:469-481](../src/indication_scout/data_sources/pubmed.py#L469-L481) (book) | Identical year/month/day string-concat logic. | Extract `_extract_pub_date(elem) -> str \| None`. |
| [pubmed.py:415-418](../src/indication_scout/data_sources/pubmed.py#L415-L418) (pubtypes cache warm, article) | [pubmed.py:488-489](../src/indication_scout/data_sources/pubmed.py#L488-L489) (book) | Identical `PublicationType` extraction + `cache_set` call. | Fold into the same shared per-article parsing helper as above. |
| [format_report.py:412](../src/indication_scout/report/format_report.py#L412) `rank_line` regex + [:436](../src/indication_scout/report/format_report.py#L436) longest-match loop | [supervisor_tools.py:1667](../src/indication_scout/agents/supervisor/supervisor_tools.py#L1667), [:1703-1705](../src/indication_scout/agents/supervisor/supervisor_tools.py#L1703-L1705), [:1726](../src/indication_scout/agents/supervisor/supervisor_tools.py#L1726) | Same "N. disease — tail" rank-line regex and "longest containing key wins" disease-match strategy, independently re-typed 3 times in `supervisor_tools.py` (comments there note the analogy to the report formatter's rule) rather than imported. `supervisor_tools.py`'s `rank_line` has an extra optional `tail` group not present in `format_report.py`'s — not byte-identical, drift already present. | Extract a shared `longest_key_match(text_lower, candidates) -> str \| None` helper; consider one shared `RANK_LINE_RE` if the tail-capture difference is intentional, document why. |
| [chembl.py:42-108](../src/indication_scout/data_sources/chembl.py#L42-L108) (`_load_chembl_names`/`_save_chembl_names`/`_lookup_chembl_id_by_name`) | [open_targets.py:70-146](../src/indication_scout/data_sources/open_targets.py#L70-L146) (`_load_target_evidences`/`_save_target_evidences`) | Both hand-roll "one JSON file per parent ID with `cached_at`/`ttl` expiry, `mkdir(parents=True, exist_ok=True)`" — the same TTL idiom `utils/cache.py`'s `cache_get`/`cache_set` already centralizes, reimplemented instead of reused. Not identical (chembl stores a flat list; open_targets stores a per-sub-key `{efo_id: entry}` map with per-entry TTL), so a shared helper needs to support both shapes. | Extract a shared "per-key multi-entry JSON cache with per-entry TTL" helper in `utils/cache.py` that both can call. |

Minor/lower-priority: `tests/unit/data_sources/test_base_client.py:11-24` duplicates
`tests/integration/data_sources/test_base_client.py:11-24`'s `ConcreteTestClient`/`_make_client`
verbatim — worth sharing via a test helper module. Everything else the scan surfaced (per-client
integration fixtures, inline `Client(cache_dir=tmp_path)` in unit tests) is idiomatic pytest
repetition, not true duplication.
