# Future Improvements

## Clinical Trials Query Quality

### 1. Expand drug synonyms before querying
ChEMBL/DrugBank already have synonym lists. Pass the top N synonyms as an OR query
(e.g. `metformin OR glucophage OR dimethylbiguanide`) to `query.intr`.
This addresses false whitespace signals caused by trials registered under brand names
or salt forms rather than the INN.

### 2. Use MeSH terms for indications
ClinicalTrials.gov indexes conditions against MeSH. Resolving the indication to its
MeSH preferred term before querying gives much better recall. The NLM has a free
MeSH lookup API.

## Literature Agent — Adaptive Search

The MVP uses a fixed call sequence with no retry logic. Once time allows, add:

- **Low hit count broadening**: if `fetch_and_cache` returns fewer than ~20 PMIDs, call
  `expand_search_terms` again with a broader disease term (e.g. `"non-alcoholic steatohepatitis"`
  → `"liver disease"`) before proceeding to `semantic_search`.
- **Low similarity retry**: if `semantic_search` returns all similarity scores below ~0.6,
  try `fetch_and_cache` with different queries before calling `synthesize`.

---

## Literature — per-PMID direction via a structured sub-call (replace the regex guard)

`synthesize` now emits a per-PMID directional verdict (contaminated | supporting | contradicting |
mixed), and `services/retrieval.py` backstops it with phrase-matching guards
(`_key_finding_is_positive` / `_key_finding_is_negative`) that upgrade/downgrade an obviously-wrong
LLM label by scanning the key_finding text. This is inherently leaky — direction is a semantic
judgment, not a regex. We have already had to patch it repeatedly (BRAVE-I positive-but-grouped-
with-failed-sibling; BRAVE-II "failing to" negation gap; imatinib × GBM where `[^.]*` spanned the
negation in "primary endpoint ... was not met" so a flat-negative read as positive and flipped the
indication to #1).

The robust fix: a small, isolated structured sub-call per RELEVANT abstract — "for THIS abstract,
did the drug help / not help / mixed for THIS disease? (supporting | contradicting | mixed)" —
returning a single enum. Build supporting/contradicting/relevant deterministically from those, as
today. This replaces both the in-prompt verdict-direction AND the regex guards with a narrow,
well-scoped LLM question (far more reliable than the holistic verdict or the phrase matcher).
Trade-off: ~N extra small LLM calls per pair (one per relevant abstract, typically ≤5) and a little
LLM nondeterminism, but on a much narrower question. Keep a thin deterministic floor only for the
egregious cases. This is the "LLM judges, code backstops" pattern applied at the right granularity.

## WhitespaceResult Schema Gap

`is_whitespace` is binary, but that misses a third state: "early stage, unproven."

Example: metformin + glioblastoma returns `is_whitespace=False` with `exact_match_count=9`,
but those 9 trials are all Phase 1/2 with small enrollment. The agent correctly identifies
"not whitespace, but not competitive either" — but that nuance lives only in the free-text
summary, not in the structured data.

The phase/maturity dimension is not captured in `WhitespaceResult`. Consider adding:
- A `max_phase` field (highest phase among exact-match trials)
- A `maturity` enum: `whitespace` / `early_stage` / `established`
- Or aggregate enrollment of exact-match trials as a proxy for how well-covered the space is

---

# TODO

Active tasks organized by component. Items here are actionable and derived from stubs, known gaps, and code issues in the codebase.

## RAG Pipeline

- [x] Implement `synthesize()` in `services/retrieval.py`
- [x] Define `EvidenceSummary` Pydantic model in `models/model_evidence_summary.py`
- [x] Write `prompts/synthesize.txt` prompt template
- [x] Implement `run_rag()` in `runners/rag_runner.py`
- [x] Unit tests for `run_rag` in `tests/unit/runners/test_rag_runner.py`
- [x] Integration tests for `synthesize` in `tests/integration/services/test_retrieval.py`
- [ ] Integration test for `run_rag` in `tests/integration/services/test_retrieval.py`
- [ ] Preserve class-level evidence DIRECTION. `judge_literature_strength` flattens
      direction to "none" whenever `evidence_basis != "drug_specific"` (enforced in
      `_parse_strength`), so a meaningful class-level negative — e.g. the whole GLP-1 class
      failed in Parkinson's (NLY01 negative) — is discarded. Safe (omission over inaccuracy)
      but informative for a repurposing tool. Fix additively: a SEPARATE `class_direction`
      field on `LiteratureStrength` / `EvidenceSummary`, populated only for `class_level`,
      rendered as distinct prose ("class-level signal: GLP-1 class failed in PD") and kept OUT
      of `es.strength` / `es.direction` so the supervisor ranking path stays clean.

## Data Sources

- [ ] Implement `DrugBankClient.get_drug()` and `get_interactions()` in `data_sources/drugbank.py`
- [ ] Dedup competitor entries case-insensitively. `data_sources/clinical_trials.py:720` keys
      competitors on `f"{t.sponsor}|{drug_name}"` using RAW strings, so casing/whitespace
      variants of the same drug from one sponsor produce duplicate rows — e.g. semaglutide's
      competitive landscape shows "Efimosfermin alfa"/"Efimosfermin Alfa" (both GSK) and
      "denifanstat"/"Denifanstat" (both Sagimet) as separate entries, inflating the count.
      Fix: normalize the key (lower + strip, and ideally collapse internal whitespace) on both
      sponsor and drug_name; pick a canonical display casing for the merged entry.

## API

- [x] Define API routes in `api/routes/`
- [x] Define request/response schemas in `api/schemas/`
- [ ] Show "Fetching from cache" (not "pulling live evidence") in the loading UI on a seed-report
      hit. Needs a `source: "seed" | "live"` field on `AnalysisStatusResponse`, set early in
      `_execute` (before the spinner sleep), and a frontend copy swap in `LoadingState`.

## CLI

- [x] Implement CLI module (`indication_scout.cli.cli`) referenced in `pyproject.toml`

## Infrastructure

- [ ] Add connection pooling singleton to `db/session.py` (currently creates new engine per call)

## Code Quality

- [ ] Fix `runners/pubmed_runner.py` to use `logging` instead of `print()`
- [ ] Remove superseded tests in `tests/integration/services/test_pubmed_query.py` (two tests marked `# TODO delete`)
- [ ] Review `get_drug_competitors()` in Open Targets client (marked `# TODO needs rework`)
