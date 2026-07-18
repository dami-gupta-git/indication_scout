# Roadmap

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

