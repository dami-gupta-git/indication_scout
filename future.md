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
