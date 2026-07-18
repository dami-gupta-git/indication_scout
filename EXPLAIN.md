# How trial phase & status feed into a repurposing verdict

When you run IndicationScout on a drug (say "metformin"), one of its jobs is to check ClinicalTrials.gov
for any trials testing that drug on a new disease (not the one it's already approved for) — this is how
it looks for repurposing opportunities.

Each trial found comes back with two key attributes:

- **Phase** — how far along the trial got (Phase 1, 2, 3, 4)
- **Status** — what happened to it (still recruiting, completed, terminated early, withdrawn before
  starting)

The naive approach would be: "more/bigger trials completed = stronger repurposing candidate." But that's
too crude, because a trial's status changes what a given phase actually means:

- A **completed Phase 3** trial is strong positive evidence — the drug made it through rigorous testing
  for this new disease.
- A **terminated Phase 3** trial is ambiguous on its face — did it fail, or did it just run out of
  funding/patients? The app looks at why it was terminated (there's a free-text reason on CT.gov). If it
  was stopped for safety or lack of efficacy, that's actually a clean, useful signal ("this was tried and
  didn't work") — different from being stopped for boring business reasons (funding, enrollment
  problems), which tells you nothing about whether the drug works.
- A **withdrawn** trial means it was cancelled before a single patient was ever dosed — so it shouldn't
  count as evidence of anything, positive or negative.

So the app doesn't just report raw counts of trials — it converts phase+status+stop-reason into a single
"how far did this repurposing idea actually get tested" verdict (things like "tested and failed at Phase
3" vs "actively being tested" vs "never seriously tried"). That verdict is what gets handed to the AI
agent that writes the final report, so the report can say something like "Phase 3 trial for this
indication was terminated for lack of efficacy" instead of just "1 terminated trial" — which would be
misleading on its own.

# How PubMed queries get built for a drug

When the literature agent investigates a drug-disease pair, it doesn't just search PubMed for
"metformin AND colorectal cancer." That's too narrow — it would miss papers that talk about the drug's
mechanism (AMPK), its drug class (biguanide), or its gene targets (PRKAB1) without ever using the drug's
brand name.

The real query-building pipeline (`RetrievalService.expand_search_terms` in
`services/retrieval.py`, invoked via the `expand_search_terms` tool):

1. **Build a drug profile** — resolve the drug to its ChEMBL ID and pull its synonyms, gene targets,
   mechanisms of action, and ATC classification codes.
2. **Extract an organ/tissue term** for the disease via a small LLM call (e.g. "colorectal cancer" →
   "colon").
3. **Resolve the disease to its canonical MeSH preferred term** — PubMed's own controlled vocabulary —
   falling back to the raw disease name if that lookup misses.
4. **Ask an LLM to generate a diverse set of PubMed keyword queries** from all of the above: drug name
   variants, mechanism-based queries, target-gene queries, drug-class queries, etc. The disease term is
   inserted as a placeholder and substituted in afterward, quoted (e.g. `"colon neoplasms"`), because
   PubMed's automatic term-mapping breaks on bare multi-word phrases.
5. **Deduplicate** the queries (case-insensitive) and cache them per drug/disease pair.

Example queries generated for metformin × colorectal cancer might include `metformin "colon
neoplasms"`, `AMPK "colon neoplasms"`, and `biguanide "colon neoplasms"` — not just the literal drug
name.

Verified live run, metformin × chronic kidney disease (2026-07-17), 10 queries across all 5 axes:
`metformin AND "Renal Insufficiency, Chronic"`, `biguanides AND kidney`, `blood glucose lowering
drugs AND kidney`, `mitochondrial complex I inhibition AND kidney`, `NADH dehydrogenase AND "Renal
Insufficiency, Chronic"`, `NDUFA1 AND "Renal Insufficiency, Chronic"`, `NDUFV1 AND kidney`,
`Glucophage AND "Renal Insufficiency, Chronic"`, `metformin hydrochloride AND "Renal Insufficiency,
Chronic"`, `glycerol-3-phosphate dehydrogenase AND kidney`.

Those queries are then run concurrently against PubMed's search API (`fetch_and_cache`), all resulting
PMIDs are deduplicated and date-filtered (for holdout/backtesting), and any new abstracts are fetched,
embedded, and cached. A final `semantic_search` step re-ranks the cached abstracts by embedding
similarity to the drug-disease pair, since a keyword match doesn't guarantee relevance.

There is a separate, simpler helper (`services/pubmed_query.py::get_pubmed_query`, drug AND
normalized-disease-OR-terms) that is not wired into this live pipeline — it's only exercised by tests
today. A `normalize_for_pubmed` disease-normalization helper is also used independently by the
safety-signal probe (`ml_models/safety_signal_probe.py`) for its own queries.

# Contamination

Contamination is the app's term for retrieval noise that has to be filtered out before it's counted
as evidence — a trial, FDA-label mapping, or PubMed abstract that superficially matches a search but
is actually about the wrong drug, a different disease, an already-approved indication, or the drug
being used for a comorbidity rather than the disease being investigated. It's judged independently at
three levels (per-trial, per-candidate-disease vs. FDA approvals, per-abstract), and excluded items
are dropped from downstream signals/counts with a "N hidden/excluded" note in the report. See
`docs/contamination.md` for details.

Each of the three levels is its own LLM judgment call, not a shared classifier:

- **Per-trial** (clinical trials agent): for each candidate trial, the LLM checks three things — is
  this drug actually the one being tested (not just a comparator arm)? Is the disease actually the
  target indication (not an already-approved indication or a narrower subtype of one)? Is the drug's
  therapeutic intent actually the target disease (not a comorbidity the patients happen to have)? A
  trial failing any of these is marked contaminated and excluded from the deterministic phase/status
  signal derivation and from the report's example-trials table — with a "N hidden as a different
  indication" note shown instead of just dropping it silently.
- **Per-candidate-disease** (FDA approval mapping): even before trials or literature are pulled, a
  candidate disease itself can be labeled contaminated if searching for it would necessarily pull in
  evidence for an already-approved indication — e.g. a broad umbrella term where the approved use is a
  narrower subtype, or a sibling disease that happens to share a search term with what's already
  approved. This is checked against a small curated list first, then falls back to an LLM
  classification.
- **Per-abstract** (literature synthesis): each PubMed abstract pulled in is classified as supporting,
  contradicting, mixed, neutral, or contaminated for the drug-disease pair — same three failure modes
  as trials (wrong drug, wrong disease/already-approved, comorbidity-not-target-intent). Any abstract
  the LLM doesn't explicitly classify defaults to contaminated rather than being silently counted as
  evidence — a deliberate conservative bias, since unverified relevance shouldn't inflate the evidence
  count.

All three feed the same principle: it's better to under-count and disclose what was hidden than to let
unverified matches inflate the strength of a repurposing signal.

# Approval awareness

A repurposing report must never present "the drug already works for an approved sub-form of this
disease" as if it were new evidence. E.g. the candidate disease might be broad ("NAFLD"), but the drug
could already be FDA-approved for a narrower slice of it ("MASH") — trials/papers about that slice
aren't repurposing evidence, they're just restating the existing approval.

The fix: classify every candidate disease against the drug's FDA label once, upstream
(`services/approval_check.py`), into one of four labels — `approved` (drop the candidate entirely),
`combination_only` (demote, only approved as part of a combination product), `contaminated` (a real
candidate, but its trial/paper counts are polluted by the approved sibling/subtype — kept but flagged
unreliable), `none` (genuinely distinct, kept and ranked normally). That label then flows through every
downstream stage — trial relevance, literature relevance, dev-stage tiering, ranking — so the report
can't contradict itself (e.g. summary text and evidence tables disagreeing about whether something is
already approved). Full design doc: `docs/APPROVAL_AWARENESS.md`.

# pgvector

**What query fetches PubMed articles into pgvector:** the keyword queries described above (e.g.
`metformin "colon neoplasms"`, `AMPK "colon neoplasms"`) are what's run against PubMed's search API in
`fetch_and_cache`. Any PMIDs not already cached get their abstracts fetched and embedded, then inserted
into the `pubmed_abstracts` table. That table has no query column — PMIDs from multiple queries are
deduplicated/flattened before insert, so which query originally surfaced a given article isn't stored.

**When queries are embedded, and how they look:** not the keyword queries above — those are only used
for the PubMed API search, never embedded. A separate natural-language query is embedded at *search*
time, in `semantic_search` (`services/retrieval.py`, ~line 673-679), built from the drug and disease
right before the similarity lookup:

```
Evidence for {drug} as a treatment for {disease}, including clinical trials, efficacy data,
mechanism of action, and preclinical studies
```

e.g. `"Evidence for metformin as a treatment for colorectal cancer, including clinical trials,
efficacy data, mechanism of action, and preclinical studies"`. This sentence is embedded with
BioLORD-2023 and compared via cosine similarity against the cached abstract embeddings (restricted to
the PMIDs just fetched) to re-rank them by relevance. It's generated fresh on every call, not cached.
