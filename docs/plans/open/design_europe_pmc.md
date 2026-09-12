# Europe PMC Literature Sourcing — Design

Status: retrieval, extraction and selection are built; invocation is not yet decided and nothing
calls them. Progress: `PLAN_europe_pmc.md`. Measurements behind these choices:
`for_me/findings_europe_pmc.md`.

## Purpose

Given a drug name, produce candidate indications drawn from published literature, each
attributable to specific papers. Runs alongside the mechanism-based candidate finder, which
requires a disease to be linked to the drug's target or class in structured data. This component
has no such requirement, so it reaches indications that exist only as literature signal: case
reports, investigator-led studies, off-label use, hypotheses raised in reviews.

## Retrieve

One query per drug against the Europe PMC search API. The drug name is matched in title or
abstract; records without an abstract are excluded. Under a temporal holdout, a publication-year
bound is added to the query.

The full result set is paginated through, following the cursor Europe PMC returns until exhausted.
Europe PMC repeats the cursor on the final page rather than omitting it, so pagination stops on an
unchanged cursor; a cursor seen earlier in the same fetch raises, which catches a server-side
cycle that would otherwise re-request pages forever.
There is no result cap and no relevance filtering, ranking, or scoring. The five drugs behind the
recall measurements gave pools of 287 to 3,298 papers, but a later bounded sildenafil query
returned 6,739 records, so that range is not a ceiling. Pool sizes across the validation runbook
are surveyed before this is wired in, because every downstream cost — one LLM call per abstract,
one merge call over every extracted name, and the size of the retrieval cache — scales with them.

Retrieval is all-or-nothing. A failure at any point, including mid-pagination, raises rather than
returning what was collected, so a large pool is a single long fetch that yields nothing if its
last page fails; the 6,739-record pool took 20.9 seconds cold across eight pages.

Preprints and non-MEDLINE records are retained. A paper is identified by the source and record
identifier Europe PMC assigns, not by PMID, which preprints lack.

Dates come from the first-publication date throughout — both the holdout bound in the query and
the publication year stored on the paper. The alternative field, publication year, is the year the
paper reached a journal issue, which disagrees with first publication on 16.1% of a measured
6,739-record pool and is absent from a small number entirely. Filtering on one and storing the other would let a
paper pass a cutoff and then report a year beyond it. First publication is also the correct
holdout semantics: a paper available online in November 2021 and issued in 2022 was readable
before a 2022 cutoff, so it belongs in that run.

The cutoff is exclusive, matching the PubMed client, but Europe PMC's date range is inclusive at
both ends, so the query sends the day before the cutoff. Passing the cutoff through directly
returned papers published on the cutoff date itself, which is a holdout cut at an approval date
seeing papers published that day.

Each paper carries forward its identifiers, title, abstract, journal, first-publication date,
type, and citation count.

## Extract

Each paper's title and abstract goes to a small model, which returns the medical conditions the
paper states the drug was used, tested, or proposed to treat — one per line, copied verbatim from
the abstract, or NONE.

The prompt confines the model to the supplied text and forbids drawing on its own knowledge of the
drug. Without that constraint a holdout run recovers the post-cutoff indication from the model
rather than from the literature, which invalidates the holdout. NONE is specified for papers where
the drug appears as a comparator, assay reagent, contaminant, or background context, and for
papers about the drug's adverse effects, pharmacokinetics, or chemistry rather than about treating
a condition.

Extraction preserves the literature's own wording, so one condition arrives under several names.
Normalization to a canonical disease is a separate step, handled by the existing disease
normalization service.

## Select

Every extracted name goes to the existing disease merge service in a single call, which both
collapses synonyms onto a canonical name and reports which of them are already-approved
indications. It is one call rather than several batches: splitting the set means no call can
compare all variants of a term at once, and re-merging the survivors of earlier rounds drifts
toward generic categories — three distinct animal pain assays collapsed into a bare "pain". The
call is sized by the pool, at 81 names for duloxetine and 257 for colchicine.

The merged conditions are grouped and counted. Each candidate indication carries the set of papers
it was extracted from and the raw wordings that collapsed into it. Conditions the drug is already
approved for are removed. An unparseable merge response raises rather than yielding an empty
removal list, which would present approved indications as novel candidates.

The approved-indication list is resolved as of the retrieval cutoff, not as of today. The grouping
step takes an already-filtered list; `approval_check.list_approved_indications_at` performs that
read, keeping only entries in the approvals table dated before the cutoff. Supplying it is the
caller's job, which is part of the undecided invocation below. Removing today's list from a
holdout run would strip the indication the run exists to discover: a colchicine run cut at 2008
would drop atherosclerotic cardiovascular disease, approved in 2022.

Removal happens after normalization, so a condition is not stripped under one phrasing while
surviving under another.

## Caching

Retrieval results and extraction results are cached separately through the shared cache helper,
keyed on the query (including any date bound) and on the paper identifier respectively. Extraction
is the expensive stage and its cache is keyed per paper, so a re-run over an expanded pool only
pays for papers it has not seen.

## Failure handling

A retrieval failure raises, as described under Retrieve. A partial pool silently treated as
complete would understate the literature and cannot be distinguished downstream from a drug with
little written about it. An unparseable record raises too, naming the source and record id.

An extraction failure on an individual paper is logged and that paper is skipped; the count of
skipped papers is reported alongside the results. Conditions are never inferred for a paper whose
extraction did not complete.

## Concurrency

Extraction calls run under a bounded concurrency limit set in config. Europe PMC publishes no rate
limit for the search API, and retrieval inherits the shared client's retry and backoff.

## Invocation

Open. The component can be a tool on the literature agent, a separate agent, or a service step
that runs before the agents and feeds the supervisor's candidate pool. The choice determines the
interface and is not yet made.

## Open questions

Whether pool sizes across the full drug runbook resemble the ones measured, or whether enough
drugs have corpora above that range to change the cost profile. The retrieval, extraction and
merge stages are all sized by the pool, so this is settled before invocation is.

The cache TTL for each stage. Retrieval goes stale as new papers publish, while extraction is
keyed on an immutable paper identifier, so the two do not obviously share the global value. TTL
also bounds how far the retrieval cache accumulates: one bounded sildenafil query writes a single
13 MB file, because the cached payload is the full article set including every abstract.

How candidate indications are ranked against each other, and what evidence threshold a candidate
must meet before reaching the supervisor.

Whether preprint and published versions of the same work are reconciled. Europe PMC links some
explicitly; those links cover a small fraction of any pool, and matching the remainder on title
similarity would assert two records are the same paper without the source saying so.

## Testing

Retrieval is covered by integration tests against the live API in
`tests/integration/data_sources/test_europe_pmc_search.py`: field parsing of a pinned article,
whole-pool retrieval with unique identities, the date bound at two cutoffs, pagination past one
page, and the empty result. These are pinned to minoxidil's pre-1988 pool, which is closed — no
new paper can publish into a window that ended in 1987 — so the expected counts stay exact.

Extraction is tested at the prompt level in `tests/unit/test_condition_extraction.py` (prompt
builder, and parsing of multi-condition, NONE, bulleted, mixed, order-preserving-dedup,
prose-line and long-but-real inputs). Leakage is tested in
`tests/integration/services/test_condition_extraction.py` by running holdout-era abstracts that
omit the eventual indication and asserting it is not emitted, with a positive control so a model
that always returned NONE could not pass. Negatives are pinned by article key and chosen against
each indication's full synonym set; a keyword-only pass had mislabeled a minoxidil paper titled
"male pattern baldness" as a negative.

Selection is unit tested on a stubbed merge in `tests/unit/test_condition_grouping.py`: synonym
grouping, approved-indication removal, a holdout case where a post-cutoff approval must survive,
an empty input making no LLM call, an unparseable merge propagating, and aliases absent from the
input being ignored. The model factory is unit tested in `tests/unit/test_model_europe_pmc.py`.

The leakage and recall evidence rests on five hand-picked drug-indication pairs and 60 abstracts,
and the negative abstracts were chosen by matching a single keyword, which mislabeled one paper
that used a synonym. Widening this to holdout pairs drawn from the validation runbook, with the
synonym-aware normalization used elsewhere in the pipeline, comes before the component is wired
in.
