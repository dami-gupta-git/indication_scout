# Literature-Only Candidate Sourcing — Design Outline

Status: draft outline, not yet fleshed out. Open questions marked inline.

## Motivation

The current pipeline sources candidate diseases from Open Targets (genetic/mechanistic
association) and competitor drugs (shared indications). Both require the disease to already
be linked to the drug's target or class in structured data. This misses indications that
exist only as literature signal: case reports, small investigator-led trials, off-label use,
or repurposing hypotheses raised in review articles, with no upstream genetic/mechanistic
trace.

This agent runs alongside the existing mechanism-based candidate finder. It takes a drug
only, no disease hypothesis, and proposes candidate indications sourced from the literature
itself.

## Difference from the existing literature retrieval

The current `semantic_search` (`services/retrieval.py`) is disease-anchored: given a
drug-disease pair, it builds a query ("evidence for X as treatment for Y"), fetches a pool
of abstracts already retrieved by disease-anchored PubMed queries, and reranks that pool
by similarity plus a pubtype boost. It presupposes the candidate disease.

This new agent has no disease to anchor on. The task is closer to open-ended discovery:
given a drug, find what diseases the literature discusses in connection with it. That's an
extraction problem over a drug-only literature pool, not a rerank of a disease-scoped pool.

## Stage 1 — PubMed query (drug-anchored, no disease term)

A bare drug-name search over a decades-old generic drug returns a very large, low-precision
pool dominated by the drug's primary-indication literature. The query needs to bias toward
content that actually discusses a candidate indication rather than routine primary-indication
research.

Candidate approaches, to be evaluated against real queries before committing:
- Drug name/synonyms restricted by publication type (case reports, small trials, reviews)
  rather than an unrestricted `[tiab]` search.
- OR-ing in repurposing-signal MeSH qualifiers (e.g. `"drug repositioning"[Mesh]`,
  `"off-label use"[Mesh]`) alongside the plain drug search.
- Open question: recall vs. precision tradeoff here is the central risk of the whole
  feature. Needs validation against a handful of drugs with known literature-only
  indications before deciding the query shape.

## Stage 2 — Disease/indication extraction

The disease-sourcing step itself. Not a vector-similarity problem — an extraction problem
over the drug-only pool.

Two candidate approaches, not mutually exclusive:
- MeSH-term pass: PubMed records already carry indexed MeSH descriptors
  (`mesh_terms`, already fetched and stored per abstract in `pubmed_abstracts`). Filter
  disease-type MeSH terms against the drug's already-known indications (Open Targets +
  competitor list) to surface only novel ones. Cheap, no LLM call, but limited to what
  MeSH indexing captured.
- LLM read per abstract (or batched): extract explicit disease/indication mentions from
  title+abstract text, similar in shape to the existing `_judge_pmid_directions` pattern.
  Higher recall, higher cost, needs its own accuracy validation.

Where the existing embedding/rerank machinery re-enters: once a candidate disease name is
extracted here, it can be handed to the *existing* disease-anchored `semantic_search` to
validate and score it against the literature, same as any other candidate today. The vector
step's role in this new agent is downstream validation, not upstream discovery.

## Stage 3 — Storage

- Reusing the existing `pubmed_abstracts` table/collection is likely correct if this stage
  fetches and embeds abstracts the same shape as today (title + abstract text) — no
  structural reason for a second vector collection just because the query that populated it
  differs.
- A new collection would only be justified if this stage needs something structurally
  different, e.g. full-text chunks instead of abstracts. Not yet decided.

## Chunking

Does not apply to abstract-only retrieval — abstracts are already atomic (~200-400 words),
below any reasonable chunk size. Chunking only becomes a real question if Stage 1 expands
to full text (PMC open-access subset), which would need its own decision: which sections to
use, how to split them, and how to attribute a chunk-level disease mention back to a citable
PMID. Out of scope until full-text retrieval is decided on.

## Open questions to resolve before implementation

1. Stage 1 query shape: pubtype restriction vs. MeSH repurposing qualifiers vs. some
   combination, validated against known literature-only indications for a handful of drugs.
2. Stage 2 extraction method: MeSH-term pass, LLM read, or both in sequence (cheap MeSH
   pass first, LLM read on the remainder).
3. Storage: same `pubmed_abstracts` table, or a distinct collection if the fetched content
   type diverges from today's abstract-only shape.
4. How this agent's output enters the supervisor's candidate ranking alongside the
   mechanism-based candidates (same `INVESTIGATION_CAP` pool, or a separate track).
