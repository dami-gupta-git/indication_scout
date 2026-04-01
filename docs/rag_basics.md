# RAG Pipeline Overview

The RAG (Retrieval-Augmented Generation) pipeline solves a specific problem: raw PubMed searches return hundreds of papers for a drug-disease pair, but many are irrelevant. For example, searching "bupropion AND obesity" returns depression papers that incidentally mention obesity, burying papers about bupropion as an actual obesity treatment.

The pipeline sits in the evidence-stacking layer. After Open Targets identifies candidate disease indications for a drug, the RAG pipeline retrieves, reranks, and synthesizes PubMed literature so the most relevant papers surface and produce structured evidence summaries.

For detailed implementation specs, schema definitions, and design rationale, see [rag_details.md](rag_details.md).

## Architecture

The pipeline is implemented as the `RetrievalService` class in `services/retrieval.py`, orchestrated by `runners/rag_runner.py`. It processes a drug across its top 15 disease indications. `RetrievalService.get_drug_competitors()` fetches raw competitor data from `OpenTargetsClient` (which returns `CompetitorRawData`), uses an LLM call (`merge_duplicate_diseases`) to deduplicate disease names, removes overly broad terms, sorts by competitor count, and slices to the top 15.

```
Drug name
  |
  v
RetrievalService.get_drug_competitors  -->  raw data from OpenTargetsClient
  |                                          + LLM merge/dedup
  |                                          + sort + top-15 slice
  v
RetrievalService.build_drug_profile    -->  DrugProfile (name, synonyms, targets, mechanisms, ATC)
  |
  v
For each disease:
  Stage 0: expand_search_terms  -->  5-10 diverse PubMed queries (LLM-generated)
  Stage 1: fetch_and_cache      -->  PubMed search + embed + store in pgvector
  Stage 2: semantic_search      -->  cosine similarity over pgvector, top-k abstracts
  Stage 3: synthesize           -->  LLM reads top abstracts, produces EvidenceSummary
```

**Output:** `dict[str, EvidenceSummary]` mapping each disease to a structured evidence summary with strength rating, study types, key findings, and supporting PMIDs.

## Pipeline Stages

### Stage 0 -- Query Expansion (`expand_search_terms`)

A small LLM call (`claude-haiku-4-5-20251001`) generates diverse PubMed keyword queries from a `DrugProfile` across 5 axes: drug name, drug class + organ, mechanism + organ, target gene, and synonym. The organ term is pre-extracted via a separate Haiku call (`extract_organ_term`). Both results are cached. This replaces naive single-query PubMed searches with 5-10 complementary queries that cover different angles of the drug-disease relationship.

### Stage 1 -- Fetch and Cache (`fetch_and_cache`)

For each query from Stage 0, searches PubMed for up to 200 PMIDs (`PUBMED_MAX_RESULTS` constant), filters out those already in the database, fetches new abstracts, and discards abstract-less articles (letters, editorials). New abstracts are embedded with BioLORD-2023 and bulk-inserted into pgvector. Returns the deduplicated union of all PMIDs across all queries.

### Stage 2 -- Semantic Search (`semantic_search`)

Constructs a natural-language therapeutic query ("Evidence for {drug} as a treatment for {disease}..."), embeds it with BioLORD-2023, and runs cosine similarity search over pgvector scoped to the PMIDs from Stage 1. Returns the top-k most relevant abstracts (default 5).

### Stage 3 -- Synthesize (`synthesize`)

The top abstracts from Stage 2 are passed to Claude (`claude-sonnet-4-6`) with a structured prompt. The LLM reads the actual retrieved papers and produces an `EvidenceSummary`:

- `summary` -- prose summary of evidence
- `study_count` -- number of studies reviewed
- `study_types` -- types of studies (e.g. RCT, cohort, preclinical)
- `strength` -- "strong", "moderate", "weak", or "none"
- `has_adverse_effects` -- whether adverse effects were reported
- `key_findings` -- list of key findings
- `supporting_pmids` -- PMIDs backing the claims

## Key Components

- **Embedding model:** BioLORD-2023 (`FremyCompany/BioLORD-2023`), loaded locally via `sentence-transformers`. 768-dimensional vectors. Trained on UMLS + SNOMED-CT biomedical ontologies.
- **Vector store:** PostgreSQL + pgvector. Abstracts and embeddings stored in `pubmed_abstracts` table. Cosine similarity for search.
- **Cache layers:** File-based cache (`_cache/` dir, SHA-256 keys, 5-day TTL) for LLM results, PubMed searches, and Open Targets data. pgvector itself acts as the abstract/embedding cache.
- **Service class:** `RetrievalService(cache_dir: Path)` in `services/retrieval.py` is the single entry point for all pipeline operations.
- **Runner:** `run_rag(drug_name, db, cache_dir)` in `runners/rag_runner.py` orchestrates the full pipeline with per-step timing logs and a final ranking by evidence strength.
