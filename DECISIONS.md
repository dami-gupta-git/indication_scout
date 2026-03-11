# Decisions

## pgvector over dedicated vector database
- **Date**: 2026-02
- **Status**: Accepted
- **Context**: Needed a vector store for PubMed abstract embeddings to support semantic search in the RAG pipeline.
- **Decision**: Use PostgreSQL with the pgvector extension rather than a dedicated vector DB (Pinecone, Weaviate, etc.).
- **Rationale**: Single infrastructure dependency (Postgres already needed). Dataset is small enough (~10k-50k abstracts total) that pgvector performance is sufficient. Simpler Docker setup with one container.

## BioLORD-2023 for biomedical embeddings
- **Date**: 2026-02
- **Status**: Accepted
- **Context**: Needed an embedding model optimized for biomedical text to support semantic search over PubMed abstracts.
- **Decision**: Use `FremyCompany/BioLORD-2023` (768-dim) via `sentence-transformers`.
- **Rationale**: Trained on UMLS ontology, SNOMED-CT, and biomedical definitions. State-of-the-art on MedSTS and EHR-Rel-B benchmarks. SentenceTransformer-compatible for easy integration.

## Pydantic defensive defaults with coerce_nones validator
- **Date**: 2026-03-01
- **Status**: Accepted
- **Context**: External APIs frequently return `null` for fields that have safe non-None defaults (empty strings, empty lists), causing downstream `or []` guards everywhere.
- **Decision**: Apply a `coerce_nones` model validator to every Pydantic model that ingests external data.
- **Rationale**: Centralizes null handling at the model boundary. Eliminates scattered defensive guards in factory methods and downstream code. Fields with `None` as their default (genuinely optional) are left alone.

## DrugProfile as flat LLM-facing projection
- **Date**: 2026-02
- **Status**: Accepted
- **Context**: LLM-based search term expansion needs drug metadata but `RichDrugData` is deeply nested and contains data irrelevant to query generation.
- **Decision**: Created `DrugProfile` as a flat Pydantic model with only the fields needed for LLM prompts (name, synonyms, target gene symbols, mechanisms, ATC codes/descriptions, drug type).
- **Rationale**: Flat structure is easier for the LLM to consume. ATC descriptions limited to level 3 and level 4 only -- levels 1-2 are too broad for useful PubMed queries.

## LLM-based disease name normalization over ontology traversal
- **Date**: 2026-02
- **Status**: Accepted
- **Context**: Open Targets disease names (e.g. "narcolepsy-cataplexy syndrome") often do not match PubMed indexing terms, causing zero-result searches.
- **Decision**: Use cheap Haiku LLM calls to normalize disease names, with a broadening step if PubMed hit count is below threshold.
- **Rationale**: More flexible than building a synonym dictionary or ontology traversal system. Broadening blocklist prevents over-generic terms like "cancer" or "disease".

## File-based caching with SHA-256 keys
- **Date**: 2026-02
- **Status**: Accepted
- **Context**: Multiple components (data source clients, services) need caching to avoid redundant API calls and LLM invocations.
- **Decision**: Shared file-based cache in `_cache/` directory with SHA-256-keyed JSON files and 5-day TTL. Centralized in `utils/cache.py`.
- **Rationale**: Simple, filesystem-based, no additional infrastructure. Namespaced to avoid collisions. TTL prevents stale data. Shared utility eliminates per-module cache duplication.

## EvidenceSummary strength as Literal type
- **Date**: 2026-03
- **Status**: Accepted
- **Context**: The `synthesize` LLM prompt constrains evidence strength to one of four values ("strong", "moderate", "weak", "none"). Needed the Pydantic model to enforce this constraint.
- **Decision**: Use `Literal["strong", "moderate", "weak", "none"]` for `EvidenceSummary.strength` with default `"none"`.
- **Rationale**: Compile-time and runtime validation that the LLM output conforms to the expected vocabulary. Prevents silent acceptance of unexpected strength labels.

## run_rag iterates over diseases (not single drug-disease pair)
- **Date**: 2026-03
- **Status**: Accepted
- **Context**: The RAG pipeline needed an entry point that runs the full pipeline. Initial plan had `run_rag(drug_name, disease_name, db)` for a single pair.
- **Decision**: `run_rag(drug_name, db)` fetches top 10 disease indications via `get_drug_competitors` and iterates over all of them, returning `dict[str, EvidenceSummary]`.
- **Rationale**: The primary use case is exploring all promising indications for a drug, not a single pair. Callers can still run a single disease by calling the underlying pipeline functions directly.

## Separate test database (scout_test)
- **Date**: 2026-02
- **Status**: Accepted
- **Context**: Integration tests that touch pgvector need a database, but test data should not contaminate the main database.
- **Decision**: Docker init script creates a `scout_test` database alongside `scout`. Integration tests use `TEST_DATABASE_URL` from `.env`.
- **Rationale**: Isolation between dev and test data. Same Postgres instance keeps infrastructure simple. Migrations must be run separately against the test URL.
