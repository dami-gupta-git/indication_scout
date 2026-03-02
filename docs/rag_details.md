# RAG Pipeline — Implementation Details

See [rag.md](rag.md) for the conceptual overview and pipeline fit.

---

## 1. PostgreSQL + pgvector (Abstract Cache + Vector Store)

Serves two purposes:
- **Caching**: Avoid re-fetching the same PubMed abstracts across runs. Deduplicate by PMID.
- **Vector search**: Store BioLORD-2023 embeddings alongside abstracts for semantic retrieval.

**Schema:**

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE pubmed_abstracts (
    pmid          TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    abstract      TEXT,
    authors       TEXT[],
    journal       TEXT,
    pub_date      TEXT,
    mesh_terms    TEXT[],
    embedding     vector(768),    -- BioLORD-2023 output dimension
    fetched_at    TIMESTAMP DEFAULT NOW()
);

CREATE INDEX ON pubmed_abstracts USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Track which drug-disease queries have been run
CREATE TABLE search_queries (
    id            SERIAL PRIMARY KEY,
    drug_name     TEXT NOT NULL,
    disease_name  TEXT NOT NULL,
    pubmed_query  TEXT NOT NULL,
    pmids         TEXT[],
    searched_at   TIMESTAMP DEFAULT NOW(),
    UNIQUE(drug_name, disease_name)
);
```

**Why pgvector over a dedicated vector DB (Pinecone, Weaviate, etc.):**
- Single infrastructure dependency (Postgres already needed)
- Abstracts are small enough (~10k-50k total across all drugs) that pgvector performance is fine
- Simpler Docker setup: one `postgres:16` container with `pgvector` extension

---

## 2. Embedding Model

**BioLORD-2023** (`FremyCompany/BioLORD-2023`)

Used for both abstract embeddings (stored at fetch time) and query embeddings (at search time).
Output dimension: **768**.

Trained on UMLS ontology (2023AA), SNOMED-CT, and 400k GPT-3.5-generated biomedical definitions,
using a three-phase strategy: contrastive learning from knowledge graphs → supervised self-distillation
→ weight averaging. This gives it richer semantic representations than a plain language model.
Achieves state-of-the-art on MedSTS (clinical sentence similarity) and EHR-Rel-B (biomedical concept
representation) benchmarks.

**Loading:** via `sentence-transformers` (included in `pyproject.toml` runtime dependencies). Implemented in `services/embeddings.py` as a lazy-loaded module-level singleton.

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("FremyCompany/BioLORD-2023")
```

**Query formulation:**
Encode therapeutic intent, not just keywords:
- Keyword query: `"bupropion AND obesity"`
- Embedding query: `"Evidence for bupropion as a treatment for obesity, including clinical trials, efficacy data, and mechanism of action"`

---

## 3. Stage Specifications

**Stage 1 — `fetch_and_cache(queries: list[str]) -> list[str]`**

```
For each PubMed keyword query:
  1. Search PubMed E-utilities → up to 500 PMIDs
  2. Filter to PMIDs not already in pgvector
  3. Fetch abstracts for new PMIDs (batch)
  4. Embed each abstract with BioLORD-2023
  5. Store (pmid, title, abstract, authors, journal, pub_date, embedding) in pgvector
  6. Return full list of PMIDs (cached + newly added)
```

**Stage 2 — `semantic_search(disease: str, drug: str, top_k: int = 20) -> list[dict]`**

```
1. Build therapeutic query: "Evidence for {drug} as a treatment for {disease}, ..."
2. Embed query with BioLORD-2023
3. Run cosine similarity search over pgvector
4. Return top_k abstracts ranked by similarity score
```

**Stage 3 — Re-rank (top 20 → top 5)**

After `semantic_search` returns 20 candidates, a re-ranker reduces to 5. Cross-encoder or LLM-based
reranker (TBD). Top 5 are passed to the Literature Agent.

**Stage 4 — `synthesize(drug, disease, top_5_abstracts) -> EvidenceSummary`**

Top 5 abstracts are stuffed into a Claude prompt. Claude reads the actual retrieved papers — not training
data. Output is a structured `EvidenceSummary` with PMIDs attached to every claim.

```python
EvidenceSummary(
    summary: str,
    study_count: int,
    study_types: list[str],
    strength: str,
    key_findings: list[str],
    supporting_pmids: list[str],
)
```

**Stage 0 — `expand_search_terms(drug_name, disease_name, drug_profile: DrugProfile) -> list[str]`** (Pre-fetch)

Send a `DrugProfile` (flat LLM-facing projection of `RichDrugData` -- name, synonyms, target gene symbols, mechanisms of action, ATC codes, ATC descriptions, drug type) to an LLM to generate diverse PubMed keyword queries across 5 axes (drug name, drug class+organ, mechanism+organ, target gene, synonym) with per-axis caps (total 5-10). Organ term is pre-extracted via a separate Haiku call (`extract_organ_term`). Both functions cache results. This runs before `fetch_and_cache`.

---

## 4. Disease Name Normalizer (`services/disease_normalizer.py`)

Converts raw Open Targets disease names (e.g. `"narcolepsy-cataplexy syndrome"`) into PubMed-friendly search terms (e.g. `"narcolepsy"`) before they are passed to `get_pubmed_query`.

**Two-step LLM strategy:**

1. **Normalize** — Haiku prompt strips subtypes, staging, etiology, and genetic qualifiers while preserving organ specificity. If the disease has a well-known synonym, both are returned joined with `OR` (e.g. `"eczema OR dermatitis"`).
2. **Verify** — If a `drug_name` is provided, the normalized term is verified with a PubMed count (`drug AND disease`). If the count is below `MIN_RESULTS` (3), a second Haiku call generalises to a broader category. The broader term is used only if it also has `>= MIN_RESULTS` hits and does not collapse to an over-generic term in `BROADENING_BLOCKLIST` (e.g. `"cancer"`, `"disease"`, `"syndrome"`).

**File-based cache (`_cache/`, 5-day TTL):**

| Namespace | Cache key | Cached value |
|-----------|-----------|--------------|
| `disease_norm` | `raw_term` | LLM-normalized string (e.g. `"narcolepsy"`) |
| `pubmed_count` | full query string | PubMed result count (int) |

Both the `drug AND disease` and `drug AND broader` PubMed counts are cached under `pubmed_count` using their respective query strings as keys. Same SHA-256-keyed JSON format and `CACHE_TTL` constant as the Open Targets client.

---

## 5. Infrastructure Setup (Docker)

```yaml
# docker-compose.yml addition
services:
  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: indication_scout
      POSTGRES_USER: scout
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

---

## 6. Sprint Mapping

| Task | Sprint | Status |
|------|--------|--------|
| Docker + pgvector setup | Sprint 1 | Not started |
| Abstract caching schema | Sprint 1 | Complete (SQLAlchemy ORM in `sqlalchemy/pubmed_abstracts.py`) |
| BioLORD-2023 embedding integration | Sprint 1 | Complete (`services/embeddings.py`, lazy singleton) |
| `fetch_and_cache` implementation | Sprint 1 | Partial (steps 1-3 done: `get_stored_pmids`, `fetch_new_abstracts`; steps 4-7 pending: embed, insert, wire) |
| `semantic_search` implementation | Sprint 1-2 | Not started |
| Re-ranking function | Sprint 2 | Not started |
| `expand_search_terms` implementation | Sprint 2 | Complete (`services/retrieval.py`, 5-axis LLM generation with caching) |
| Disease name normalization (Haiku) | Sprint 2 | Complete (`services/disease_normalizer.py`, two-step LLM with blocklist) |
| `synthesize` / Literature Agent integration | Sprint 2 | Not started |
| Full pipeline wiring | Sprint 2-3 | Not started |

---

## 7. Key Design Decisions

1. **pgvector over dedicated vector DB** — simplicity; dataset is small enough (~10k-50k abstracts)
2. **BioLORD-2023 for embeddings** — trained on UMLS + SNOMED-CT + biomedical definitions; sentence-level embedding model; state-of-the-art on biomedical similarity benchmarks; 768-dim vectors
3. **SentenceTransformer loading** — standard interface; BioLORD-2023 is a SentenceTransformer-compatible model
4. **Therapeutic query framing** — embed intent ("evidence for X as treatment for Y") not just keywords; enables conceptual matches
5. **Fetch → embed → store at ingest time** — embeddings computed once and cached; semantic search only needs to embed the query
6. **500 PMIDs per keyword query** — wider initial retrieval net; semantic search + re-rank handle noise reduction
7. **Two-stage reduction (20 → 5)** — semantic search casts a wide net; re-ranker applies precision filter before the LLM
8. **Grounded generation with PMIDs** — Claude synthesises from retrieved documents, not training weights; every claim in `EvidenceSummary` is traceable to a real paper
9. **Cache-first retrieval** — avoid redundant PubMed API calls and re-embedding
10. **LLM disease name normalization** — cheap Haiku calls instead of building a synonym dictionary or ontology traversal
11. **`DrugProfile` for query expansion** — flat LLM-facing projection of `RichDrugData` (name, synonyms, target gene symbols, mechanisms, ATC codes/descriptions, drug type) inform better PubMed queries than drug name alone
