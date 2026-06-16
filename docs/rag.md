# IndicationScout RAG Pipeline

## Why RAG Is Needed

PubMed keyword search returns many irrelevant papers. Searching `bupropion AND obesity` returns
depression papers that incidentally mention obesity — burying the papers about bupropion as an actual
*treatment* for obesity.

The RAG pipeline embeds retrieved abstracts and runs semantic search so the Literature Agent receives
the most relevant papers, not just keyword matches.

**Confirmed empirically with:**

| Drug | Candidate Disease | Issue Without RAG |
|------|-------------------|-------------------|
| Bupropion | Obesity | Depression papers mentioning obesity dominate; Contrave papers buried |
| Bupropion | Narcolepsy | EFO name mismatch (`narcolepsy-cataplexy syndrome`) returned zero results |
| Sildenafil | BPH | Confirmed signal (PMID 40678732, 38448685) but mixed with erectile dysfunction papers |
| Sildenafil | Diabetic nephropathy | Semantic search separates PDE5-specific from non-selective PDE literature |
| Baricitinib | Eczematoid dermatitis | EFO name doesn't match PubMed indexing (`atopic dermatitis`) |

---

## Architecture

Implemented as `RetrievalService` in `services/retrieval.py`, orchestrated by `runners/rag_runner.py`.
Processes a drug across its top disease indications (capped at `settings.literature_top_k`).
`RetrievalService.get_drug_competitors(chembl_id, date_before)` fetches raw competitor data from
`OpenTargetsClient` (which returns `CompetitorRawData`), uses an LLM call (`merge_duplicate_diseases`
in `services/disease_helper.py`) to deduplicate disease names, removes overly broad terms, sorts by
competitor count, and slices to the top `literature_top_k`. It returns `dict[str, set[str]]`
(disease name → set of competitor drug names).

```
Drug name
  |
  v
RetrievalService.get_drug_competitors  -->  raw data from OpenTargetsClient
  |                                          + LLM merge/dedup
  |                                          + sort + top-literature_top_k slice
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

**Output:** `dict[str, EvidenceSummary]` mapping each disease to a structured evidence summary with
strength rating, study types, key findings, and supporting PMIDs.

### Where RAG Fits in the Full Pipeline

```
┌─────────────┐     ┌─────────────┐
│   Path 1     │     │   Path 2     │
│  (target-    │     │  (drug class │
│   disease    │     │   analogy)   │
│   assoc.)    │     │              │
└──────┬───────┘     └──────┬───────┘
       │                     │
       └──────────┬──────────┘
                  │
                  ▼
          ┌───────────────┐
          │   Merge &      │  Deduplicate by EFO ID,
          │   Rank Top 10  │  flag dual-path candidates
          └───────┬────────┘
                  │
                  ▼
    ┌─────────────────────────────┐
    │     For each candidate:      │
    │     RAG processing steps     │
    └─────────────┬────────────────┘
                  │
                  ▼
          ┌───────────────┐
          │  Evidence      │  Combine Path scores +
          │  Scoring &     │  literature + trial evidence
          │  Report Gen    │
          └────────────────┘
```

The RAG pipeline sits between raw PubMed retrieval and the Literature Agent. It does not change the
discovery logic (Path 1 / Path 2) — it improves the quality of evidence presented to the LLM for
synthesis.

---

## Pipeline Stages

### Stage 0 — Query Expansion (`expand_search_terms`)

Send a `DrugProfile` (flat LLM-facing projection of `RichDrugData` — name, synonyms, target gene
symbols, mechanisms of action, ATC codes, ATC descriptions, drug type) to Haiku
(`claude-haiku-4-5-20251001`) to generate diverse PubMed keyword queries across 5 axes: drug name,
drug class + organ, mechanism + organ, target gene, synonym. Per-axis caps yield 5–10 queries total.
Organ term is pre-extracted via a separate Haiku call (`extract_organ_term`). Both functions cache
results. The small-LLM model is `settings.small_llm_model` (currently
`claude-haiku-4-5-20251001`), not hardcoded.

Signature: `expand_search_terms(chembl_id: str, disease_name: str, drug_profile: DrugProfile) -> list[str]`.

Example for "Metformin + colorectal": `"metformin AND colorectal neoplasm"`,
`"biguanide AND colorectal"`, `"metformin AND AMPK AND colon"`, ...

### Stage 1 — Fetch and Cache (`fetch_and_cache`)

Signature: `fetch_and_cache(queries: list[str], db: Session, date_before: date | None = None) -> list[str]`.

```
Across all queries (batched, not per-query loops):
  1. Search PubMed E-utilities for each query (max_results = pubmed_search_default_max_results)
  2. Single bulk check against pgvector for already-stored PMIDs
  3. Single fetch for all new abstracts
  4. Discard abstract-less articles (letters, editorials) — no text to embed
  5. Single batch embed of remaining abstracts with BioLORD-2023
  6. Store (pmid, title, abstract, authors, journal, pub_date, embedding) in pgvector
  7. Return deduplicated list of all PMIDs returned by search (cached + newly added)
```

When `date_before` is set, the esummary date guard reads `pub_date` from pgvector for known
PMIDs and only calls esummary for unknowns.

**Note:** Not every returned PMID has a row in `pubmed_abstracts`. Articles without an abstract are
excluded before insert. Callers that pass this list to `semantic_search` will see those PMIDs
silently skipped by the `WHERE pmid = ANY(:pmids)` clause, which is intentional.

### Stage 2 — Semantic Search (`semantic_search`)

Signature: `semantic_search(disease: str, chembl_id: str, pmids: list[str], db: Session,
date_before: date | None = None) -> list[AbstractResult]`. Top-k is config-driven (a setting,
not a parameter). Each `AbstractResult` carries `pmid`, `title`, `abstract`, `similarity`,
`pubtype`.

```
1. Build therapeutic query: "Evidence for {drug} as a treatment for {disease}, ..."
2. Embed query with BioLORD-2023
3. Format query vector as canonical pgvector string: "[v1,v2,...,v768]"
4. Run cosine similarity search over pgvector, scoped to the provided PMIDs:
   - Inner query: compute 1 - (embedding <=> CAST(:query_vec AS vector)) AS similarity
   - WHERE pmid = ANY(:pmids) restricts to the current drug-disease PMID set
   - Outer query: ORDER BY similarity DESC, LIMIT top_k
5. Return top_k `AbstractResult` objects (pmid, title, abstract, similarity, pubtype)
```

Finds conceptually relevant papers even without exact keyword matches (e.g. "biguanide antineoplastic
mechanisms" matches a metformin/cancer query).

### Stage 3 — Synthesize (`synthesize`)

`synthesize(chembl_id, disease, top_abstracts: list[AbstractResult], holdout_mode=False)` stuffs the
top abstracts into a Claude (`settings.llm_model`, currently `claude-sonnet-4-6`) prompt. Claude reads
the actual retrieved papers — not training data. Output is a structured `EvidenceSummary` with PMIDs
attached to every claim.

```python
EvidenceSummary(
    summary: str = "",
    study_count: int = 0,
    # strength = evidence quantity/quality only; direction = which way it points
    strength: Literal["strong", "moderate", "weak", "none"] = "none",
    direction: Literal["supports", "contradicts", "mixed", "none"] = "none",
    # whether strength/direction grade THIS drug or only its class
    evidence_basis: Literal["drug_specific", "class_level", "none"] = "none",
    is_observational: bool | None = None,   # None = undetermined (no-data)
    key_findings: list[str] = [],
    supporting_pmids: list[str] = [],
    contradicting_pmids: list[str] = [],
)
```

---

## Disease Name Normalizer (`services/disease_helper.py`)

Converts raw Open Targets disease names (e.g. `"narcolepsy-cataplexy syndrome"`) into PubMed-friendly
search terms (e.g. `"narcolepsy"`) before they are passed to `get_pubmed_query`.

**Two-step LLM strategy:**

1. **Normalize** — Haiku prompt strips subtypes, staging, etiology, and genetic qualifiers while
   preserving organ specificity. If the disease has a well-known synonym, both are returned joined
   with `OR` (e.g. `"eczema OR dermatitis"`).
2. **Verify** — If a `drug_name` is provided, the normalized term is verified with a PubMed count
   (`drug AND disease`). If the count is below `MIN_RESULTS` (3), a second Haiku call generalises to
   a broader category. The broader term is used only if it also has `>= MIN_RESULTS` hits and does
   not collapse to an over-generic term in `BROADENING_BLOCKLIST` (defined in `constants.py`; e.g.
   `"cancer"`, `"carcinoma"`, `"disease"`, `"syndrome"`).

**File-based cache (`cache/`, config-driven TTL (currently 60 days)):**

| Namespace | Cache key | Cached value |
|-----------|-----------|--------------|
| `disease_norm` | `raw_term` | LLM-normalized string (e.g. `"narcolepsy"`) |
| `pubmed_count` | full query string | PubMed result count (int) |

Both the `drug AND disease` and `drug AND broader` PubMed counts are cached under `pubmed_count`
using their respective query strings as keys. Same SHA-256-keyed JSON format and `CACHE_TTL` constant
as the Open Targets client.

**Pre-merge normalization in the competitor pipeline:** `RetrievalService._normalize_disease_groups()`
(which called `llm_normalize_disease` per disease and merged groups collapsing to the same key) is
**deprecated and commented out** in `services/retrieval.py`. Disease dedup now happens solely through
`merge_duplicate_diseases` in `services/disease_helper.py`.

---

## PostgreSQL + pgvector (Abstract Cache + Vector Store)

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

-- Track which drug-disease queries have been run (not yet implemented --
-- no ORM model or migration exists for this table)
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

---

## Embedding Model

**BioLORD-2023** (`FremyCompany/BioLORD-2023`). Used for both abstract embeddings (stored at fetch
time) and query embeddings (at search time). Output dimension: **768**.

Trained on UMLS ontology (2023AA), SNOMED-CT, and 400k GPT-3.5-generated biomedical definitions,
using a three-phase strategy: contrastive learning from knowledge graphs → supervised
self-distillation → weight averaging. Achieves state-of-the-art on MedSTS (clinical sentence
similarity) and EHR-Rel-B (biomedical concept representation) benchmarks.

**Loading:** via `sentence-transformers` (included in `pyproject.toml` runtime dependencies).
Implemented in `services/embeddings.py` as a lazy-loaded module-level singleton.

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("FremyCompany/BioLORD-2023")
```

**Query formulation:** encode therapeutic intent, not just keywords:
- Keyword query: `"bupropion AND obesity"`
- Embedding query: `"Evidence for bupropion as a treatment for obesity, including clinical trials,
  efficacy data, and mechanism of action"`

---

## Infrastructure Setup (Docker)

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
      - "5438:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

---

## Key Design Decisions

1. **pgvector over a dedicated vector DB (Pinecone, Weaviate, etc.)** — single infrastructure
   dependency (Postgres already needed); dataset is small (~10k–50k abstracts); simpler Docker setup.
2. **BioLORD-2023 for embeddings** — trained on UMLS + SNOMED-CT + biomedical definitions;
   sentence-level embedding model; state-of-the-art on biomedical similarity benchmarks; 768-dim.
3. **SentenceTransformer loading** — standard interface; BioLORD-2023 is SentenceTransformer-
   compatible.
4. **Therapeutic query framing** — embed intent ("evidence for X as treatment for Y") not just
   keywords; enables conceptual matches.
5. **Fetch → embed → store at ingest time** — embeddings computed once and cached; semantic search
   only needs to embed the query.
6. **200 PMIDs per keyword query** (`PUBMED_MAX_RESULTS` env var → `pubmed_search_default_max_results`
   setting; 200 in prod) — wider initial retrieval net; semantic search handles noise reduction;
   reduced from 500 to limit embedding time on cold cache.
7. **Grounded generation with PMIDs** — Claude synthesises from retrieved documents, not training
   weights; every claim in `EvidenceSummary` is traceable to a real paper.
8. **Cache-first retrieval** — avoid redundant PubMed API calls and re-embedding.
9. **LLM disease name normalization** — cheap Haiku calls instead of building a synonym dictionary
   or ontology traversal.
10. **`DrugProfile` for query expansion** — flat LLM-facing projection of `RichDrugData` (name,
    synonyms, target gene symbols, mechanisms, ATC codes/descriptions, drug type) inform better
    PubMed queries than drug name alone.

---

## Key Components Summary

- **Embedding model:** BioLORD-2023 (`FremyCompany/BioLORD-2023`), loaded locally via
  `sentence-transformers`. 768-dim vectors. Trained on UMLS + SNOMED-CT biomedical ontologies.
- **Vector store:** PostgreSQL + pgvector. Abstracts and embeddings stored in `pubmed_abstracts`
  table. Cosine similarity for search.
- **Cache layers:** File-based cache (`cache/` dir, SHA-256 keys, config-driven TTL (currently 60 days)) for LLM results,
  PubMed searches, and Open Targets data. pgvector itself acts as the abstract/embedding cache.
- **Service class:** `RetrievalService(cache_dir: Path)` in `services/retrieval.py` is the single
  entry point for all pipeline operations.
- **Runner:** `run_rag(drug_name, db, cache_dir) -> dict[str, EvidenceSummary]` in
  `runners/rag_runner.py` orchestrates the full pipeline. Disease indications are processed
  concurrently (capped by `RAG_DISEASE_CONCURRENCY`), with per-step timing logs and a final
  ranking by evidence strength.
