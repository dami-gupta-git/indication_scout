# pgvector Index Playbook

Reference for setting up, deciding on, and maintaining vector indexes on
`pubmed_abstracts.embedding` (768-dim BioLORD-2023 vectors).

## 1. Base setup (already in place)

- Extension: `CREATE EXTENSION IF NOT EXISTS vector` (in
  `alembic/versions/f0ccb024a181_create_pubmed_abstracts_table.py`).
- Column: `embedding vector(768)` via `pgvector.sqlalchemy.vector.VECTOR(dim=768)`.
- Docker: `pgvector/pgvector:pg16` image in `docker-compose.yml`.
- No index currently exists on `embedding`. Every query in the codebase
  (`services/retrieval.py`, `ml_models/trial_risk/literature.py`,
  `ml_models/trial_risk/inspect.py`) pre-filters with
  `WHERE pmid = ANY(:pmids)` before computing `<=>` distance, scoped to
  candidate sets of ~100 rows. An ANN index buys nothing on that access
  pattern — exact brute-force distance over a filtered set is correct as-is.

## 2. Decide if you actually need an index

Only add one if you introduce a query that ranks across the **whole**
`pubmed_abstracts` table (or a large unfiltered slice of it), e.g. a future
"find similar papers across all cached drugs/diseases" feature. Checklist:

- [ ] Query does NOT restrict by `pmid = ANY(...)` or another highly
      selective filter first.
- [ ] Table has grown past a few thousand rows where a full scan is
      measurably slow (`EXPLAIN ANALYZE` the query first — don't guess).
- [ ] You can tolerate approximate (not exact) nearest-neighbor results.

If any of those is false, skip the index.

## 3. Which index type

| | IVFFlat | HNSW |
|---|---|---|
| Build cost | Low | Higher |
| Query speed/recall | Good, needs tuning (`lists`/`probes`) | Generally better recall at same speed |
| Needs data before building | Yes — clusters trained from existing rows | No — builds incrementally |
| Memory | Lower | Higher |
| Rebuild need as data grows | Yes, periodically | Rarely |

Default recommendation for this project if/when needed: **HNSW**. The
dataset is modest (~10k–50k abstracts per `docs/rag.md`), memory isn't a
constraint, and HNSW avoids the "must rebuild as the table grows" trap that
IVFFlat has. Only prefer IVFFlat if build time or memory becomes a real
constraint at much larger scale.

## 4. Applying an index (Alembic migration)

Create a new revision, e.g. `alembic revision -m "add hnsw index on pubmed_abstracts embedding"`:

```python
from alembic import op

def upgrade() -> None:
    op.execute(
        """
        CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_pubmed_abstracts_embedding_hnsw
        ON pubmed_abstracts
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        """
    )

def downgrade() -> None:
    op.execute("DROP INDEX CONCURRENTLY IF EXISTS ix_pubmed_abstracts_embedding_hnsw")
```

Notes:

- `CONCURRENTLY` avoids locking writes during build — required since
  `fetch_and_cache` inserts rows continuously. `CONCURRENTLY` can't run
  inside a transaction block; if Alembic wraps migrations in a transaction,
  set `op.execute(...)` outside autocommit or configure the migration with
  `autocommit_block()`.
- `vector_cosine_ops` must match the distance operator used in queries
  (`<=>`, cosine). If a query ever switches to `<->` (L2) or `<#>` (inner
  product), the ops class must match or the index won't be used.
- If choosing IVFFlat instead:

```sql
CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_pubmed_abstracts_embedding_ivfflat
ON pubmed_abstracts
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

  Build this only after bulk-loading a representative sample of data —
  building on a near-empty table gives degenerate clusters.

## 5. Tuning knobs

**HNSW**
- `m` (default 16): graph connections per node. Higher = better recall,
  more memory/build time.
- `ef_construction` (default 64): candidate list size at build time. Higher
  = better index quality, slower build.
- `ef_search` (session/query-level, default 40): candidate list size at
  query time. Set per-query: `SET hnsw.ef_search = 100;` before the query
  for higher recall at the cost of latency.

**IVFFlat**
- `lists` at build time: rule of thumb `rows / 1000` for up to ~1M rows, or
  `sqrt(rows)` for smaller tables. For ~50k abstracts, `lists = 100–200`.
- `probes` at query time: `SET ivfflat.probes = 10;` — more probes = better
  recall, slower query. Start at `sqrt(lists)`.

## 6. Maintenance runbook

- **Monitor query plans.** Periodically `EXPLAIN ANALYZE` the semantic
  search query to confirm the index is actually used (`Index Scan using
  ix_pubmed_abstracts_embedding_hnsw`) rather than falling back to `Seq
  Scan`. If a filter (`pmid = ANY(...)`) is present and highly selective,
  Postgres may correctly choose the seq scan over the index — that's
  expected, not a bug.
- **`ANALYZE pubmed_abstracts;`** after large bulk inserts so the planner's
  row-count estimates stay accurate. `fetch_and_cache` inserts in batches;
  consider running `ANALYZE` after large backfills (e.g.
  `scripts/backfill_pubtypes.py`-style jobs) rather than on every insert.
- **IVFFlat only — periodic `REINDEX`.** As the table grows well past the
  row count used when `lists` was chosen, cluster quality degrades (recall
  drops silently, no error). Re-run `REINDEX INDEX CONCURRENTLY
  ix_pubmed_abstracts_embedding_ivfflat;` after roughly a 2–3x growth in row
  count, or recreate with a larger `lists` value.
- **HNSW — no periodic rebuild needed**, but `VACUUM` still matters for
  dead tuples if rows are ever deleted/updated (unlikely here since
  `pubmed_abstracts` is insert-only, keyed by `pmid`).
- **Track index size vs. table size.** `SELECT pg_size_pretty(pg_relation_size('ix_pubmed_abstracts_embedding_hnsw'));`
  HNSW indexes can be several times the size of the raw vector data — budget
  disk/memory accordingly if the corpus grows past ~100k abstracts.
- **Before adding the index to prod, benchmark against the current seq
  scan** on the real filtered query shape (not just an unfiltered
  synthetic benchmark) to confirm it's actually a win for this codebase's
  access pattern, per §2.
