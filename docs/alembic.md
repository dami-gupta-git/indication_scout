# Alembic — Migration Guide

## Prerequisites

Docker container must be running before any migration commands:

```bash
docker compose up -d
```

---

## Creating a migration

After changing a SQLAlchemy model in `src/indication_scout/sqlalchemy/`, generate a migration:

```bash
alembic revision --autogenerate -m "describe the change"
```

Review the generated file in `alembic/versions/` before applying — autogenerate is not always perfect (e.g. it cannot detect column renames or custom types like `vector`).

---

## Applying migrations

```bash
alembic upgrade head      # apply all pending migrations
alembic upgrade +1        # apply one migration forward
```

---

## Rolling back

```bash
alembic downgrade -1      # roll back one migration
alembic downgrade base    # roll back all migrations
```

---

## Checking status

```bash
alembic current           # show current revision in the DB
alembic history           # show full migration history
```

---

## Notes

- `DATABASE_URL` is read from `.env` via `get_settings()` — ensure `.env` is present before running any alembic command.
- The first migration must manually add `CREATE EXTENSION IF NOT EXISTS vector;` before the `pubmed_abstracts` table, since autogenerate does not emit extension creation.
- Never edit a migration that has already been applied to a shared or production database — create a new one instead.
