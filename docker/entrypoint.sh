#!/usr/bin/env bash
# Container entrypoint: wait for Postgres, apply migrations, then run the CMD.
#
# Migrations run here (not as a separate service) so `docker compose up` is a
# single zero-step command. The DB connection details come from DATABASE_URL.
set -euo pipefail

# Derive host/port from DATABASE_URL for the readiness probe. Falls back to the
# compose service name + default port if parsing fails.
db_host="$(python -c "from urllib.parse import urlparse,unquote; import os; u=urlparse(os.environ['DATABASE_URL'].replace('+psycopg2','')); print(u.hostname or 'db')")"
db_port="$(python -c "from urllib.parse import urlparse; import os; u=urlparse(os.environ['DATABASE_URL'].replace('+psycopg2','')); print(u.port or 5432)")"

echo "entrypoint: waiting for Postgres at ${db_host}:${db_port} ..."
until pg_isready -h "$db_host" -p "$db_port" >/dev/null 2>&1; do
  sleep 1
done
echo "entrypoint: Postgres is up."

echo "entrypoint: applying migrations (alembic upgrade head) ..."
alembic upgrade head

echo "entrypoint: starting: $*"
exec "$@"
