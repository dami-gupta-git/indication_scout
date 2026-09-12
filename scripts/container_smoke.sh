#!/usr/bin/env bash
# Production-container smoke check.
#
# Builds the prod image, starts a fresh pgvector Postgres, and runs the image against
# it so the entrypoint applies `alembic upgrade head` on an empty database. Then:
#   1. the API must answer on /health, and
#   2. `alembic check` must report no pending autogenerate diff, i.e. the migrations
#      produce exactly the schema the ORM models declare.
#
# Everything runs on a private Docker network created and torn down here, so the same
# command works locally and on a CI runner.
set -euo pipefail

IMAGE="${SMOKE_IMAGE:-indication-scout:smoke}"
NETWORK="scout-smoke-net-$$"
DB_CONTAINER="scout-smoke-db-$$"
API_CONTAINER="scout-smoke-api-$$"
DB_USER=smoke
DB_NAME=smoke
DB_PASSWORD=smoke
# Published on the host so the health probe does not depend on a curl inside the image.
API_PORT="${SMOKE_API_PORT:-18000}"
DB_READY_TIMEOUT=60
HEALTH_TIMEOUT=120

cleanup() {
  docker rm -f "$API_CONTAINER" "$DB_CONTAINER" >/dev/null 2>&1 || true
  docker network rm "$NETWORK" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "smoke: building production image ($IMAGE) ..."
DOCKER_BUILDKIT=1 docker build --target api -t "$IMAGE" .

docker network create "$NETWORK" >/dev/null

echo "smoke: starting Postgres ..."
docker run -d --name "$DB_CONTAINER" --network "$NETWORK" \
  -e POSTGRES_USER="$DB_USER" \
  -e POSTGRES_PASSWORD="$DB_PASSWORD" \
  -e POSTGRES_DB="$DB_NAME" \
  pgvector/pgvector:pg16 >/dev/null

# Probe over TCP, not the Unix socket: the image's entrypoint first runs a temporary server on the
# socket only, and a socket probe reports ready while that server is about to shut down.
db_ready() {
  docker exec "$DB_CONTAINER" pg_isready -h 127.0.0.1 -U "$DB_USER" -d "$DB_NAME" >/dev/null 2>&1
}
for _ in $(seq "$DB_READY_TIMEOUT"); do
  if db_ready; then
    break
  fi
  sleep 1
done
if ! db_ready; then
  echo "smoke: Postgres did not become ready within ${DB_READY_TIMEOUT}s" >&2
  docker logs "$DB_CONTAINER" >&2
  exit 1
fi

# The vector extension is created by the ORM bootstrap script in the regression job;
# the migrations expect it to exist already, so create it here too.
docker exec "$DB_CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" \
  -c "CREATE EXTENSION IF NOT EXISTS vector" >/dev/null

DATABASE_URL="postgresql+psycopg2://${DB_USER}:${DB_PASSWORD}@${DB_CONTAINER}:5432/${DB_NAME}"

echo "smoke: starting the production container ..."
docker run -d --name "$API_CONTAINER" --network "$NETWORK" \
  -e DATABASE_URL="$DATABASE_URL" \
  -e DB_PASSWORD="$DB_PASSWORD" \
  -p "${API_PORT}:8000" \
  "$IMAGE" >/dev/null

echo "smoke: waiting for /health ..."
healthy=0
for _ in $(seq "$HEALTH_TIMEOUT"); do
  if ! docker inspect -f '{{.State.Running}}' "$API_CONTAINER" 2>/dev/null | grep -q true; then
    echo "smoke: the container exited before becoming healthy" >&2
    docker logs "$API_CONTAINER" >&2
    exit 1
  fi
  if curl -fsS "http://localhost:${API_PORT}/health" >/dev/null 2>&1; then
    healthy=1
    break
  fi
  sleep 1
done
if [ "$healthy" -ne 1 ]; then
  echo "smoke: /health did not respond within ${HEALTH_TIMEOUT}s" >&2
  docker logs "$API_CONTAINER" >&2
  exit 1
fi
curl -fsS "http://localhost:${API_PORT}/health"
echo

echo "smoke: checking the migrated schema matches the ORM models ..."
if ! docker exec -e DATABASE_URL="$DATABASE_URL" -e DB_PASSWORD="$DB_PASSWORD" \
     "$API_CONTAINER" alembic check; then
  echo "smoke: migrations and models disagree — a migration is missing or incomplete" >&2
  exit 1
fi

echo "smoke: PASSED"
