-- Creates the test database inside the same Postgres instance as the main DB.
-- This script runs automatically on first container startup via
-- docker-entrypoint-initdb.d. It is idempotent: the IF NOT EXISTS guard
-- prevents errors if the DB already exists.
SELECT 'CREATE DATABASE scout_test OWNER scout'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'scout_test')\gexec
