.PHONY: install lint lint-fix format format-check typecheck test test-regression \
	create-tables prefetch-model regression-reports regression-specs candidate-precision \
	seed-recall regression \
	frontend-check check ci fix

SRC = src/ tests/

install:
	pip install -e ".[dev]"

lint:
	ruff check $(SRC)

lint-fix:
	ruff check --fix $(SRC)

format:
	black $(SRC)

format-check:
	black --check $(SRC)

typecheck:
	mypy src/

test:
	pytest tests/unit/

# Offline half of the regression suite: the deterministic evidence-gate tests and
# the report-diff unit tests. No network, no LLM, no DB. The structural specs are
# not here — they need freshly generated reports (see regression-reports).
test-regression:
	pytest tests/regression/layer1_deterministic/ tests/regression/pipeline_replay/test_compare_reports.py

# --- Live regression: the same steps the CI regression job runs ---------------
# Needs a reachable Postgres + pgvector (DATABASE_URL) and an Anthropic key.
REGRESSION_DRUGS = semaglutide sildenafil bupropion metformin

create-tables:
	python scripts/create_tables.py

prefetch-model:
	python scripts/prefetch_embedding_model.py

# Regenerate the pinned drugs' reports. Hits the real data sources and the LLM.
# The structural specs assert against the newest test_reports/<drug>_*.json, so
# this must run before them.
regression-reports:
	@for drug in $(REGRESSION_DRUGS); do \
		echo "=== scout find -d $$drug ==="; \
		scout find -d $$drug || exit 1; \
	done

regression-specs:
	pytest -m regression_layer2 tests/regression/layer2_structural/

candidate-precision:
	python scripts/check_candidate_precision.py \
		tests/regression/labels/candidate_precision.json \
		test_reports \
		results/precision/live_candidate_precision.md

# Seed-phase candidate recall for the drugs named in tests/regression/specs/seed_recall.yaml.
# One seed run per runbook cutoff; independent of the generated reports.
seed-recall:
	python scripts/check_seed_recall.py

regression: create-tables prefetch-model regression-reports regression-specs candidate-precision seed-recall

frontend-check:
	cd frontend && npm ci && npm run lint && npm run build && npm test

# Fast gate: everything that runs offline.
check: lint format-check typecheck test test-regression frontend-check

# Everything CI runs on a push, including the live regression job. Slow, and it
# spends real API/LLM calls.
ci: check regression

fix: lint-fix format
