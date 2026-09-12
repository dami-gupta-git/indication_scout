.PHONY: install lint lint-fix format format-check typecheck test test-regression \
	create-tables prefetch-model regression-reports regression-specs candidate-precision \
	seed-recall seed-examples regression container-smoke observability-up observability-down \
	frontend-check check ci fix

SRC = src/ tests/

# Stage banners, so a long `make check` / `make regression` run reads as a
# checklist rather than a wall of tool output. A stage's PASSED line is the last
# line of its recipe, so make aborts before it if the stage failed.
START = @printf '\n==> %s\n'
OK = @printf '    %s: PASSED\n'

install:
	pip install -e ".[dev]"

lint:
	$(START) "lint"
	ruff check $(SRC)
	$(OK) "lint"

lint-fix:
	ruff check --fix $(SRC)

format:
	black $(SRC)

format-check:
	$(START) "format-check"
	black --check $(SRC)
	$(OK) "format-check"

typecheck:
	$(START) "typecheck"
	mypy src/
	$(OK) "typecheck"

test:
	$(START) "unit tests"
	pytest tests/unit/
	$(OK) "unit tests"

# Offline half of the regression suite: the deterministic evidence-gate tests and
# the report-diff unit tests. No network, no LLM, no DB. The structural specs are
# not here — they need freshly generated reports (see regression-reports).
test-regression:
	$(START) "offline regression tests"
	pytest tests/regression/layer1_deterministic/ tests/regression/pipeline_replay/test_compare_reports.py
	$(OK) "offline regression tests"

# --- Live regression: the same steps the CI regression job runs ---------------
# Needs a reachable Postgres + pgvector (DATABASE_URL) and an Anthropic key.
# Drugs a live regression run covers. `?=` so the environment or the command line overrides it:
#   make regression REGRESSION_DRUGS="semaglutide sildenafil bupropion metformin"
# The precision and seed-recall checks and the structural specs restrict themselves to this list.
REGRESSION_DRUGS ?= semaglutide bupropion

create-tables:
	$(START) "create-tables"
	python scripts/create_tables.py
	$(OK) "create-tables"

prefetch-model:
	$(START) "prefetch-model"
	python scripts/prefetch_embedding_model.py
	$(OK) "prefetch-model"

# Regenerate the pinned drugs' reports. Hits the real data sources and the LLM.
# The structural specs assert against the newest test_reports/<drug>_*.json, so
# this must run before them.
#
# Runs REGRESSION_PARALLEL drugs at a time. Each run's output streams to the
# console as it happens, prefixed with the drug name so the concurrent runs can
# be told apart, and is also written to its own log file. The exit status is
# captured through a side file because the pipeline would otherwise report tee's.
# xargs exits non-zero if any run failed.
REGRESSION_LOG_DIR = results/regression-logs
REGRESSION_PARALLEL = 2

regression-reports:
	$(START) "regenerate reports ($(REGRESSION_DRUGS))"
	@mkdir -p $(REGRESSION_LOG_DIR)
	@printf '%s\n' $(REGRESSION_DRUGS) | PYTHONUNBUFFERED=1 xargs -P $(REGRESSION_PARALLEL) -n1 \
		sh -c 'drug=$$1; log=$(REGRESSION_LOG_DIR)/$$drug.log; \
		{ echo "=== scout find -d $$drug ==="; scout find -d "$$drug"; echo $$? > "$$log.status"; } 2>&1 \
			| while IFS= read -r line; do printf "[%s] %s\n" "$$drug" "$$line"; done | tee "$$log"; \
		status=$$(cat "$$log.status"); rm -f "$$log.status"; \
		echo "=== scout find -d $$drug finished (exit $$status) ===" | tee -a "$$log"; \
		exit $$status' sh
	$(OK) "regenerate reports"

regression-specs:
	$(START) "structural regression specs"
	REGRESSION_DRUGS="$(REGRESSION_DRUGS)" pytest -m regression_layer2 tests/regression/layer2_structural/
	$(OK) "structural regression specs"

candidate-precision:
	$(START) "candidate-selection precision"
	python scripts/check_candidate_precision.py \
		tests/regression/labels/candidate_precision.json \
		test_reports \
		results/precision/live_candidate_precision.md \
		--drugs $(REGRESSION_DRUGS)
	$(OK) "candidate-selection precision"

# Seed-phase candidate recall for the REGRESSION_DRUGS that tests/regression/labels/seed_recall.yaml
# names. One seed run per runbook cutoff; independent of the generated reports.
seed-recall:
	$(START) "seed-phase candidate recall"
	python scripts/check_seed_recall.py --drugs $(REGRESSION_DRUGS)
	$(OK) "seed-phase candidate recall"

# Refresh seed_examples/ from the payloads the live runs just wrote. Runs after the checks so a
# report that failed one never becomes a seed.
seed-examples:
	$(START) "seed examples"
	python scripts/seed_examples_from_reports.py $(REGRESSION_DRUGS)
	$(OK) "seed examples"

regression: create-tables prefetch-model regression-reports regression-specs candidate-precision seed-recall seed-examples
	@printf '\n==> live regression: ALL STAGES PASSED\n'

# Build the production image, run it against a fresh Postgres so its entrypoint applies
# the migrations, then require /health to answer and the migrated schema to match the
# ORM models. Needs a working Docker daemon; makes no network calls beyond image pulls.
container-smoke:
	$(START) "container smoke"
	bash scripts/container_smoke.sh
	$(OK) "container smoke"

observability-up:
	docker compose up -d --build

observability-down:
	docker compose down

frontend-check:
	$(START) "frontend check"
	cd frontend && npm ci && npm run lint && npm run build && npm test
	$(OK) "frontend check"

# Fast gate: everything that runs offline.
check: lint format-check typecheck test test-regression frontend-check
	@printf '\n==> check: ALL STAGES PASSED\n'

# Everything CI runs on a push, including the live regression job. Slow, and it
# spends real API/LLM calls.
ci: check regression container-smoke
	@printf '\n==> ci: ALL STAGES PASSED\n'

fix: lint-fix format
