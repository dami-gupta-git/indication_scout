.PHONY: install lint lint-fix format format-check typecheck test frontend-check check fix

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

frontend-check:
	cd frontend && npm ci && npm run lint && npm run build && npm test

check: lint format-check typecheck test frontend-check

fix: lint-fix format
