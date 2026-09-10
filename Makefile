.PHONY: lint lint-fix format format-check typecheck test check fix

SRC = src/ tests/

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

check: lint format-check typecheck test

fix: lint-fix format
