.PHONY: lint test

lint:
	uv run ruff check .
	uv run python scripts/check_lines.py

test:
	uv run pytest -q
