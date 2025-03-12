.PHONY: lint format install-dev

install-dev:
	pip install -e ".[dev]"

format:
	black --preview .

lint:
	black --check .
