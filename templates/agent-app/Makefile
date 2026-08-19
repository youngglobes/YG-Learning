.PHONY: install install-claude install-ollama test lint run clean

install:         ## base + dev + rag
	python3 -m venv .venv
	.venv/bin/pip install -e ".[dev,rag]"

install-claude: install
	.venv/bin/pip install -e ".[claude]"

install-ollama: install
	.venv/bin/pip install -e ".[ollama]"

test:            ## no API key needed
	.venv/bin/pytest -q

lint:
	.venv/bin/ruff check src tests

run:
	.venv/bin/agent "How hot is Chennai?"

clean:
	rm -rf .venv .pytest_cache .ruff_cache **/__pycache__
