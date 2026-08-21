# Set PROVIDER to whichever LangChain extra matches your model.
# Examples: openai, google-genai, anthropic, groq, mistralai, deepseek,
#           together, fireworks, baseten, xai, perplexity, huggingface,
#           aws, google-vertexai, ollama
PROVIDER ?= ollama

.PHONY: install test lint run clean

install:         ## venv + dev + rag deps + your provider
	python3 -m venv .venv
	.venv/bin/pip install -e ".[dev,rag]"
	.venv/bin/pip install "langchain[$(PROVIDER)]"

test:            ## no API key needed
	.venv/bin/pytest -q

lint:
	.venv/bin/ruff check src tests

run:
	.venv/bin/agent "How hot is Chennai?"

clean:
	rm -rf .venv .pytest_cache .ruff_cache **/__pycache__
