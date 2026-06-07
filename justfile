set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

default:
    @just --list

install:
    uv sync --dev

test:
    uv run pytest

lint:
    uv run ruff check .

format:
    uv run ruff format .

typecheck:
    uv run mypy streamwright tests

check: lint typecheck test

# Run every provider smoke script in sequence (real API calls).
# Loads .env at the repo root via python-dotenv inside each script.
smoke:
    uv run python examples/smoke/run_all.py

# Run the live integration test suite (real API calls, costs money).
# Skips per-test when a provider's key is unset.
[unix]
integration:
    STREAMWRIGHT_RUN_INTEGRATION=1 uv run pytest tests/integration/ -v

[windows]
integration:
    $env:STREAMWRIGHT_RUN_INTEGRATION = '1'; uv run pytest tests/integration/ -v
