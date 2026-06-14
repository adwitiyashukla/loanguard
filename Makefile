# LoanGuard developer Makefile

PYTHON ?= python
PIP ?= pip

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

.PHONY: setup
setup:  ## Install dev dependencies
	$(PIP) install -r requirements.txt -r requirements-dev.txt
	$(PIP) install -e .
	pre-commit install || true

.PHONY: data
data:  ## Download (or fall back to synthetic) data
	$(PYTHON) scripts/download_data.py --sample 250000

.PHONY: train
train:  ## Train end-to-end
	$(PYTHON) scripts/train.py --config config/config.yaml

.PHONY: serve
serve:  ## Run the FastAPI scoring service
	uvicorn src.api.main:app --reload --port 8000

.PHONY: dashboard
dashboard:  ## Run the Streamlit dashboard
	streamlit run src/dashboard/app.py

.PHONY: test
test:  ## Run unit tests with coverage
	pytest -q

.PHONY: smoke
smoke:  ## End-to-end smoke test (~1 minute on synthetic data)
	$(PYTHON) scripts/smoke_test.py

.PHONY: lint
lint:  ## Lint with ruff + check formatting with black
	ruff check src tests
	black --check src tests

.PHONY: fmt
fmt:  ## Auto-format code
	ruff check --fix src tests
	black src tests

.PHONY: docker-build
docker-build:  ## Build the API Docker image
	docker build -t loanguard:latest -f docker/Dockerfile .

.PHONY: docker-up
docker-up:  ## Run full stack (api + mlflow + dashboard + prometheus)
	docker compose -f docker/docker-compose.yml up --build

.PHONY: docker-down
docker-down:  ## Stop the full stack
	docker compose -f docker/docker-compose.yml down

.PHONY: clean
clean:  ## Remove caches and build artifacts
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
