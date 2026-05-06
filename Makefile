.PHONY: help install lock lint format test train serve mlflow-ui docker clean

UV  ?= uv
RUN := $(UV) run
PORT ?= 8000

help:
	@echo "make install     - sync .venv from pyproject.toml + uv.lock"
	@echo "make lock        - regenerate uv.lock from pyproject.toml"
	@echo "make lint        - ruff check (no fixes)"
	@echo "make format      - ruff format + ruff --fix"
	@echo "make test        - pytest with coverage"
	@echo "make train       - train baselines + MLP, log to MLflow"
	@echo "make serve       - run FastAPI on PORT (default 8000)"
	@echo "make mlflow-ui   - launch MLflow UI on port 5000"
	@echo "make docker      - build the Docker image"
	@echo "make docker-up   - rebuild and start the container"
	@echo "make docker-down - stop and remove the container"
	@echo "make clean       - remove caches and build artifacts"

install:
	$(UV) sync --extra dev

lock:
	$(UV) lock

lint:
	$(RUN) ruff check src tests

format:
	$(RUN) ruff format src tests
	$(RUN) ruff check --fix src tests

test:
	$(RUN) pytest --cov

train:
	$(RUN) python -m churn.cli.train

serve:
	$(RUN) uvicorn churn.api.app:app --host 0.0.0.0 --port $(PORT) --reload

mlflow-ui:
	$(RUN) mlflow ui --backend-store-uri ./mlruns --port 5000

docker:
	docker compose -f build/docker-compose.yml build

docker-up:
	docker compose -f build/docker-compose.yml up -d --build

docker-down:
	docker compose -f build/docker-compose.yml down

clean:
	rm -rf build dist .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name "*.egg-info" -prune -not -path "./.venv/*" -exec rm -rf {} +
