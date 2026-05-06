# syntax=docker/dockerfile:1.7

# ---------- builder ----------
FROM python:3.10-slim AS builder

# Bring in uv (the same tool we use locally) from the official image.
COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /usr/local/bin/uv

WORKDIR /app

# Layer cache: copy dep manifests first so a code-only change doesn't
# trigger a full reinstall.
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1

# --frozen: refuse to update uv.lock; --no-dev: skip pytest/ruff/jupyter.
RUN uv sync --frozen --no-dev


# ---------- runtime ----------
FROM python:3.10-slim AS runtime

WORKDIR /app

# Install curl for the healthcheck (smallest possible addition).
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the resolved venv and the source from the builder stage.
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/pyproject.toml /app/pyproject.toml

# Bake the trained model artifacts into the image.
# Run `make train` locally before `docker build` so this directory is populated.
COPY models/ /app/models/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LOG_FORMAT=json \
    HOST=0.0.0.0 \
    PORT=8000

# Run as a non-root user.
RUN useradd --create-home --shell /bin/bash --uid 1001 churn \
    && chown -R churn:churn /app
USER churn

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl --fail --silent http://localhost:8000/health || exit 1

CMD ["uvicorn", "churn.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
