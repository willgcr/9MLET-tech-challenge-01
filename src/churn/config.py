"""Project-wide configuration: paths, seeds, defaults.

Centralising these here means every module reads the same values and tests
can monkeypatch them in one place.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Repository root (this file lives at <repo>/src/churn/config.py).
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

# Load .env from the repo root into the process environment, if present.
# Existing environment variables take precedence (override=False) so CI and
# Docker can still inject values without bumping into a checked-out .env.
load_dotenv(REPO_ROOT / ".env", override=False)

# Data layout.
DATA_DIR: Path = REPO_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
INTERIM_DATA_DIR: Path = DATA_DIR / "interim"

# Model artifacts (sklearn pipeline + torch state dict + metadata).
MODELS_DIR: Path = REPO_ROOT / "models"

# MLflow tracking store. Local file-store by default; override with env var
# in CI or when pointing at a remote server.
MLFLOW_TRACKING_URI: str = os.environ.get(
    "MLFLOW_TRACKING_URI",
    f"file://{REPO_ROOT / 'mlruns'}",
)
MLFLOW_EXPERIMENT_NAME: str = os.environ.get(
    "MLFLOW_EXPERIMENT_NAME",
    "telco-churn",
)

# Reproducibility. Seed every RNG (python, numpy, torch) with this value.
RANDOM_SEED: int = int(os.environ.get("RANDOM_SEED", "42"))

# Dataset. Kaggle slug used by kagglehub; can be swapped without code changes.
KAGGLE_DATASET_SLUG: str = "blastchar/telco-customer-churn"

# Target column in the Telco Churn dataset.
TARGET_COLUMN: str = "Churn"
ID_COLUMN: str = "customerID"


def ensure_directories() -> None:
    """Create the data/model directories if they do not yet exist.

    Safe to call multiple times. Used by training scripts and tests.
    """
    for directory in (RAW_DATA_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
