"""MLflow setup and a thin context manager for runs."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import mlflow

from churn.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI


def configure_mlflow() -> None:
    """Point MLflow at the configured tracking URI and experiment."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


@contextmanager
def start_run(run_name: str, **tags: str) -> Iterator[Any]:
    """Open an MLflow run in the configured experiment, with optional tags."""
    configure_mlflow()
    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        yield run
