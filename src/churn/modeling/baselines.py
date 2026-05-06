"""Sklearn baseline classifiers and a stratified cross-validation runner."""

from __future__ import annotations

from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from churn.config import RANDOM_SEED
from churn.dataset.preprocessing import build_preprocessor
from churn.logging_config import get_logger
from churn.modeling.evaluation import compute_classification_metrics
from churn.tracking import start_run

logger = get_logger(__name__)

CV_FOLDS: int = 5


def build_baseline_classifiers() -> dict[str, BaseEstimator]:
    """Return the registry of sklearn classifiers compared against the MLP."""
    return {
        "dummy_majority": DummyClassifier(
            strategy="most_frequent",
            random_state=RANDOM_SEED,
        ),
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
    }


def _stratified_cv_metrics(
    pipe: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Run stratified K-fold CV and return per-fold + averaged metrics."""
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    fold_metrics: list[dict[str, Any]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipe.fit(X_train, y_train)
        y_proba = pipe.predict_proba(X_val)[:, 1]
        metrics = compute_classification_metrics(y_val.to_numpy(), y_proba)
        metrics["fold"] = fold_idx
        fold_metrics.append(metrics)

    float_keys = [k for k, v in fold_metrics[0].items() if isinstance(v, float)]
    averaged = {
        f"cv_mean_{k}": float(np.mean([m[k] for m in fold_metrics])) for k in float_keys
    }
    return fold_metrics, averaged


def cross_validate_baseline(
    name: str,
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int = CV_FOLDS,
) -> dict[str, float]:
    """Cross-validate a single baseline pipeline and log everything to MLflow."""
    pipe = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("classifier", estimator),
        ]
    )

    with start_run(run_name=f"baseline_{name}", model=name, stage="stage1"):
        mlflow.log_params(
            {
                "model": name,
                "cv_folds": n_splits,
                "random_seed": RANDOM_SEED,
            }
        )
        # Estimator-specific params (mlflow stringifies them).
        mlflow.log_params({f"clf_{k}": v for k, v in estimator.get_params().items()})

        fold_metrics, averaged = _stratified_cv_metrics(pipe, X, y, n_splits)

        for m in fold_metrics:
            for k, v in m.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"fold{m['fold']}_{k}", float(v))

        for k, v in averaged.items():
            mlflow.log_metric(k, v)

        logger.info(
            "Baseline cross-validation complete",
            extra={"model": name, **{k: round(v, 4) for k, v in averaged.items()}},
        )
        return averaged


def run_all_baselines(X: pd.DataFrame, y: pd.Series) -> dict[str, dict[str, float]]:
    """Train every baseline classifier with CV and return their averaged metrics."""
    return {
        name: cross_validate_baseline(name, est, X, y)
        for name, est in build_baseline_classifiers().items()
    }
