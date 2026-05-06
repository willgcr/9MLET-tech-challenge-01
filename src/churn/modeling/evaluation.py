"""Classification metrics used to compare every model on the same yardstick."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Return the canonical metric set for binary churn classification.

    Threshold-independent: ``roc_auc`` and ``pr_auc`` (operate on probabilities).
    Threshold-dependent: ``accuracy``, ``precision``, ``recall``, ``f1``
    (computed at the supplied ``threshold``, default 0.5).
    Confusion-matrix counts (tn, fp, fn, tp) are also returned for cost analysis.
    """
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def expected_cost(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    threshold: float,
    fp_cost: float,
    fn_cost: float,
) -> float:
    """Total cost at the given decision threshold, given per-error costs."""
    y_pred = (y_proba >= threshold).astype(int)
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return fp * fp_cost + fn * fn_cost


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    fp_cost: float = 1.0,
    fn_cost: float = 5.0,
    n_grid: int = 101,
) -> dict[str, float]:
    """Grid-search the decision threshold that minimises ``fp*fp_cost + fn*fn_cost``.

    Default ratio (1:5) reflects the common churn assumption: missing a churner
    costs more than annoying a stayer with a retention offer.
    """
    grid = np.linspace(0.0, 1.0, n_grid)
    costs = np.array(
        [
            expected_cost(
                y_true, y_proba, threshold=t, fp_cost=fp_cost, fn_cost=fn_cost
            )
            for t in grid
        ]
    )
    best_idx = int(np.argmin(costs))
    return {
        "threshold": float(grid[best_idx]),
        "cost": float(costs[best_idx]),
        "fp_cost": float(fp_cost),
        "fn_cost": float(fn_cost),
    }


def find_threshold_for_max_f1(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Decision threshold that maximises F1 on the supplied probabilities."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = int(np.argmax(f1[:-1]))
    return float(thresholds[best_idx])


def find_threshold_for_target_recall(
    y_true: np.ndarray, y_proba: np.ndarray, *, min_recall: float = 0.85
) -> float:
    """Highest threshold that still achieves ``recall >= min_recall``.

    Picking the *highest* such threshold means we trade off as little precision
    as possible while still meeting the recall target. Falls back to the
    threshold with maximum recall when the target is unreachable.
    """
    _, recall, thresholds = precision_recall_curve(y_true, y_proba)
    valid = recall[:-1] >= min_recall
    idx = int(np.where(valid)[0].max()) if valid.any() else int(np.argmax(recall[:-1]))
    return float(thresholds[idx])


def operating_point_table(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    min_recall: float = 0.85,
    fp_cost: float = 1.0,
    fn_cost: float = 5.0,
) -> dict[str, dict[str, float]]:
    """Evaluate the model at four operating points and return all metrics."""
    points = {
        "default_0.5": 0.5,
        "max_f1": find_threshold_for_max_f1(y_true, y_proba),
        f"recall_min_{min_recall:.2f}": find_threshold_for_target_recall(
            y_true, y_proba, min_recall=min_recall
        ),
        f"cost_optimal_{int(fp_cost)}_to_{int(fn_cost)}": find_optimal_threshold(
            y_true, y_proba, fp_cost=fp_cost, fn_cost=fn_cost
        )["threshold"],
    }
    return {
        name: {**compute_classification_metrics(y_true, y_proba, threshold=t), "threshold": t}
        for name, t in points.items()
    }
