"""MLP training loop with early stopping, plus a stratified-CV runner."""

from __future__ import annotations

import itertools
import random
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from churn.config import RANDOM_SEED
from churn.dataset.preprocessing import build_preprocessor
from churn.logging_config import get_logger
from churn.modeling.evaluation import compute_classification_metrics
from churn.modeling.mlp import ChurnMLP
from churn.tracking import start_run

logger = get_logger(__name__)

CV_FOLDS: int = 5


def seed_everything(seed: int = RANDOM_SEED) -> None:
    """Seed Python, NumPy and PyTorch RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_tensor(arr: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.tensor(np.asarray(arr), dtype=dtype)


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    hidden_dims: tuple[int, ...] = (64, 32),
    dropout: float = 0.2,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    max_epochs: int = 100,
    patience: int = 8,
    use_pos_weight: bool = True,
) -> tuple[ChurnMLP, list[dict[str, float]]]:
    """Train an MLP with early stopping on validation ROC AUC.

    Returns the model (with best weights restored) and the per-epoch history.
    """
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pos_weight: torch.Tensor | None = None
    if use_pos_weight:
        n_pos = float(y_train.sum())
        n_neg = float(len(y_train) - n_pos)
        if n_pos > 0:
            pos_weight = torch.tensor([n_neg / n_pos], device=device)

    model = ChurnMLP(
        input_dim=X_train.shape[1], hidden_dims=hidden_dims, dropout=dropout
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = DataLoader(
        TensorDataset(_to_tensor(X_train), _to_tensor(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    X_val_t = _to_tensor(X_val).to(device)
    y_val_t = _to_tensor(y_val).to(device)

    best_auc = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_since_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
        train_loss = running_loss / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()
            val_proba = torch.sigmoid(val_logits).cpu().numpy()
        val_metrics = compute_classification_metrics(y_val, val_proba)
        val_auc = val_metrics["roc_auc"]

        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **val_metrics}
        )

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                logger.info(
                    "Early stopping",
                    extra={"epoch": epoch, "best_val_roc_auc": round(best_auc, 4)},
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def predict_proba(model: ChurnMLP, X: np.ndarray) -> np.ndarray:
    """Run inference and return P(churn=1)."""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        logits = model(_to_tensor(X).to(device))
        return torch.sigmoid(logits).cpu().numpy()


def _cv_loop(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int,
    hidden_dims: tuple[int, ...],
    dropout: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Stratified-CV the MLP without touching MLflow. Returns per-fold + averages."""
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    fold_metrics: list[dict[str, Any]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds.split(X, y)):
        X_train_raw = X.iloc[train_idx]
        X_val_raw = X.iloc[val_idx]
        y_train_np = y.iloc[train_idx].to_numpy()
        y_val_np = y.iloc[val_idx].to_numpy()

        preprocessor = build_preprocessor()
        X_train = preprocessor.fit_transform(X_train_raw)
        X_val = preprocessor.transform(X_val_raw)

        model, history = train_mlp(
            X_train,
            y_train_np,
            X_val,
            y_val_np,
            hidden_dims=hidden_dims,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
        )
        y_proba = predict_proba(model, X_val)
        metrics = compute_classification_metrics(y_val_np, y_proba)
        metrics["fold"] = fold_idx
        metrics["epochs_run"] = len(history)
        fold_metrics.append(metrics)

    float_keys = [k for k, v in fold_metrics[0].items() if isinstance(v, float)]
    averaged = {
        f"cv_mean_{k}": float(np.mean([m[k] for m in fold_metrics])) for k in float_keys
    }
    return fold_metrics, averaged


def cross_validate_mlp(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int = CV_FOLDS,
    hidden_dims: tuple[int, ...] = (64, 32),
    dropout: float = 0.2,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    max_epochs: int = 100,
    patience: int = 8,
) -> dict[str, float]:
    """Stratified-CV the MLP and log everything as one MLflow run."""
    with start_run(run_name="mlp", model="mlp", stage="stage2"):
        mlflow.log_params(
            {
                "model": "mlp",
                "framework": "pytorch",
                "cv_folds": n_splits,
                "random_seed": RANDOM_SEED,
                "hidden_dims": str(hidden_dims),
                "dropout": dropout,
                "lr": lr,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "max_epochs": max_epochs,
                "patience": patience,
            }
        )

        fold_metrics, averaged = _cv_loop(
            X,
            y,
            n_splits=n_splits,
            hidden_dims=hidden_dims,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
        )
        for m in fold_metrics:
            for k, v in m.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"fold{m['fold']}_{k}", float(v))
        for k, v in averaged.items():
            mlflow.log_metric(k, v)

        logger.info(
            "MLP cross-validation complete",
            extra={k: round(v, 4) for k, v in averaged.items()},
        )
        return averaged


DEFAULT_GRID: dict[str, list] = {
    "hidden_dims": [(32, 16), (64, 32), (128, 64)],
    "dropout": [0.1, 0.3],
    "lr": [1e-3, 5e-4],
}


def tune_mlp(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    grid: dict[str, list] | None = None,
    n_splits: int = CV_FOLDS,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    max_epochs: int = 100,
    patience: int = 8,
) -> dict[str, Any]:
    """Grid-search MLP hyperparameters via stratified CV.

    Logs one parent run plus a nested child run per config. Returns
    ``{"best_config": dict, "best_metrics": dict, "all_runs": list}``.
    """
    grid = grid or DEFAULT_GRID
    keys = list(grid.keys())
    combos = list(itertools.product(*grid.values()))
    logger.info("Starting MLP grid search", extra={"n_configs": len(combos)})

    best_score = -1.0
    best_config: dict[str, Any] = {}
    best_metrics: dict[str, float] = {}
    all_runs: list[dict[str, Any]] = []

    with start_run(run_name="mlp_grid_search", model="mlp", stage="stage2_tune"):
        mlflow.log_params(
            {
                "n_configs": len(combos),
                "n_splits": n_splits,
                "grid": str(grid),
                "random_seed": RANDOM_SEED,
            }
        )

        for idx, values in enumerate(combos):
            config = dict(zip(keys, values, strict=True))
            with mlflow.start_run(nested=True, run_name=f"config_{idx}"):
                mlflow.log_params(
                    {
                        **{k: (str(v) if isinstance(v, tuple) else v) for k, v in config.items()},
                        "config_idx": idx,
                    }
                )
                _, averaged = _cv_loop(
                    X,
                    y,
                    n_splits=n_splits,
                    hidden_dims=config["hidden_dims"],
                    dropout=config["dropout"],
                    lr=config["lr"],
                    weight_decay=weight_decay,
                    batch_size=batch_size,
                    max_epochs=max_epochs,
                    patience=patience,
                )
                for k, v in averaged.items():
                    mlflow.log_metric(k, v)

                score = averaged["cv_mean_roc_auc"]
                all_runs.append({"config": config, "metrics": averaged})
                logger.info(
                    "Tuning config done",
                    extra={
                        "idx": idx,
                        "config": str(config),
                        "roc_auc": round(score, 4),
                        "f1": round(averaged["cv_mean_f1"], 4),
                    },
                )
                if score > best_score:
                    best_score = score
                    best_config = config
                    best_metrics = averaged

        mlflow.log_metric("best_cv_mean_roc_auc", best_score)
        mlflow.log_params(
            {f"best_{k}": (str(v) if isinstance(v, tuple) else v) for k, v in best_config.items()}
        )
        logger.info(
            "Grid search complete",
            extra={"best_config": str(best_config), "best_roc_auc": round(best_score, 4)},
        )
        return {"best_config": best_config, "best_metrics": best_metrics, "all_runs": all_runs}
