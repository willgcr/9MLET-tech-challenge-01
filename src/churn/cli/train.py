"""End-to-end training: tune, fit, evaluate at multiple thresholds, persist artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import mlflow
import torch
from sklearn.model_selection import train_test_split

from churn.config import MODELS_DIR, RANDOM_SEED, ensure_directories
from churn.dataset.loader import load_raw_dataset
from churn.dataset.preprocessing import build_preprocessor, split_features_target
from churn.logging_config import configure_logging, get_logger
from churn.modeling.evaluation import operating_point_table
from churn.modeling.training import predict_proba, train_mlp, tune_mlp
from churn.tracking import start_run

logger = get_logger(__name__)

PREPROCESSOR_PATH: Path = MODELS_DIR / "preprocessor.joblib"
MODEL_PATH: Path = MODELS_DIR / "mlp.pt"
METADATA_PATH: Path = MODELS_DIR / "metadata.json"

TEST_SIZE: float = 0.20
INNER_VAL_SIZE: float = 0.20
PRODUCTION_OPERATING_POINT: str = "max_f1"


def _print_operating_points(table: dict[str, dict[str, float]]) -> None:
    cols = ("threshold", "precision", "recall", "f1", "accuracy", "roc_auc", "pr_auc")
    header = f"{'operating_point':<28}" + "".join(f"{c:>12}" for c in cols)
    print(header)
    print("-" * len(header))
    for name, m in table.items():
        row = f"{name:<28}" + "".join(f"{m[c]:>12.4f}" for c in cols)
        print(row)


def main() -> int:
    configure_logging(level="INFO", json_format=False)
    ensure_directories()

    logger.info("Loading raw dataset")
    df = load_raw_dataset()
    X, y = split_features_target(df)
    logger.info("Loaded dataset", extra={"rows": len(df), "cols": X.shape[1]})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )
    logger.info(
        "Stratified outer split",
        extra={"train_rows": len(X_train), "test_rows": len(X_test)},
    )

    logger.info("Tuning MLP hyperparameters via stratified CV")
    tuning_result = tune_mlp(X_train, y_train)
    best_config = tuning_result["best_config"]
    logger.info("Best config", extra={"config": str(best_config)})

    X_inner_tr, X_inner_val, y_inner_tr, y_inner_val = train_test_split(
        X_train,
        y_train,
        test_size=INNER_VAL_SIZE,
        stratify=y_train,
        random_state=RANDOM_SEED,
    )
    preprocessor = build_preprocessor()
    X_tr_t = preprocessor.fit_transform(X_inner_tr)
    X_val_t = preprocessor.transform(X_inner_val)
    X_test_t = preprocessor.transform(X_test)

    logger.info("Fitting final MLP with best config")
    model, history = train_mlp(
        X_tr_t,
        y_inner_tr.to_numpy(),
        X_val_t,
        y_inner_val.to_numpy(),
        hidden_dims=best_config["hidden_dims"],
        dropout=best_config["dropout"],
        lr=best_config["lr"],
    )
    logger.info("Final MLP trained", extra={"epochs_run": len(history)})

    y_test_np = y_test.to_numpy()
    y_proba = predict_proba(model, X_test_t)
    operating_points = operating_point_table(y_test_np, y_proba)
    production_threshold = operating_points[PRODUCTION_OPERATING_POINT]["threshold"]

    print()
    print("=" * 96)
    print("FINAL MLP - HELD-OUT TEST SET METRICS AT DIFFERENT OPERATING POINTS")
    print("=" * 96)
    _print_operating_points(operating_points)
    print()
    print(f"Production threshold (chosen: {PRODUCTION_OPERATING_POINT}): {production_threshold:.4f}")

    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": X_tr_t.shape[1],
            "hidden_dims": list(best_config["hidden_dims"]),
            "dropout": best_config["dropout"],
        },
        MODEL_PATH,
    )
    metadata = {
        "model": "ChurnMLP",
        "framework": "pytorch",
        "input_dim": X_tr_t.shape[1],
        "hidden_dims": list(best_config["hidden_dims"]),
        "dropout": best_config["dropout"],
        "lr": best_config["lr"],
        "outer_test_size": TEST_SIZE,
        "inner_val_size": INNER_VAL_SIZE,
        "random_seed": RANDOM_SEED,
        "production_operating_point": PRODUCTION_OPERATING_POINT,
        "production_threshold": production_threshold,
        "operating_points": operating_points,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    logger.info(
        "Artifacts persisted",
        extra={
            "preprocessor": str(PREPROCESSOR_PATH),
            "model": str(MODEL_PATH),
            "metadata": str(METADATA_PATH),
        },
    )

    with start_run(run_name="mlp_final", model="mlp", stage="final"):
        mlflow.log_params(
            {
                "model": "mlp_final",
                "framework": "pytorch",
                "test_size": TEST_SIZE,
                "inner_val_size": INNER_VAL_SIZE,
                "random_seed": RANDOM_SEED,
                "production_operating_point": PRODUCTION_OPERATING_POINT,
                "production_threshold": production_threshold,
                **{f"best_{k}": (str(v) if isinstance(v, tuple) else v) for k, v in best_config.items()},
            }
        )
        for op_name, m in operating_points.items():
            for k, v in m.items():
                mlflow.log_metric(f"{op_name}_{k}", float(v))
        mlflow.log_artifact(str(PREPROCESSOR_PATH))
        mlflow.log_artifact(str(MODEL_PATH))
        mlflow.log_artifact(str(METADATA_PATH))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
