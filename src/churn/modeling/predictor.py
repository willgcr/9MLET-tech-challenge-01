"""Load saved artifacts and produce churn predictions for raw customer rows."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import torch
from sklearn.pipeline import Pipeline

from churn.config import MODELS_DIR
from churn.modeling.mlp import ChurnMLP
from churn.modeling.training import predict_proba

PREPROCESSOR_PATH: Path = MODELS_DIR / "preprocessor.joblib"
MODEL_PATH: Path = MODELS_DIR / "mlp.pt"
METADATA_PATH: Path = MODELS_DIR / "metadata.json"


class ArtifactsMissingError(RuntimeError):
    """Raised when ``models/`` is missing the trained artifacts."""


class ChurnPredictor:
    """Bundles the fitted preprocessor, MLP and decision threshold."""

    def __init__(
        self,
        preprocessor: Pipeline,
        model: ChurnMLP,
        threshold: float,
        model_version: str,
    ) -> None:
        self.preprocessor = preprocessor
        self.model = model
        self.threshold = threshold
        self.model_version = model_version

    @classmethod
    def from_artifacts(cls, models_dir: Path = MODELS_DIR) -> ChurnPredictor:
        for path in (PREPROCESSOR_PATH, MODEL_PATH, METADATA_PATH):
            if not path.exists():
                raise ArtifactsMissingError(
                    f"Required artifact missing: {path}. Run `make train` first."
                )
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        checkpoint = torch.load(MODEL_PATH, weights_only=True)
        metadata = json.loads(METADATA_PATH.read_text())

        model = ChurnMLP(
            input_dim=checkpoint["input_dim"],
            hidden_dims=tuple(checkpoint["hidden_dims"]),
            dropout=checkpoint["dropout"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        return cls(
            preprocessor=preprocessor,
            model=model,
            threshold=float(metadata["production_threshold"]),
            model_version=str(metadata.get("model", "ChurnMLP")),
        )

    def predict_dataframe(self, df: pd.DataFrame) -> tuple[list[float], list[bool]]:
        """Return ``(probabilities, decisions)`` for every row in the dataframe."""
        X = self.preprocessor.transform(df)
        probas = predict_proba(self.model, X)
        decisions = (probas >= self.threshold).tolist()
        return probas.tolist(), decisions
