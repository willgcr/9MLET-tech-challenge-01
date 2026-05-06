"""Sklearn preprocessing pipeline for the Telco Churn dataset."""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.config import ID_COLUMN, TARGET_COLUMN

NUMERIC_FEATURES: list[str] = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizen",
]

CATEGORICAL_FEATURES: list[str] = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "tenure_bin",
]

TENURE_BINS: int = 4


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Coerce ``TotalCharges`` to float and add a discretised ``tenure_bin``."""

    def fit(self, X: pd.DataFrame, y=None) -> FeatureEngineer:  # noqa: ARG002
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
        X["tenure_bin"] = pd.cut(X["tenure"], bins=TENURE_BINS).astype(str)
        return X


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(X, y)`` with X stripped of ID/target and y mapped to 0/1."""
    y = (df[TARGET_COLUMN] == "Yes").astype(int)
    X = df.drop(columns=[ID_COLUMN, TARGET_COLUMN])
    return X, y


def build_preprocessor() -> Pipeline:
    """Return the full preprocessing pipeline as a single sklearn estimator."""
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    column_transform = ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        [
            ("feature_engineer", FeatureEngineer()),
            ("transform", column_transform),
        ]
    )
