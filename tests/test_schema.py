"""Pandera schema test for the raw Telco Churn dataframe."""

from __future__ import annotations

import os

import pytest

from churn.config import RAW_DATA_DIR
from churn.dataset.loader import LOCAL_FILENAME, load_raw_dataset
from churn.dataset.schema import raw_schema


def _data_available() -> bool:
    return (RAW_DATA_DIR / LOCAL_FILENAME).exists() or bool(
        os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    )


@pytest.mark.skipif(not _data_available(), reason="raw dataset not available")
def test_raw_dataset_matches_schema() -> None:
    df = load_raw_dataset()
    raw_schema.validate(df, lazy=True)
