"""Download the raw Telco Customer Churn dataset and load it as a DataFrame."""

from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub
import pandas as pd

from churn.config import KAGGLE_DATASET_SLUG, RAW_DATA_DIR, ensure_directories
from churn.logging_config import get_logger

logger = get_logger(__name__)

LOCAL_FILENAME: str = "telco_churn.csv"


def download_raw_dataset(*, force: bool = False) -> Path:
    """Fetch the raw CSV from Kaggle and cache it under ``data/raw/``.

    Parameters
    ----------
    force:
        Re-download even if a cached copy is present.

    Returns
    -------
    Path
        Absolute path to the cached CSV inside the repository.
    """
    ensure_directories()
    target = RAW_DATA_DIR / LOCAL_FILENAME

    if target.exists() and not force:
        logger.info("Dataset already cached", extra={"path": str(target)})
        return target

    logger.info("Downloading dataset from Kaggle", extra={"slug": KAGGLE_DATASET_SLUG})
    download_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET_SLUG))

    csv_files = sorted(download_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV file found in kagglehub download directory: {download_dir}"
        )
    if len(csv_files) > 1:
        logger.warning(
            "Multiple CSVs found in download; using the first",
            extra={"candidates": [f.name for f in csv_files]},
        )

    source_csv = csv_files[0]
    shutil.copy2(source_csv, target)
    logger.info(
        "Cached raw dataset",
        extra={"target": str(target), "source_name": source_csv.name},
    )
    return target


def load_raw_dataset() -> pd.DataFrame:
    """Return the raw Telco Churn dataset as a pandas DataFrame.

    Calls :func:`download_raw_dataset` first so the file is guaranteed present.
    """
    path = download_raw_dataset()
    df = pd.read_csv(path)
    logger.info(
        "Loaded raw dataset",
        extra={"path": str(path), "rows": df.shape[0], "cols": df.shape[1]},
    )
    return df
