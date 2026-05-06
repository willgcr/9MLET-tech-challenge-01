"""Smoke tests - the package imports and core utilities behave."""

from __future__ import annotations

import re

import churn
from churn import config
from churn.logging_config import configure_logging, get_logger


def test_package_exposes_version() -> None:
    """Package version is a valid PEP 440 string."""
    assert hasattr(churn, "__version__")
    assert re.fullmatch(r"\d+\.\d+\.\d+", churn.__version__) is not None


def test_config_paths_resolve_under_repo_root() -> None:
    """All path constants resolve to subdirectories of the repo root."""
    repo_root = config.REPO_ROOT.resolve()
    for path in (
        config.DATA_DIR,
        config.RAW_DATA_DIR,
        config.PROCESSED_DATA_DIR,
        config.MODELS_DIR,
    ):
        assert path.resolve().is_relative_to(repo_root)


def test_logging_can_be_configured() -> None:
    """configure_logging is idempotent and returns a usable logger."""
    configure_logging()
    configure_logging()  # second call must not raise
    logger = get_logger("churn.tests.smoke")
    logger.info("smoke test log")
    assert logger.isEnabledFor(20)  # INFO == 20
