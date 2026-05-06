"""Shared pytest fixtures and global test configuration."""

from __future__ import annotations

import random

import numpy as np
import pytest

from churn.config import RANDOM_SEED


@pytest.fixture(autouse=True)
def _seed_rngs() -> None:
    """Seed Python and NumPy RNGs before every test for deterministic output."""
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
