"""Shared fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add repo root to path so `import src` works
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import LendingClubLoader, build_fraud_labels, LabelConfig  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    """Small synthetic LendingClub-like dataframe for tests."""
    df = LendingClubLoader._synthetic(n=5000, seed=123)
    df = build_fraud_labels(df, LabelConfig())
    return df


@pytest.fixture
def small_df() -> pd.DataFrame:
    """Tiny dataframe for unit-level tests."""
    df = LendingClubLoader._synthetic(n=500, seed=7)
    return build_fraud_labels(df, LabelConfig())
