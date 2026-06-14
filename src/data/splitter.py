"""Time-aware and stratified data splitters.

Out-of-time validation is essential for credit / fraud models — random
splits hugely overestimate generalisation because borrower behaviour
drifts and macro conditions change. We default to time-based splits
and use stratified random only for unit tests.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..utils.logging import get_logger

log = get_logger(__name__)


def time_based_split(
    df: pd.DataFrame,
    date_col: str = "issue_d",
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sort by date, then chronologically slice into train/val/test.

    Older loans go to train, the next chunk to val, the latest to test.
    This mirrors how a credit model actually gets deployed.
    """
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not in DataFrame")

    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    n = len(df_sorted)
    test_start = int(n * (1 - test_size))
    val_start = int(n * (1 - test_size - val_size))

    train = df_sorted.iloc[:val_start].reset_index(drop=True)
    val = df_sorted.iloc[val_start:test_start].reset_index(drop=True)
    test = df_sorted.iloc[test_start:].reset_index(drop=True)

    def _span(d: pd.DataFrame) -> str:
        if d.empty or d[date_col].isna().all():
            return "empty"
        return f"{d[date_col].min().date()} → {d[date_col].max().date()}"

    log.info(
        f"Time-split: train={len(train):,} ({_span(train)}), "
        f"val={len(val):,} ({_span(val)}), test={len(test):,} ({_span(test)})"
    )
    return train, val, test


def stratified_split(
    df: pd.DataFrame,
    target_col: str = "is_fraud",
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified random split — fallback when timestamps aren't available."""
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[target_col], random_state=random_seed
    )
    relative_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val,
        stratify=train_val[target_col],
        random_state=random_seed,
    )

    log.info(
        f"Stratified split: train={len(train):,}, val={len(val):,}, test={len(test):,} | "
        f"fraud rate train={train[target_col].mean():.2%}, "
        f"val={val[target_col].mean():.2%}, test={test[target_col].mean():.2%}"
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )
