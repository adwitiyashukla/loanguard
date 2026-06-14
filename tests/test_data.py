"""Tests for data loading, validation and labelling."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data import LendingClubLoader, build_fraud_labels, LabelConfig
from src.data.splitter import stratified_split, time_based_split


def test_synthetic_loader_shape():
    df = LendingClubLoader._synthetic(n=1000, seed=1)
    assert len(df) == 1000
    assert "loan_amnt" in df.columns
    assert df["loan_amnt"].between(500, 50_000).all()


def test_synthetic_loader_no_nan_in_target_fields():
    df = LendingClubLoader._synthetic(n=500, seed=2)
    # Critical fields used downstream cannot be all-NaN
    for col in ("loan_amnt", "annual_inc", "grade", "issue_d"):
        assert df[col].notna().any()


def test_fraud_labels_positive_rate_in_range(synthetic_df):
    rate = synthetic_df["is_fraud"].mean()
    # Defensible bounds for a synthetic 5k sample
    assert 0.0 < rate < 0.5


def test_fraud_labels_columns_present(synthetic_df):
    for col in ("rule_fpd", "rule_income_anomaly", "rule_debt_inconsist",
                "rule_address_ring", "n_anomalies", "is_fraud"):
        assert col in synthetic_df.columns


def test_time_split_chronological(synthetic_df):
    train, val, test = time_based_split(synthetic_df, "issue_d", 0.15, 0.15)
    assert len(train) + len(val) + len(test) == len(synthetic_df)
    assert train["issue_d"].max() <= val["issue_d"].max()
    assert val["issue_d"].max() <= test["issue_d"].max()


def test_stratified_split_preserves_rate(synthetic_df):
    train, val, test = stratified_split(synthetic_df, "is_fraud", 0.15, 0.15, random_seed=1)
    rates = [d["is_fraud"].mean() for d in (train, val, test)]
    assert max(rates) - min(rates) < 0.02  # within 2pp


def test_label_config_from_dict():
    cfg = LabelConfig.from_dict(
        {"rules": {"first_payment_default": {"enabled": False, "max_days_to_default": 60}}}
    )
    assert cfg.fpd_enabled is False
    assert cfg.fpd_max_days == 60
