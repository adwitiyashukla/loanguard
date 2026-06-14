"""Tests for the feature engineering pipeline."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features import (
    FeatureBuilder,
    WoEEncoder,
    TargetEncoder,
    build_behavioral_features,
    build_velocity_features,
)


def test_behavioral_features_add_columns(small_df):
    out = build_behavioral_features(small_df)
    for col in ("loan_to_income", "credit_history_years", "n_negative_events",
                "flag_high_dti", "app_month"):
        assert col in out.columns


def test_velocity_features_monotone(small_df):
    # 30-day window count should be >= 7-day window count
    out = build_velocity_features(small_df, keys=("zip_code",), windows_days=(7, 30, 90))
    if "vel_zip_code_7d" in out.columns and "vel_zip_code_30d" in out.columns:
        assert (out["vel_zip_code_30d"] >= out["vel_zip_code_7d"]).all()


def test_woe_encoder_fit_transform(small_df):
    enc = WoEEncoder()
    X = small_df[["grade", "home_ownership"]]
    y = small_df["is_fraud"]
    out = enc.fit_transform(X, y)
    assert out.shape == (len(X), 2)
    # IV should be non-negative and finite
    for col, iv in enc.iv_.items():
        assert iv >= 0
        assert iv == iv  # not NaN


def test_target_encoder_handles_unseen(small_df):
    enc = TargetEncoder()
    X = small_df[["addr_state"]]
    y = small_df["is_fraud"]
    enc.fit(X, y)
    new = pd.DataFrame({"addr_state": ["__UNKNOWN__"] * 5})
    out = enc.transform(new)
    # Should fall back to global mean
    assert (out["addr_state_te"] == enc.global_mean_).all()


def test_feature_builder_end_to_end(small_df):
    y = small_df["is_fraud"]
    X = small_df.drop(columns=["is_fraud", "rule_fpd", "rule_income_anomaly",
                                "rule_debt_inconsist", "rule_address_ring",
                                "n_anomalies", "loan_status", "last_pymnt_d"])
    builder = FeatureBuilder(use_velocity=False, use_graph=False)
    out_train = builder.fit_transform(X, y)
    out_inf = builder.transform(X.head(20))
    assert out_train.shape[1] == out_inf.shape[1] == len(builder.feature_names_)
    assert not out_train.isna().any().any()
