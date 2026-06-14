"""Tests for fraud models — make sure each fits and predicts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import FeatureBuilder
from src.models import (
    XGBFraudModel,
    LGBMFraudModel,
    IsolationForestFraudModel,
)


@pytest.fixture
def small_features(small_df):
    y = small_df["is_fraud"]
    drop = ["is_fraud", "rule_fpd", "rule_income_anomaly", "rule_debt_inconsist",
            "rule_address_ring", "n_anomalies", "loan_status", "last_pymnt_d"]
    X = small_df.drop(columns=drop)
    builder = FeatureBuilder(use_velocity=False, use_graph=False)
    X_eng = builder.fit_transform(X, y)
    return X_eng, y


def test_xgb_fit_and_predict(small_features):
    X, y = small_features
    m = XGBFraudModel(params={"n_estimators": 30, "max_depth": 3})
    m.fit(X, y)
    proba = m.predict_proba(X)
    assert proba.shape == (len(X),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_lgbm_fit_and_predict(small_features):
    X, y = small_features
    m = LGBMFraudModel(params={"n_estimators": 30, "num_leaves": 7})
    m.fit(X, y)
    proba = m.predict_proba(X)
    assert proba.shape == (len(X),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_isolation_forest_fit_and_predict(small_features):
    X, _ = small_features
    m = IsolationForestFraudModel(params={"n_estimators": 50, "contamination": 0.05})
    m.fit(X)
    proba = m.predict_proba(X)
    assert proba.shape == (len(X),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_model_raises_if_not_trained():
    m = XGBFraudModel()
    with pytest.raises(RuntimeError):
        m.predict_proba(pd.DataFrame({"x": [1, 2]}))
