"""Isolation Forest — unsupervised anomaly scorer.

Used as an extra signal for the stacking ensemble. Particularly
valuable on novel fraud patterns that the supervised models
haven't seen.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from ..utils.logging import get_logger
from .base import FraudModel

log = get_logger(__name__)


class IsolationForestFraudModel(FraudModel):
    name = "isolation_forest"

    DEFAULT_PARAMS = {
        "n_estimators": 200,
        "contamination": 0.02,
        "max_samples": 0.5,
        "n_jobs": -1,
        "random_state": 42,
    }

    def __init__(self, params: dict | None = None):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.scaler = StandardScaler()
        self.model: IsolationForest | None = None
        self.score_min_: float = 0.0
        self.score_max_: float = 1.0
        self._trained = False

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None, **kwargs) -> "IsolationForestFraudModel":
        X_scaled = self.scaler.fit_transform(X.values)
        self.model = IsolationForest(**self.params)
        self.model.fit(X_scaled)
        # Score the training set to establish a normalisation range.
        raw = -self.model.score_samples(X_scaled)  # higher = more anomalous
        self.score_min_ = float(np.percentile(raw, 1))
        self.score_max_ = float(np.percentile(raw, 99))
        self._trained = True
        log.info(
            f"[IF] trained. score range [{self.score_min_:.3f}, {self.score_max_:.3f}]"
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self._trained or self.model is None:
            raise RuntimeError("Model not trained")
        X_scaled = self.scaler.transform(X.values)
        raw = -self.model.score_samples(X_scaled)
        span = max(self.score_max_ - self.score_min_, 1e-9)
        prob = (raw - self.score_min_) / span
        return np.clip(prob, 0.0, 1.0)

    def get_params(self) -> dict[str, Any]:
        return {"params": self.params}
