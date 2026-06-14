"""LightGBM fraud model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import lightgbm as lgb

from ..utils.logging import get_logger
from .base import FraudModel

log = get_logger(__name__)


class LGBMFraudModel(FraudModel):
    name = "lightgbm"

    DEFAULT_PARAMS = {
        "objective": "binary",
        "metric": "average_precision",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "n_estimators": 800,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_data_in_leaf": 50,
        "lambda_l2": 1.0,
        "is_unbalance": True,
        "verbosity": -1,
        "n_jobs": -1,
        "random_state": 42,
    }

    def __init__(self, params: dict | None = None, early_stopping_rounds: int = 50):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.early_stopping_rounds = early_stopping_rounds
        self.model: lgb.LGBMClassifier | None = None
        self._trained = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        eval_set: list[tuple[pd.DataFrame, pd.Series]] | None = None,
        **kwargs,
    ) -> "LGBMFraudModel":
        if y is None:
            raise ValueError("LGBMFraudModel requires labels")
        self.model = lgb.LGBMClassifier(**self.params)

        callbacks: list[Any] = []
        if eval_set is not None:
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=False))
            callbacks.append(lgb.log_evaluation(0))

        self.model.fit(
            X,
            y,
            eval_set=eval_set,
            callbacks=callbacks if callbacks else None,
        )
        self._trained = True
        best_iter = getattr(self.model, "best_iteration_", None)
        log.info(f"[LGBM] trained. best_iter={best_iter}")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self._trained or self.model is None:
            raise RuntimeError("Model not trained")
        return self.model.predict_proba(X)[:, 1]

    def get_params(self) -> dict[str, Any]:
        return {"params": self.params}

    def feature_importance(self) -> dict[str, float]:
        if not self._trained or self.model is None:
            return {}
        return dict(zip(self.model.feature_name_, self.model.feature_importances_))
