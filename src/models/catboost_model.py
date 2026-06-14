"""CatBoost fraud model — best out-of-the-box performance on categoricals."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    from catboost import CatBoostClassifier  # type: ignore

    _CATBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover
    CatBoostClassifier = None  # type: ignore
    _CATBOOST_AVAILABLE = False

from ..utils.logging import get_logger
from .base import FraudModel

log = get_logger(__name__)


class CatBoostFraudModel(FraudModel):
    name = "catboost"

    DEFAULT_PARAMS = {
        "loss_function": "Logloss",
        "eval_metric": "PRAUC",
        "iterations": 800,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3.0,
        "auto_class_weights": "Balanced",
        "random_seed": 42,
        "verbose": False,
        "allow_writing_files": False,
    }

    def __init__(self, params: dict | None = None, early_stopping_rounds: int = 50):
        if not _CATBOOST_AVAILABLE:
            raise ImportError(
                "catboost is not installed. Install with `pip install catboost`."
            )
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.early_stopping_rounds = early_stopping_rounds
        self.model: Any = None
        self._trained = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        eval_set: list[tuple[pd.DataFrame, pd.Series]] | None = None,
        **kwargs,
    ) -> "CatBoostFraudModel":
        if y is None:
            raise ValueError("CatBoostFraudModel requires labels")
        self.model = CatBoostClassifier(**self.params)
        self.model.fit(
            X,
            y,
            eval_set=eval_set[0] if eval_set else None,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose=False,
        )
        self._trained = True
        log.info(f"[CatBoost] trained. tree_count={self.model.tree_count_}")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self._trained or self.model is None:
            raise RuntimeError("Model not trained")
        return self.model.predict_proba(X)[:, 1]

    def get_params(self) -> dict[str, Any]:
        return {"params": self.params}
