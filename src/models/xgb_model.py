"""XGBoost fraud model with monotonic constraints."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from ..utils.logging import get_logger
from .base import FraudModel

log = get_logger(__name__)


class XGBFraudModel(FraudModel):
    name = "xgboost"

    DEFAULT_PARAMS: dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 800,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    }

    def __init__(
        self,
        params: dict | None = None,
        monotonic_constraints: dict[str, int] | None = None,
        early_stopping_rounds: int = 50,
    ):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.monotonic_constraints = monotonic_constraints or {}
        self.early_stopping_rounds = early_stopping_rounds
        self.model: xgb.XGBClassifier | None = None
        self._trained = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        eval_set: list[tuple[pd.DataFrame, pd.Series]] | None = None,
        **kwargs,
    ) -> "XGBFraudModel":
        if y is None:
            raise ValueError("XGBFraudModel requires labels")

        # Build monotonic constraint vector aligned with X columns
        monotone = self._build_monotone_vector(list(X.columns))

        # Don't mutate self.params — make a working copy
        params = {**self.params}
        scale = params.pop("scale_pos_weight", None)
        if scale is None and y.mean() > 0:
            scale = (1 - y.mean()) / y.mean()

        self.model = xgb.XGBClassifier(
            **params,
            scale_pos_weight=scale,
            monotone_constraints=tuple(monotone) if any(monotone) else None,
            early_stopping_rounds=self.early_stopping_rounds if eval_set else None,
        )

        fit_kwargs: dict[str, Any] = {}
        if eval_set is not None:
            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["verbose"] = False

        self.model.fit(X, y, **fit_kwargs)
        self._trained = True
        best_iter = getattr(self.model, "best_iteration", None)
        log.info(f"[XGB] trained. best_iter={best_iter}")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self._trained or self.model is None:
            raise RuntimeError("Model not trained")
        return self.model.predict_proba(X)[:, 1]

    def _build_monotone_vector(self, columns: list[str]) -> list[int]:
        """Return list of -1/0/1 matched to feature order.

        Constraint keys can match suffixes (e.g. 'grade' matches
        'grade_woe'), which is what we want after WoE encoding.
        """
        vec = []
        for col in columns:
            constraint = 0
            for key, sign in self.monotonic_constraints.items():
                if col == key or col.startswith(f"{key}_"):
                    constraint = sign
                    break
            vec.append(constraint)
        return vec

    def get_params(self) -> dict[str, Any]:
        return {
            "params": self.params,
            "monotonic_constraints": self.monotonic_constraints,
        }

    def feature_importance(self, importance_type: str = "gain") -> dict[str, float]:
        if not self._trained or self.model is None:
            return {}
        booster = self.model.get_booster()
        return booster.get_score(importance_type=importance_type)
