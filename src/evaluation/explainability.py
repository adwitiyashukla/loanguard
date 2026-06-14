"""SHAP-based global and local explanations.

For tree models we use TreeExplainer (fast, exact). For autoencoders
and isolation forests we fall back to KernelExplainer on a small
background sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

try:
    import shap  # type: ignore
    _SHAP_OK = True
except ImportError:  # pragma: no cover
    shap = None  # type: ignore
    _SHAP_OK = False

from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class Explanation:
    feature: str
    value: float
    contribution: float
    direction: str  # "↑ fraud" or "↓ fraud"


class ShapExplainer:
    """Wraps SHAP for both global and per-application explanations."""

    def __init__(self, model: Any, X_background: pd.DataFrame | None = None):
        if not _SHAP_OK:
            raise ImportError("shap is not installed")
        self.model = model
        self.X_background = X_background
        self.explainer_: Any = None

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #

    def fit(self) -> "ShapExplainer":
        # Detect which kind of explainer to use
        cls_name = type(self.model).__name__
        try:
            # TreeExplainer works on XGB / LGBM / CatBoost via duck-typing
            inner = getattr(self.model, "model", self.model)
            self.explainer_ = shap.TreeExplainer(inner)
            log.info(f"[SHAP] using TreeExplainer for {cls_name}")
        except Exception:  # pragma: no cover
            bg = self.X_background.sample(min(100, len(self.X_background)))
            self.explainer_ = shap.KernelExplainer(
                lambda X: self.model.predict_proba(pd.DataFrame(X, columns=bg.columns)),
                bg,
            )
            log.info(f"[SHAP] using KernelExplainer for {cls_name}")
        return self

    # ------------------------------------------------------------------ #
    # Global importance
    # ------------------------------------------------------------------ #

    def global_importance(self, X: pd.DataFrame, max_display: int = 20) -> pd.DataFrame:
        sv = self.explainer_.shap_values(X)
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        mean_abs = np.abs(sv).mean(axis=0)
        return (
            pd.DataFrame({"feature": X.columns, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .head(max_display)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------ #
    # Local explanation — used by the API for adverse-action notice
    # ------------------------------------------------------------------ #

    def explain_one(self, x: pd.DataFrame, top_k: int = 5) -> list[Explanation]:
        if len(x) != 1:
            raise ValueError("explain_one expects a single-row DataFrame")
        sv = self.explainer_.shap_values(x)
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        sv = np.asarray(sv).reshape(-1)
        order = np.argsort(-np.abs(sv))[:top_k]

        results: list[Explanation] = []
        for i in order:
            feat = x.columns[i]
            val = float(x.iloc[0, i])
            contrib = float(sv[i])
            direction = "↑ fraud" if contrib > 0 else "↓ fraud"
            results.append(Explanation(feat, val, contrib, direction))
        return results
