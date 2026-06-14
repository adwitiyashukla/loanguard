"""High-cardinality categorical encoders.

WoE (Weight of Evidence) is the credit industry standard — it's
monotonic, interpretable, and plays well with logistic regression
and tree-based models alike. Target encoding with smoothing is
a strong alternative for high-cardinality columns like emp_title.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class WoEEncoder(BaseEstimator, TransformerMixin):
    """Weight of Evidence encoder.

    For each category c of a feature:
        WoE_c = ln( (good_c / total_good) / (bad_c / total_bad) )

    Where 'bad' = positive class (fraud). A small smoothing constant
    is added to avoid log(0).
    """

    smoothing: float = 0.5
    unknown_value: float = 0.0
    mapping_: dict = field(default_factory=dict)
    iv_: dict = field(default_factory=dict)  # information value per column

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WoEEncoder":
        y = pd.Series(y).reset_index(drop=True)
        total_good = (1 - y).sum() + self.smoothing
        total_bad = y.sum() + self.smoothing

        for col in X.columns:
            df = pd.DataFrame({"_x": X[col].fillna("__NA__").reset_index(drop=True), "_y": y})
            grouped = df.groupby("_x")["_y"].agg(["sum", "count"])
            grouped["bad"] = grouped["sum"] + self.smoothing
            grouped["good"] = (grouped["count"] - grouped["sum"]) + self.smoothing
            grouped["woe"] = np.log((grouped["good"] / total_good) / (grouped["bad"] / total_bad))
            self.mapping_[col] = grouped["woe"].to_dict()
            # Information Value
            grouped["iv_part"] = (
                (grouped["good"] / total_good) - (grouped["bad"] / total_bad)
            ) * grouped["woe"]
            self.iv_[col] = float(grouped["iv_part"].sum())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for col in X.columns:
            mapping = self.mapping_.get(col, {})
            vals = X[col].fillna("__NA__")
            out[f"{col}_woe"] = vals.map(mapping).fillna(self.unknown_value)
        return out

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


@dataclass
class TargetEncoder(BaseEstimator, TransformerMixin):
    """Smoothed target (mean) encoder.

    Useful for very high-cardinality columns (employer name, zip).
    Smoothing prevents over-fitting on rare categories.
    """

    smoothing: float = 20.0
    unknown_value: float = 0.0
    global_mean_: float = 0.0
    mapping_: dict = field(default_factory=dict)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TargetEncoder":
        y = pd.Series(y).reset_index(drop=True)
        self.global_mean_ = float(y.mean())
        for col in X.columns:
            df = pd.DataFrame({"_x": X[col].fillna("__NA__").reset_index(drop=True), "_y": y})
            agg = df.groupby("_x")["_y"].agg(["mean", "count"])
            smoothed = (
                agg["mean"] * agg["count"] + self.global_mean_ * self.smoothing
            ) / (agg["count"] + self.smoothing)
            self.mapping_[col] = smoothed.to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for col in X.columns:
            mapping = self.mapping_.get(col, {})
            out[f"{col}_te"] = X[col].fillna("__NA__").map(mapping).fillna(self.global_mean_)
        return out

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
