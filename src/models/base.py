"""Abstract base class for all fraud models.

Every model implements the same minimal interface so they can be
swapped, ensembled, and tracked uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class FraudModel(ABC):
    """Common interface for supervised / unsupervised fraud models."""

    name: str = "base"

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None, **kwargs) -> "FraudModel":
        """Fit the model. For unsupervised models y may be None."""

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return a 1-D array of fraud probabilities in [0, 1]."""

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_params(self) -> dict[str, Any]:
        return {}

    @property
    def trained(self) -> bool:
        return getattr(self, "_trained", False)
