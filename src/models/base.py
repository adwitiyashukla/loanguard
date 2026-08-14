from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import numpy as np
import pandas as pd

class FraudModel(ABC):
    name: str = 'base'

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series | None=None, **kwargs) -> 'FraudModel':
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        pass

    def predict(self, X: pd.DataFrame, threshold: float=0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_params(self) -> dict[str, Any]:
        return {}

    @property
    def trained(self) -> bool:
        return getattr(self, '_trained', False)
