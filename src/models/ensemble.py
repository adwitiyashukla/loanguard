from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from ..utils.logging import get_logger
from .base import FraudModel
log = get_logger(__name__)

class StackingFraudEnsemble(FraudModel):
    name = 'stacking_ensemble'

    def __init__(self, base_models: list[FraudModel], n_folds: int=5, meta_C: float=1.0, calibration: str='isotonic', random_seed: int=42):
        self.base_models = base_models
        self.n_folds = n_folds
        self.meta_C = meta_C
        self.calibration = calibration
        self.random_seed = random_seed
        self.meta_model: LogisticRegression | None = None
        self.calibrator_: IsotonicRegression | None = None
        self._trained = False

    def fit(self, X: pd.DataFrame, y: pd.Series | None=None, X_val: pd.DataFrame | None=None, y_val: pd.Series | None=None, **kwargs) -> 'StackingFraudEnsemble':
        if y is None:
            raise ValueError('StackingFraudEnsemble requires labels')
        log.info(f'Building OOF predictions across {len(self.base_models)} models...')
        oof = np.zeros((len(X), len(self.base_models)))
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_seed)
        for (fold, (tr_idx, va_idx)) in enumerate(skf.split(X, y)):
            log.info(f'  fold {fold + 1}/{self.n_folds}')
            (X_tr, X_va) = (X.iloc[tr_idx], X.iloc[va_idx])
            y_tr = y.iloc[tr_idx]
            for (m_idx, m) in enumerate(self.base_models):
                cls = type(m)
                params = m.get_params()
                fitted = cls(**params) if params else cls()
                fitted.fit(X_tr, y_tr)
                oof[va_idx, m_idx] = fitted.predict_proba(X_va)
        for m in self.base_models:
            if not getattr(m, '_trained', False):
                m.fit(X, y)
        self.meta_model = LogisticRegression(C=self.meta_C, max_iter=500)
        self.meta_model.fit(oof, y)
        log.info(f'Meta-learner coefficients: {dict(zip([m.name for m in self.base_models], self.meta_model.coef_[0].round(3)))}')
        meta_proba_oof = self.meta_model.predict_proba(oof)[:, 1]
        if self.calibration == 'isotonic':
            self.calibrator_ = IsotonicRegression(out_of_bounds='clip')
            if X_val is not None and y_val is not None:
                X_meta_val = np.column_stack([m.predict_proba(X_val) for m in self.base_models])
                meta_val = self.meta_model.predict_proba(X_meta_val)[:, 1]
                self.calibrator_.fit(meta_val, y_val)
            else:
                self.calibrator_.fit(meta_proba_oof, y)
        self._trained = True
        log.info('Ensemble trained.')
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self._trained or self.meta_model is None:
            raise RuntimeError('Ensemble not trained')
        meta_X = np.column_stack([m.predict_proba(X) for m in self.base_models])
        raw = self.meta_model.predict_proba(meta_X)[:, 1]
        if self.calibrator_ is not None:
            return self.calibrator_.predict(raw)
        return raw

    def base_probas(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({m.name: m.predict_proba(X) for m in self.base_models})

    def get_params(self) -> dict[str, Any]:
        return {'n_folds': self.n_folds, 'meta_C': self.meta_C, 'calibration': self.calibration, 'base_models': [m.name for m in self.base_models]}
