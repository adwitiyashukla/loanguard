from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from ..utils.logging import get_logger
from .behavioral import build_behavioral_features
from .velocity import build_velocity_features
from .graph_features import build_graph_features
from .encoders import WoEEncoder, TargetEncoder
log = get_logger(__name__)
NUMERIC_BASE = ['loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'mort_acc', 'pub_rec_bankruptcies', 'emp_length', 'term']
NUMERIC_DERIVED = ['loan_to_income', 'installment_to_income', 'revol_bal_to_income', 'credit_history_years', 'pct_open_acc', 'n_negative_events', 'flag_high_dti', 'flag_no_employment', 'app_month', 'app_quarter', 'app_year', 'title_len', 'title_digit_ratio', 'emp_title_len', 'emp_title_digit_ratio', 'graph_component_size', 'graph_degree', 'graph_isolated']
CATEGORICAL_LOW = ['grade', 'home_ownership', 'verification_status', 'purpose', 'addr_state']
CATEGORICAL_HIGH = ['sub_grade', 'emp_title', 'zip_code']

@dataclass
class FeatureBuilder(BaseEstimator, TransformerMixin):
    winsorize_quantiles: tuple[float, float] = (0.005, 0.995)
    use_velocity: bool = True
    use_graph: bool = True
    fitted_: bool = False
    winsor_bounds_: dict = field(default_factory=dict)
    imputer_: Optional[SimpleImputer] = None
    scaler_: Optional[StandardScaler] = None
    woe_encoder_: Optional[WoEEncoder] = None
    target_encoder_: Optional[TargetEncoder] = None
    feature_names_: list[str] = field(default_factory=list)
    numeric_cols_: list[str] = field(default_factory=list)
    categorical_low_cols_: list[str] = field(default_factory=list)
    categorical_high_cols_: list[str] = field(default_factory=list)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FeatureBuilder':
        log.info(f'FeatureBuilder.fit on {len(X):,} rows')
        X_eng = self._engineer(X)
        self.numeric_cols_ = [c for c in NUMERIC_BASE + NUMERIC_DERIVED if c in X_eng.columns]
        self.categorical_low_cols_ = [c for c in CATEGORICAL_LOW if c in X_eng.columns]
        self.categorical_high_cols_ = [c for c in CATEGORICAL_HIGH if c in X_eng.columns]
        X_num = X_eng[self.numeric_cols_].apply(pd.to_numeric, errors='coerce')
        (lo_q, hi_q) = self.winsorize_quantiles
        self.winsor_bounds_ = {col: (X_num[col].quantile(lo_q), X_num[col].quantile(hi_q)) for col in self.numeric_cols_}
        X_num = self._apply_winsor(X_num)
        self.imputer_ = SimpleImputer(strategy='median')
        X_num_imp = pd.DataFrame(self.imputer_.fit_transform(X_num), columns=self.numeric_cols_, index=X.index)
        self.scaler_ = StandardScaler()
        self.scaler_.fit(X_num_imp)
        if self.categorical_low_cols_:
            self.woe_encoder_ = WoEEncoder()
            self.woe_encoder_.fit(X_eng[self.categorical_low_cols_], y)
            log.info(f'WoE IV (low-card): ' + ', '.join((f'{k}={v:.3f}' for (k, v) in self.woe_encoder_.iv_.items())))
        if self.categorical_high_cols_:
            self.target_encoder_ = TargetEncoder()
            self.target_encoder_.fit(X_eng[self.categorical_high_cols_], y)
        feat_names = list(self.numeric_cols_)
        feat_names += [f'{c}_woe' for c in self.categorical_low_cols_]
        feat_names += [f'{c}_te' for c in self.categorical_high_cols_]
        self.feature_names_ = feat_names
        self.fitted_ = True
        log.info(f'FeatureBuilder ready - {len(feat_names)} features')
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError('FeatureBuilder not fitted')
        X_eng = self._engineer(X)
        X_num = X_eng.reindex(columns=self.numeric_cols_).apply(pd.to_numeric, errors='coerce')
        X_num = self._apply_winsor(X_num)
        X_num_imp = pd.DataFrame(self.imputer_.transform(X_num), columns=self.numeric_cols_, index=X.index)
        frames = [X_num_imp]
        if self.woe_encoder_ is not None:
            frames.append(self.woe_encoder_.transform(X_eng[self.categorical_low_cols_]))
        if self.target_encoder_ is not None:
            frames.append(self.target_encoder_.transform(X_eng[self.categorical_high_cols_]))
        out = pd.concat(frames, axis=1)
        out = out.reindex(columns=self.feature_names_, fill_value=0.0)
        out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        return out

    def fit_transform(self, X: pd.DataFrame, y: pd.Series=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def _engineer(self, X: pd.DataFrame) -> pd.DataFrame:
        original_index = X.index
        out = X.copy()
        out['__row_id__'] = np.arange(len(out))
        out = build_behavioral_features(out)
        if self.use_velocity:
            out = build_velocity_features(out)
        if self.use_graph:
            out = build_graph_features(out)
        out = out.sort_values('__row_id__').drop(columns='__row_id__')
        out.index = original_index
        return out

    def _apply_winsor(self, X_num: pd.DataFrame) -> pd.DataFrame:
        for (col, (lo, hi)) in self.winsor_bounds_.items():
            if col in X_num.columns and pd.notna(lo) and pd.notna(hi):
                X_num[col] = X_num[col].clip(lower=lo, upper=hi)
        return X_num
