"""Optuna hyperparameter tuning for XGBoost.

Use as a stand-alone tool to find good params, then drop them into
config.yaml.

    python -m src.training.tuner --trials 50
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    import optuna  # type: ignore
    _OPTUNA = True
except ImportError:  # pragma: no cover
    optuna = None  # type: ignore
    _OPTUNA = False

from ..models import XGBFraudModel
from ..evaluation import binary_classification_metrics
from ..utils.logging import get_logger

log = get_logger(__name__)


def optuna_tune_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 50,
    timeout: int | None = None,
) -> dict[str, Any]:
    """Run Optuna search and return best params + best PR-AUC."""
    if not _OPTUNA:
        raise ImportError("optuna is not installed")

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        }
        model = XGBFraudModel(params=params, early_stopping_rounds=30)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        proba = model.predict_proba(X_val)
        return binary_classification_metrics(y_val, proba)["pr_auc"]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    log.info(f"Best PR-AUC: {study.best_value:.4f}")
    log.info(f"Best params: {study.best_params}")
    return {"best_params": study.best_params, "best_value": float(study.best_value)}
