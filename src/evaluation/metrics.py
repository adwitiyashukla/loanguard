from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve, log_loss

def binary_classification_metrics(y_true: np.ndarray, y_proba: np.ndarray, fpr_target: float=0.05) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    return {'roc_auc': float(roc_auc_score(y_true, y_proba)), 'pr_auc': float(average_precision_score(y_true, y_proba)), 'ks': float(ks_statistic(y_true, y_proba)), 'gini': float(gini(y_true, y_proba)), 'log_loss': float(log_loss(y_true, np.clip(y_proba, 1e-07, 1 - 1e-07))), 'brier': float(brier_score_loss(y_true, y_proba)), f'recall_at_fpr_{int(fpr_target * 100)}pct': float(recall_at_fpr(y_true, y_proba, fpr_target)), 'lift_top_5pct': float(lift_at_k(y_true, y_proba, 0.05)), 'lift_top_10pct': float(lift_at_k(y_true, y_proba, 0.1)), 'n': int(len(y_true)), 'positive_rate': float(y_true.mean())}

def ks_statistic(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    (fpr, tpr, _) = roc_curve(y_true, y_proba)
    return float(np.max(tpr - fpr))

def gini(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    return 2 * roc_auc_score(y_true, y_proba) - 1

def recall_at_fpr(y_true: np.ndarray, y_proba: np.ndarray, fpr_target: float) -> float:
    (fpr, tpr, _) = roc_curve(y_true, y_proba)
    if (fpr <= fpr_target).any():
        return float(tpr[fpr <= fpr_target].max())
    return 0.0

def lift_at_k(y_true: np.ndarray, y_proba: np.ndarray, top_k: float) -> float:
    n = len(y_true)
    n_top = max(int(n * top_k), 1)
    order = np.argsort(-y_proba)
    top = y_true[order[:n_top]]
    base = max(np.mean(y_true), 1e-09)
    return float(top.mean() / base)

def psi(expected: np.ndarray | pd.Series, actual: np.ndarray | pd.Series, bins: int=10) -> float:
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))
    (breakpoints[0], breakpoints[-1]) = (-np.inf, np.inf)
    (e_counts, _) = np.histogram(expected, breakpoints)
    (a_counts, _) = np.histogram(actual, breakpoints)
    e_pct = np.clip(e_counts / len(expected), 1e-06, None)
    a_pct = np.clip(a_counts / len(actual), 1e-06, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))
