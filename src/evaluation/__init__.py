"""Evaluation: metrics, calibration, explainability, business value."""

from .metrics import (
    binary_classification_metrics,
    ks_statistic,
    gini,
    recall_at_fpr,
    lift_at_k,
    psi,
)
from .business import (
    expected_loss_avoidance,
    optimal_threshold,
    cost_sensitive_evaluation,
)
from .explainability import ShapExplainer
from .fairness import fairness_report

__all__ = [
    "binary_classification_metrics",
    "ks_statistic",
    "gini",
    "recall_at_fpr",
    "lift_at_k",
    "psi",
    "expected_loss_avoidance",
    "optimal_threshold",
    "cost_sensitive_evaluation",
    "ShapExplainer",
    "fairness_report",
]
