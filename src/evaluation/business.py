"""Cost-sensitive (business) evaluation.

A fraud model's AUC is interesting; its expected net loss avoidance is
what the business will be measured on. This module converts model
outputs into rupees using a cost matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CostMatrix:
    false_negative: float = 100_000.0  # missed fraud — average loss
    false_positive: float = 1_500.0    # cost of manual review for a clean app
    true_positive: float = 0.0          # caught fraud — model wins
    true_negative: float = 0.0


def cost_sensitive_evaluation(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost: CostMatrix | None = None,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Sweep thresholds and report expected loss per applicant at each."""
    cost = cost or CostMatrix()
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    rows = []
    n = len(y_true)
    n_pos = max(y_true.sum(), 1)
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        total_cost = (
            fn * cost.false_negative
            + fp * cost.false_positive
            + tp * cost.true_positive
            + tn * cost.true_negative
        )
        rows.append({
            "threshold": float(t),
            "review_rate": float(pred.mean()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "recall": tp / n_pos,
            "precision": tp / max(tp + fp, 1),
            "total_cost": float(total_cost),
            "cost_per_applicant": float(total_cost / n),
        })
    return pd.DataFrame(rows)


def optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost: CostMatrix | None = None,
) -> tuple[float, float]:
    """Return (threshold, expected cost) that minimises total cost."""
    df = cost_sensitive_evaluation(y_true, y_proba, cost)
    best = df.loc[df["total_cost"].idxmin()]
    return float(best["threshold"]), float(best["total_cost"])


def expected_loss_avoidance(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    avg_fraud_loss: float = 100_000.0,
    review_cost: float = 1_500.0,
    baseline_recall: float = 0.0,
) -> dict[str, float]:
    """Compare model decisions against a do-nothing baseline.

    Returns net loss avoided in rupees.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    pred = (y_proba >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())

    losses_caught = tp * avg_fraud_loss
    losses_caught_baseline = baseline_recall * y_true.sum() * avg_fraud_loss
    review_cost_total = (tp + fp) * review_cost
    net_savings = (losses_caught - losses_caught_baseline) - review_cost_total

    return {
        "frauds_caught": tp,
        "false_alarms": fp,
        "gross_savings": float(losses_caught - losses_caught_baseline),
        "review_cost": float(review_cost_total),
        "net_savings": float(net_savings),
    }
