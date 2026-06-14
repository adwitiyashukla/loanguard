"""Group-wise fairness metrics.

We don't have protected attributes (gender/caste/religion) in the
LendingClub data. But we can still check fairness across legitimate
proxies like addr_state and home_ownership — and the same machinery
is what you'd use in production with an actual protected-class column.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def fairness_report(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    groups: pd.Series,
    threshold: float = 0.5,
    min_group_size: int = 200,
) -> pd.DataFrame:
    """Per-group performance breakdown."""
    df = pd.DataFrame({"y": y_true, "p": y_proba, "g": groups.values})
    rows = []
    overall_auc = roc_auc_score(y_true, y_proba)
    overall_fpr = float(((y_proba >= threshold) & (y_true == 0)).sum() / max((y_true == 0).sum(), 1))

    for g, sub in df.groupby("g"):
        if len(sub) < min_group_size or sub["y"].nunique() < 2:
            continue
        pred = (sub["p"] >= threshold).astype(int)
        tp = int(((pred == 1) & (sub["y"] == 1)).sum())
        fp = int(((pred == 1) & (sub["y"] == 0)).sum())
        fn = int(((pred == 0) & (sub["y"] == 1)).sum())
        tn = int(((pred == 0) & (sub["y"] == 0)).sum())

        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        auc = roc_auc_score(sub["y"], sub["p"])
        rows.append({
            "group": g,
            "n": len(sub),
            "positive_rate": float(sub["y"].mean()),
            "auc": float(auc),
            "auc_gap": float(auc - overall_auc),
            "tpr": float(tpr),
            "fpr": float(fpr),
            "fpr_gap": float(fpr - overall_fpr),
            "approval_rate": float((pred == 0).mean()),
        })
    return pd.DataFrame(rows).sort_values("auc_gap").reset_index(drop=True)
