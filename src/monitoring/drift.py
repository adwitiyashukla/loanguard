"""Population Stability Index drift monitor.

Stores a reference distribution per feature at training time and
compares production traffic against it.

Evidently is used for the rich HTML report; the lightweight PSI
calculation lives in evaluation.metrics so the API can compute it
hot-path without spinning up Evidently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..evaluation.metrics import psi
from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class DriftMonitor:
    """Captures reference distributions; compares actual at scoring time."""

    reference_: dict[str, np.ndarray] = field(default_factory=dict)
    psi_alert: float = 0.2

    def fit(self, X_ref: pd.DataFrame, sample_size: int = 50_000) -> "DriftMonitor":
        if len(X_ref) > sample_size:
            X_ref = X_ref.sample(sample_size, random_state=42)
        for col in X_ref.columns:
            self.reference_[col] = X_ref[col].dropna().to_numpy()
        return self

    def psi_report(self, X_now: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for col, ref in self.reference_.items():
            if col not in X_now.columns:
                continue
            actual = X_now[col].dropna().to_numpy()
            if len(actual) < 10:
                continue
            val = psi(ref, actual)
            rows.append({"feature": col, "psi": val, "alert": val > self.psi_alert})
        return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)

    def evidently_report(self, X_now: pd.DataFrame, out_path: str | Path) -> Path:
        """Generate an Evidently HTML report for the ops team."""
        try:
            from evidently.report import Report  # type: ignore
            from evidently.metric_preset import DataDriftPreset  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError("evidently not installed") from exc

        ref_df = pd.DataFrame({c: v[: min(len(v), len(X_now))] for c, v in self.reference_.items()})
        common = list(set(ref_df.columns) & set(X_now.columns))
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_df[common], current_data=X_now[common])
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(out_path))
        log.info(f"Evidently drift report saved to {out_path}")
        return out_path
