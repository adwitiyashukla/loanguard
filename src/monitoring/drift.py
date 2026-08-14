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
    reference_: dict[str, np.ndarray] = field(default_factory=dict)
    psi_alert: float = 0.2

    def fit(self, X_ref: pd.DataFrame, sample_size: int=50000) -> 'DriftMonitor':
        if len(X_ref) > sample_size:
            X_ref = X_ref.sample(sample_size, random_state=42)
        for col in X_ref.columns:
            self.reference_[col] = X_ref[col].dropna().to_numpy()
        return self

    def psi_report(self, X_now: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for (col, ref) in self.reference_.items():
            if col not in X_now.columns:
                continue
            actual = X_now[col].dropna().to_numpy()
            if len(actual) < 10:
                continue
            val = psi(ref, actual)
            rows.append({'feature': col, 'psi': val, 'alert': val > self.psi_alert})
        return pd.DataFrame(rows).sort_values('psi', ascending=False).reset_index(drop=True)

    def evidently_report(self, X_now: pd.DataFrame, out_path: str | Path) -> Path:
        try:
            from evidently.report import Report
            from evidently.metric_preset import DataDriftPreset
        except ImportError as exc:
            raise ImportError('evidently not installed') from exc
        ref_df = pd.DataFrame({c: v[:min(len(v), len(X_now))] for (c, v) in self.reference_.items()})
        common = list(set(ref_df.columns) & set(X_now.columns))
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_df[common], current_data=X_now[common])
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(out_path))
        log.info(f'Evidently drift report saved to {out_path}')
        return out_path
