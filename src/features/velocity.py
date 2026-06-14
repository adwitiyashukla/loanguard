"""Velocity features — count of applications in a rolling window.

Fraud rings hit lenders in bursts. A genuine applicant from a zip
might appear once every few months; a synthetic-identity ring will
push 50 applications from the same zip in two days. Velocity
features catch this.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

log = get_logger(__name__)


def build_velocity_features(
    df: pd.DataFrame,
    date_col: str = "issue_d",
    keys: tuple[str, ...] = ("zip_code", "emp_title", "addr_state"),
    windows_days: tuple[int, ...] = (7, 30, 90),
) -> pd.DataFrame:
    """Add velocity counts per key and window.

    For each (key, window) pair, compute:
      - number of applications with the same key value in the
        previous `window` days

    These are leakage-safe because we only look backwards.
    """
    if date_col not in df.columns:
        log.warning(f"{date_col} not in df — skipping velocity features")
        return df.copy()

    df = df.copy()
    # Ensure datetime, then sort. Unparsed dates -> NaT -> sorted first.
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values(date_col).reset_index(drop=True)
    # Nanoseconds since epoch; NaT becomes a sentinel but never crashes.
    df["_t"] = df[date_col].astype("int64")

    for key in keys:
        if key not in df.columns:
            continue
        for w in windows_days:
            new_col = f"vel_{key}_{w}d"
            df[new_col] = 0

        # Process per-group with searchsorted for O(N log N) overall.
        for grp_value, idx in df.groupby(key, observed=True).groups.items():
            sub_idx = np.array(idx)
            t = df.loc[sub_idx, "_t"].to_numpy()
            order = np.argsort(t)
            t_sorted = t[order]
            sorted_idx = sub_idx[order]
            for w in windows_days:
                window_ns = w * 24 * 3600 * 1_000_000_000
                # for each i, count of j < i with t_sorted[j] >= t_sorted[i] - window
                lower_bounds = t_sorted - window_ns
                left_positions = np.searchsorted(t_sorted, lower_bounds, side="left")
                indices = np.arange(len(t_sorted))
                counts = indices - left_positions  # excludes self
                df.loc[sorted_idx, f"vel_{key}_{w}d"] = counts.astype(int)

    df = df.drop(columns=["_t"])
    return df
