from __future__ import annotations
import numpy as np
import pandas as pd
from ..utils.logging import get_logger
log = get_logger(__name__)

def build_velocity_features(df: pd.DataFrame, date_col: str='issue_d', keys: tuple[str, ...]=('zip_code', 'emp_title', 'addr_state'), windows_days: tuple[int, ...]=(7, 30, 90)) -> pd.DataFrame:
    if date_col not in df.columns:
        log.warning(f'{date_col} not in df - skipping velocity features')
        return df.copy()
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.sort_values(date_col).reset_index(drop=True)
    df['_t'] = df[date_col].astype('int64')
    for key in keys:
        if key not in df.columns:
            continue
        for w in windows_days:
            new_col = f'vel_{key}_{w}d'
            df[new_col] = 0
        for (grp_value, idx) in df.groupby(key, observed=True).groups.items():
            sub_idx = np.array(idx)
            t = df.loc[sub_idx, '_t'].to_numpy()
            order = np.argsort(t)
            t_sorted = t[order]
            sorted_idx = sub_idx[order]
            for w in windows_days:
                window_ns = w * 24 * 3600 * 1000000000
                lower_bounds = t_sorted - window_ns
                left_positions = np.searchsorted(t_sorted, lower_bounds, side='left')
                indices = np.arange(len(t_sorted))
                counts = indices - left_positions
                df.loc[sorted_idx, f'vel_{key}_{w}d'] = counts.astype(int)
    df = df.drop(columns=['_t'])
    return df
