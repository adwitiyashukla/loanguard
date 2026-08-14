from __future__ import annotations
import numpy as np
import pandas as pd

def _safe_qcut(series: pd.Series, q: int=10) -> pd.Series:
    s = pd.to_numeric(series, errors='coerce')
    try:
        binned = pd.qcut(s, q=q, labels=False, duplicates='drop')
        return binned.astype('Int64')
    except (ValueError, IndexError):
        return pd.Series([pd.NA] * len(s), index=s.index, dtype='Int64')

def build_behavioral_features(df: pd.DataFrame, today: pd.Timestamp | None=None) -> pd.DataFrame:
    df = df.copy()
    today = today or df.get('issue_d', pd.Series([pd.Timestamp.today()])).max()
    if {'loan_amnt', 'annual_inc'}.issubset(df.columns):
        df['loan_to_income'] = df['loan_amnt'] / df['annual_inc'].replace(0, np.nan)
    if {'installment', 'annual_inc'}.issubset(df.columns):
        df['installment_to_income'] = df['installment'] * 12 / df['annual_inc'].replace(0, np.nan)
    if {'revol_bal', 'annual_inc'}.issubset(df.columns):
        df['revol_bal_to_income'] = df['revol_bal'] / df['annual_inc'].replace(0, np.nan)
    if {'earliest_cr_line', 'issue_d'}.issubset(df.columns):
        df['credit_history_years'] = ((df['issue_d'] - df['earliest_cr_line']).dt.days / 365.25).clip(lower=0)
    if {'open_acc', 'total_acc'}.issubset(df.columns):
        df['pct_open_acc'] = df['open_acc'] / df['total_acc'].replace(0, np.nan)
    neg_cols = [c for c in ('delinq_2yrs', 'pub_rec', 'pub_rec_bankruptcies') if c in df.columns]
    if neg_cols:
        df['n_negative_events'] = df[neg_cols].fillna(0).sum(axis=1)
    if 'dti' in df.columns:
        df['flag_high_dti'] = (df['dti'] > 35).astype(int)
    if 'emp_length' in df.columns:
        df['flag_no_employment'] = df['emp_length'].fillna(0).eq(0).astype(int)
    if 'annual_inc' in df.columns:
        df['income_bucket'] = _safe_qcut(df['annual_inc'], q=10)
    if 'loan_amnt' in df.columns:
        df['loan_bucket'] = _safe_qcut(df['loan_amnt'], q=10)
    if 'issue_d' in df.columns:
        df['app_month'] = df['issue_d'].dt.month
        df['app_quarter'] = df['issue_d'].dt.quarter
        df['app_year'] = df['issue_d'].dt.year
    for col in ('title', 'emp_title'):
        if col in df.columns:
            s = df[col].fillna('').astype(str)
            df[f'{col}_len'] = s.str.len()
            df[f'{col}_digit_ratio'] = s.apply(lambda x: sum((c.isdigit() for c in x)) / max(len(x), 1))
    return df
