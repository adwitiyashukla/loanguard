"""Weak-supervision fraud labelling.

LendingClub doesn't ship explicit 'fraud' labels — nor does any real
lender during their first year of operations. This module implements a
defensible weak-supervision policy that combines four signals:

  1. First-payment default (FPD): defaulted within first 90 days
  2. Income / DTI inconsistency: dti > 60 AND annual_inc > 500k
  3. Income anomaly: annual_inc > 99.5th percentile with low emp_length
  4. Address ring anomaly: zip_code shared by >N high-default applicants

A loan is labelled fraud if it triggers (FPD) OR (>=2 anomaly rules).

This is exactly the bootstrapping a real lender would use in week 1
while a labelled fraud panel is being assembled by the ops team.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class LabelConfig:
    fpd_enabled: bool = True
    fpd_max_days: int = 90

    income_anomaly_enabled: bool = True
    income_quantile: float = 0.995

    debt_inconsistency_enabled: bool = True
    dti_threshold: float = 60.0
    income_threshold: float = 500_000.0

    address_anomaly_enabled: bool = True
    shared_zip_threshold: int = 25

    # If a row has >= n_anomaly_rules anomalies, label as fraud
    n_anomaly_rules_for_fraud: int = 2

    @classmethod
    def from_dict(cls, d: dict) -> "LabelConfig":
        rules = d.get("rules", {})
        return cls(
            fpd_enabled=rules.get("first_payment_default", {}).get("enabled", True),
            fpd_max_days=rules.get("first_payment_default", {}).get("max_days_to_default", 90),
            income_anomaly_enabled=rules.get("income_anomaly", {}).get("enabled", True),
            income_quantile=rules.get("income_anomaly", {}).get("annual_inc_quantile", 0.995),
            debt_inconsistency_enabled=rules.get("debt_inconsistency", {}).get("enabled", True),
            dti_threshold=rules.get("debt_inconsistency", {}).get("dti_threshold", 60),
            income_threshold=rules.get("debt_inconsistency", {}).get("annual_inc_threshold", 500_000),
            address_anomaly_enabled=rules.get("address_anomaly", {}).get("enabled", True),
            shared_zip_threshold=rules.get("address_anomaly", {}).get("shared_zip_threshold", 25),
        )


def build_fraud_labels(df: pd.DataFrame, cfg: LabelConfig | None = None) -> pd.DataFrame:
    """Apply weak-supervision rules and return df + label cols.

    Adds:
      - `rule_fpd`              : first-payment default
      - `rule_income_anomaly`   : income outlier
      - `rule_debt_inconsist`   : dti/income inconsistency
      - `rule_address_ring`     : shared-zip ring
      - `n_anomalies`           : sum of anomaly rules (excluding FPD)
      - `is_fraud`              : final label
    """
    cfg = cfg or LabelConfig()
    df = df.copy()

    # ---------- Rule 1: First-payment default ---------- #
    if cfg.fpd_enabled and {"loan_status", "issue_d", "last_pymnt_d"}.issubset(df.columns):
        defaulted = df["loan_status"].isin(["Charged Off", "Default"])
        # If last_pymnt_d is null/equal to issue_d, treat as first-payment default
        delta = (df["last_pymnt_d"] - df["issue_d"]).dt.days
        early = (delta.isna()) | (delta <= cfg.fpd_max_days)
        df["rule_fpd"] = (defaulted & early).astype(int)
    else:
        df["rule_fpd"] = 0

    # ---------- Rule 2: Income anomaly ---------- #
    if cfg.income_anomaly_enabled and "annual_inc" in df.columns:
        cutoff = df["annual_inc"].quantile(cfg.income_quantile)
        low_emp = df["emp_length"].fillna(0) <= 1 if "emp_length" in df.columns else False
        df["rule_income_anomaly"] = ((df["annual_inc"] >= cutoff) & low_emp).astype(int)
    else:
        df["rule_income_anomaly"] = 0

    # ---------- Rule 3: Debt / income inconsistency ---------- #
    if cfg.debt_inconsistency_enabled and {"dti", "annual_inc"}.issubset(df.columns):
        df["rule_debt_inconsist"] = (
            (df["dti"] > cfg.dti_threshold) & (df["annual_inc"] > cfg.income_threshold)
        ).astype(int)
    else:
        df["rule_debt_inconsist"] = 0

    # ---------- Rule 4: Address-ring anomaly ---------- #
    if cfg.address_anomaly_enabled and {"zip_code", "loan_status"}.issubset(df.columns):
        defaulted_by_zip = (
            df.assign(_def=df["loan_status"].isin(["Charged Off", "Default"]).astype(int))
            .groupby("zip_code")["_def"]
            .sum()
        )
        ring_zips = set(defaulted_by_zip[defaulted_by_zip >= cfg.shared_zip_threshold].index)
        df["rule_address_ring"] = df["zip_code"].isin(ring_zips).astype(int)
    else:
        df["rule_address_ring"] = 0

    # ---------- Combine ---------- #
    anomaly_cols = ["rule_income_anomaly", "rule_debt_inconsist", "rule_address_ring"]
    df["n_anomalies"] = df[anomaly_cols].sum(axis=1)

    df["is_fraud"] = (
        (df["rule_fpd"] == 1)
        | (df["n_anomalies"] >= cfg.n_anomaly_rules_for_fraud)
    ).astype(int)

    rate = df["is_fraud"].mean()
    log.info(
        f"Fraud labelling complete — positive rate {rate:.2%} "
        f"(FPD={df['rule_fpd'].mean():.2%}, "
        f"anomaly>=2={(df['n_anomalies'] >= 2).mean():.2%})"
    )

    return df
