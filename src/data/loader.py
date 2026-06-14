"""LendingClub loan data loader.

The LendingClub dataset is the only public dataset of comparable size
and richness to a real lender's book. We treat its 'accepted loans' CSV
(2007–2018, ~2.2M rows) as a proxy for an SME / unsecured retail book.

If the raw CSV isn't available locally, the loader falls back to a
deterministic synthetic generator that produces statistically similar
data — used in CI and for quickstart demos.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

log = get_logger(__name__)


# Columns we actually use downstream. The full CSV has 150+ columns,
# most of which are post-funding (and therefore data-leakage if used
# for application-time fraud detection).
APPLICATION_TIME_COLUMNS: list[str] = [
    "id",
    "issue_d",
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_title",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "purpose",
    "title",
    "zip_code",
    "addr_state",
    "dti",
    "delinq_2yrs",
    "earliest_cr_line",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "mort_acc",
    "pub_rec_bankruptcies",
    # Outcome columns — used only for label construction, then dropped:
    "loan_status",
    "last_pymnt_d",
]


@dataclass
class LendingClubLoader:
    """Loader for LendingClub accepted-loans CSV."""

    raw_path: str | Path
    sample_size: int | None = None
    random_seed: int = 42

    def load(self) -> pd.DataFrame:
        raw_path = Path(self.raw_path)
        if not raw_path.exists():
            log.warning(
                f"Raw file not found at {raw_path} — falling back to synthetic generator. "
                "Run scripts/download_data.py for real data."
            )
            return self._synthetic(n=self.sample_size or 50_000, seed=self.random_seed)

        log.info(f"Loading LendingClub data from {raw_path}")
        df = pd.read_csv(
            raw_path,
            usecols=lambda c: c in APPLICATION_TIME_COLUMNS,
            low_memory=False,
        )
        log.info(f"Loaded {len(df):,} rows, {df.shape[1]} columns")

        if self.sample_size is not None and self.sample_size < len(df):
            df = df.sample(n=self.sample_size, random_state=self.random_seed).reset_index(drop=True)
            log.info(f"Sampled down to {len(df):,} rows")

        df = self._normalise(df)
        return df

    # ------------------------------------------------------------------ #
    # Cleaning
    # ------------------------------------------------------------------ #

    def _normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cast types, strip percent signs, parse dates."""
        df = df.copy()

        if "int_rate" in df.columns and df["int_rate"].dtype == object:
            df["int_rate"] = (
                df["int_rate"].astype(str).str.rstrip("%").replace("nan", np.nan).astype(float)
            )
        if "revol_util" in df.columns and df["revol_util"].dtype == object:
            df["revol_util"] = (
                df["revol_util"].astype(str).str.rstrip("%").replace("nan", np.nan).astype(float)
            )
        if "term" in df.columns and df["term"].dtype == object:
            df["term"] = (
                df["term"].astype(str).str.extract(r"(\d+)")[0].astype(float)
            )

        # Date parsing
        for date_col in ("issue_d", "last_pymnt_d", "earliest_cr_line"):
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col], format="%b-%Y", errors="coerce")

        # emp_length: '< 1 year' -> 0, '10+ years' -> 10, '4 years' -> 4
        if "emp_length" in df.columns:
            df["emp_length"] = df["emp_length"].map(self._parse_emp_length)

        # Trim whitespace on string cols
        for c in df.select_dtypes(include="object").columns:
            df[c] = df[c].astype(str).str.strip()

        return df

    @staticmethod
    def _parse_emp_length(val: str | float) -> float:
        if pd.isna(val):
            return np.nan
        s = str(val).strip().lower()
        if "<" in s:
            return 0.0
        if "10" in s:
            return 10.0
        digits = "".join(c for c in s if c.isdigit())
        return float(digits) if digits else np.nan

    # ------------------------------------------------------------------ #
    # Synthetic fallback — used for CI and quick demos
    # ------------------------------------------------------------------ #

    @staticmethod
    def _synthetic(n: int = 50_000, seed: int = 42) -> pd.DataFrame:
        """Generate a realistic-looking synthetic LendingClub dataset.

        Distributions are calibrated to roughly match the real data so
        that EDA / unit tests behave the same way.
        """
        rng = np.random.default_rng(seed)

        grades = ["A", "B", "C", "D", "E", "F", "G"]
        grade_probs = [0.15, 0.27, 0.27, 0.18, 0.08, 0.04, 0.01]
        grade = rng.choice(grades, size=n, p=grade_probs)

        sub_grade = np.array([f"{g}{rng.integers(1, 6)}" for g in grade])

        # Risk increases with grade index
        grade_idx = np.array([grades.index(g) for g in grade])

        loan_amnt = np.clip(rng.normal(15000, 9000, n), 1000, 40000).round(-2)
        term = rng.choice([36, 60], size=n, p=[0.72, 0.28]).astype(float)
        int_rate = np.clip(5 + grade_idx * 2.5 + rng.normal(0, 1.5, n), 5, 30)
        installment = (loan_amnt * (int_rate / 1200)) / (1 - (1 + int_rate / 1200) ** -term)

        annual_inc = np.clip(rng.lognormal(mean=11.0, sigma=0.6, size=n), 8000, 1_500_000).round(-2)
        dti = np.clip(rng.normal(18, 9, n), 0, 60)

        emp_length = rng.choice(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            size=n,
            p=[0.07, 0.06, 0.07, 0.06, 0.06, 0.06, 0.05, 0.04, 0.04, 0.03, 0.46],
        ).astype(float)

        home_ownership = rng.choice(
            ["RENT", "MORTGAGE", "OWN", "OTHER"], size=n, p=[0.40, 0.50, 0.09, 0.01]
        )
        verification_status = rng.choice(
            ["Verified", "Source Verified", "Not Verified"], size=n, p=[0.34, 0.34, 0.32]
        )
        purpose = rng.choice(
            [
                "debt_consolidation", "credit_card", "home_improvement",
                "other", "major_purchase", "small_business", "car",
                "medical", "moving", "vacation", "house",
            ],
            size=n,
            p=[0.50, 0.23, 0.06, 0.06, 0.04, 0.03, 0.02, 0.02, 0.02, 0.01, 0.01],
        )

        delinq_2yrs = rng.poisson(0.3 + grade_idx * 0.15, n)
        inq_last_6mths = rng.poisson(0.6 + grade_idx * 0.1, n)
        open_acc = rng.poisson(10, n)
        pub_rec = rng.binomial(1, 0.05, n)
        revol_bal = np.clip(rng.lognormal(8.5, 1.2, n), 0, 1_500_000).round(-1)
        revol_util = np.clip(rng.normal(55, 25, n), 0, 150)
        total_acc = open_acc + rng.poisson(15, n)
        mort_acc = rng.poisson(1.5, n)
        pub_rec_bankruptcies = rng.binomial(1, 0.03, n)

        states = [
            "CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA", "NC", "MI",
            "NJ", "VA", "WA", "MA", "AZ", "TN", "IN", "MO", "MD", "WI",
        ]
        addr_state = rng.choice(states, size=n)
        zip_code = np.array([f"{rng.integers(100, 999):03d}xx" for _ in range(n)])

        # Issue date over 4 years
        days_offset = rng.integers(0, 365 * 4, size=n)
        issue_d = pd.to_datetime("2015-01-01") + pd.to_timedelta(days_offset, unit="D")
        # Make it month-resolution like the real data
        issue_d = issue_d.to_period("M").to_timestamp()

        earliest_cr_line = issue_d - pd.to_timedelta(
            rng.integers(365 * 3, 365 * 25, size=n), unit="D"
        )

        # Loan status: charge-off rate increases with grade
        co_prob = 0.05 + grade_idx * 0.045
        is_co = rng.random(n) < co_prob
        loan_status = np.where(is_co, "Charged Off", "Fully Paid")

        # last_pymnt_d: a few months after issue for charged-off
        months_to_co = np.where(is_co, rng.integers(1, 20, n), rng.integers(20, 60, n))
        last_pymnt_d = issue_d + pd.to_timedelta(months_to_co * 30, unit="D")

        emp_title = rng.choice(
            ["Manager", "Engineer", "Teacher", "Sales", "Driver", "Nurse", "Analyst",
             "Owner", "Director", "Consultant", "Clerk", "Technician"],
            size=n,
        )
        title = rng.choice(
            ["Debt consolidation", "Credit card refinancing", "Home improvement",
             "Major purchase", "Business", "Medical expenses"],
            size=n,
        )

        df = pd.DataFrame({
            "id": np.arange(1_000_000, 1_000_000 + n),
            "issue_d": issue_d,
            "loan_amnt": loan_amnt,
            "term": term,
            "int_rate": int_rate.round(2),
            "installment": installment.round(2),
            "grade": grade,
            "sub_grade": sub_grade,
            "emp_title": emp_title,
            "emp_length": emp_length,
            "home_ownership": home_ownership,
            "annual_inc": annual_inc,
            "verification_status": verification_status,
            "purpose": purpose,
            "title": title,
            "zip_code": zip_code,
            "addr_state": addr_state,
            "dti": dti.round(2),
            "delinq_2yrs": delinq_2yrs,
            "earliest_cr_line": earliest_cr_line,
            "inq_last_6mths": inq_last_6mths,
            "open_acc": open_acc,
            "pub_rec": pub_rec,
            "revol_bal": revol_bal,
            "revol_util": revol_util.round(1),
            "total_acc": total_acc,
            "mort_acc": mort_acc,
            "pub_rec_bankruptcies": pub_rec_bankruptcies,
            "loan_status": loan_status,
            "last_pymnt_d": last_pymnt_d,
        })

        # Inject some realistic NaNs
        for col, frac in [
            ("emp_length", 0.05),
            ("emp_title", 0.06),
            ("title", 0.10),
            ("revol_util", 0.005),
            ("dti", 0.001),
        ]:
            mask = rng.random(n) < frac
            df.loc[mask, col] = np.nan

        log.info(f"Generated synthetic LendingClub-like dataset: {len(df):,} rows")
        return df


def load_raw(
    raw_path: str | Path = "data/raw/accepted_2007_to_2018Q4.csv",
    sample_size: int | None = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Convenience function — instantiate loader and return DataFrame."""
    return LendingClubLoader(
        raw_path=raw_path, sample_size=sample_size, random_seed=random_seed
    ).load()
