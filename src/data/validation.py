"""Pandera schema validation.

Catch bad data at the boundary, not in the middle of a training run.
"""

from __future__ import annotations

import pandera as pa
from pandera.typing import Series

from ..utils.logging import get_logger

log = get_logger(__name__)


class LoanSchema(pa.DataFrameModel):
    """Schema for LendingClub application-time data after normalisation.

    We use ``Series[float]`` for nullable numeric columns to avoid the
    pandas int-with-NaN trap (nullable ints require Int64 dtype which
    is finicky across versions).
    """

    id: Series[int] = pa.Field(unique=True, coerce=True)
    loan_amnt: Series[float] = pa.Field(ge=500, le=50_000, nullable=True)
    term: Series[float] = pa.Field(isin=[36.0, 60.0], nullable=True)
    int_rate: Series[float] = pa.Field(ge=0, le=40, nullable=True)
    installment: Series[float] = pa.Field(ge=0, nullable=True)
    grade: Series[str] = pa.Field(isin=["A", "B", "C", "D", "E", "F", "G"], nullable=True)
    annual_inc: Series[float] = pa.Field(ge=0, le=10_000_000, nullable=True)
    dti: Series[float] = pa.Field(ge=-1, le=999, nullable=True)
    delinq_2yrs: Series[float] = pa.Field(ge=0, le=100, nullable=True)
    open_acc: Series[float] = pa.Field(ge=0, le=200, nullable=True)
    revol_bal: Series[float] = pa.Field(ge=0, nullable=True)
    revol_util: Series[float] = pa.Field(ge=0, le=300, nullable=True)
    pub_rec: Series[float] = pa.Field(ge=0, nullable=True)

    class Config:
        strict = False  # allow extra columns (loan_status, etc.)
        coerce = True


def validate_schema(df, schema: type[pa.DataFrameModel] = LoanSchema, lazy: bool = True):
    """Validate a DataFrame against a Pandera schema.

    Args:
        df: input DataFrame
        schema: Pandera DataFrameModel class
        lazy: if True, collect all errors before raising

    Returns:
        validated DataFrame (Pandera may coerce types)
    """
    try:
        validated = schema.validate(df, lazy=lazy)
        log.info(f"Schema validation passed for {len(df):,} rows")
        return validated
    except pa.errors.SchemaErrors as exc:
        log.error(f"Schema validation failed: {exc.failure_cases.head()}")
        raise
