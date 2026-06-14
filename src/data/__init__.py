"""Data ingestion, validation, labelling, splitting."""

from .loader import LendingClubLoader, load_raw
from .validation import validate_schema, LoanSchema
from .labels import build_fraud_labels, LabelConfig
from .splitter import time_based_split, stratified_split

__all__ = [
    "LendingClubLoader",
    "load_raw",
    "validate_schema",
    "LoanSchema",
    "build_fraud_labels",
    "LabelConfig",
    "time_based_split",
    "stratified_split",
]
