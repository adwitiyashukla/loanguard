"""Pydantic request/response models."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, ConfigDict


class LoanApplication(BaseModel):
    """Inbound loan application payload."""

    model_config = ConfigDict(extra="ignore")

    id: int | None = None
    loan_amnt: float = Field(..., ge=500, le=50_000)
    term: Literal[36, 60]
    int_rate: float = Field(..., ge=0, le=40)
    installment: float = Field(..., ge=0)
    grade: Literal["A", "B", "C", "D", "E", "F", "G"]
    sub_grade: str
    emp_title: str | None = None
    emp_length: float | None = Field(None, ge=0, le=10)
    home_ownership: Literal["RENT", "MORTGAGE", "OWN", "OTHER", "ANY", "NONE"] = "RENT"
    annual_inc: float = Field(..., ge=0)
    verification_status: Literal["Verified", "Source Verified", "Not Verified"] = "Not Verified"
    purpose: str
    title: str | None = None
    zip_code: str
    addr_state: str
    dti: float = Field(..., ge=-1, le=999)
    delinq_2yrs: int = Field(0, ge=0)
    earliest_cr_line: date | None = None
    inq_last_6mths: int = Field(0, ge=0)
    open_acc: int = Field(0, ge=0)
    pub_rec: int = Field(0, ge=0)
    revol_bal: float = Field(0, ge=0)
    revol_util: float = Field(0, ge=0)
    total_acc: int = Field(0, ge=0)
    mort_acc: int = Field(0, ge=0)
    pub_rec_bankruptcies: int = Field(0, ge=0)
    issue_d: date | None = None


class ReasonCode(BaseModel):
    feature: str
    value: float
    contribution: float
    direction: str


class ScoreResponse(BaseModel):
    application_id: int | None
    fraud_score: float = Field(..., ge=0, le=1, description="Calibrated probability of fraud.")
    decision: Literal["APPROVE", "REVIEW", "DECLINE"]
    threshold_review: float
    threshold_decline: float
    reason_codes: list[ReasonCode] = Field(default_factory=list)
    model_version: str
    scored_at: datetime


class BatchScoreRequest(BaseModel):
    applications: list[LoanApplication]


class BatchScoreResponse(BaseModel):
    scored_at: datetime
    model_version: str
    results: list[ScoreResponse]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "down"]
    model_loaded: bool
    model_version: str | None
    uptime_seconds: float
