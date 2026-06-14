"""Scoring service — wraps the trained model + builder + explainer."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..utils.io import load_joblib
from ..utils.logging import get_logger
from .schemas import LoanApplication, ReasonCode, ScoreResponse

log = get_logger(__name__)


class ScoringService:
    """Singleton-style scoring service used by the API endpoints."""

    def __init__(
        self,
        artifacts_dir: str | Path = "artifacts",
        review_threshold: float = 0.30,
        decline_threshold: float = 0.70,
        model_version: str = "0.1.0",
    ):
        self.artifacts_dir = Path(artifacts_dir)
        self.review_threshold = review_threshold
        self.decline_threshold = decline_threshold
        self.model_version = model_version

        self.builder: Any = None
        self.model: Any = None
        self.explainer: Any = None
        self.loaded_at: float | None = None

    # ------------------------------------------------------------------ #
    def load(self) -> None:
        if not self.artifacts_dir.exists():
            log.warning(
                f"Artifacts dir {self.artifacts_dir} not found. Service starts empty — "
                "call /reload after training."
            )
            return

        builder_path = self.artifacts_dir / "feature_builder.joblib"
        if builder_path.exists():
            self.builder = load_joblib(builder_path)
        ensemble_path = self.artifacts_dir / "model_ensemble.joblib"
        xgb_path = self.artifacts_dir / "model_xgboost.joblib"
        if ensemble_path.exists():
            self.model = load_joblib(ensemble_path)
        elif xgb_path.exists():
            self.model = load_joblib(xgb_path)

        # Build explainer lazily once we have something to explain
        if self.model is not None and hasattr(self.model, "base_models"):
            # Use XGB / LGBM from the ensemble for SHAP
            for m in self.model.base_models:  # type: ignore
                if m.name in ("xgboost", "lightgbm"):
                    try:
                        from ..evaluation.explainability import ShapExplainer
                        self.explainer = ShapExplainer(m).fit()
                        log.info(f"SHAP explainer built from base model: {m.name}")
                        break
                    except Exception as exc:  # pragma: no cover
                        log.warning(f"Could not init SHAP explainer: {exc}")
        elif self.model is not None:
            try:
                from ..evaluation.explainability import ShapExplainer
                self.explainer = ShapExplainer(self.model).fit()
            except Exception as exc:  # pragma: no cover
                log.warning(f"Could not init SHAP explainer: {exc}")

        self.loaded_at = time.time()
        log.info(f"Scoring service ready. version={self.model_version}")

    @property
    def is_ready(self) -> bool:
        return self.builder is not None and self.model is not None

    # ------------------------------------------------------------------ #
    def score_one(self, app: LoanApplication) -> ScoreResponse:
        if not self.is_ready:
            raise RuntimeError("ScoringService is not loaded")

        df = pd.DataFrame([app.model_dump()])
        # Coerce dates
        for c in ("issue_d", "earliest_cr_line"):
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
        if "issue_d" not in df.columns or df["issue_d"].isna().all():
            # tz-naive "now" — issue_d only drives seasonality/velocity features
            df["issue_d"] = pd.Timestamp.now()

        X = self.builder.transform(df)
        proba = float(self.model.predict_proba(X)[0])

        if proba >= self.decline_threshold:
            decision = "DECLINE"
        elif proba >= self.review_threshold:
            decision = "REVIEW"
        else:
            decision = "APPROVE"

        # Reason codes
        reasons: list[ReasonCode] = []
        if self.explainer is not None:
            try:
                for exp in self.explainer.explain_one(X.iloc[[0]], top_k=5):
                    reasons.append(
                        ReasonCode(
                            feature=exp.feature,
                            value=float(exp.value),
                            contribution=float(exp.contribution),
                            direction=exp.direction,
                        )
                    )
            except Exception as exc:  # pragma: no cover
                log.warning(f"Could not produce reason codes: {exc}")

        return ScoreResponse(
            application_id=app.id,
            fraud_score=round(proba, 6),
            decision=decision,  # type: ignore[arg-type]
            threshold_review=self.review_threshold,
            threshold_decline=self.decline_threshold,
            reason_codes=reasons,
            model_version=self.model_version,
            scored_at=datetime.now(timezone.utc),
        )

    def score_many(self, apps: list[LoanApplication]) -> list[ScoreResponse]:
        if not self.is_ready:
            raise RuntimeError("ScoringService is not loaded")
        return [self.score_one(a) for a in apps]
