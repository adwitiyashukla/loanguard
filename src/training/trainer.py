"""End-to-end training orchestrator.

Drives:
  1. Data load → validate → label → split
  2. Feature build (fit on train, transform on val/test)
  3. Train each enabled model
  4. Build stacking ensemble
  5. Evaluate on test set, log to MLflow
  6. Persist artifacts

Run from CLI:
    python scripts/train.py --config config/config.yaml
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import mlflow  # type: ignore
    _MLFLOW = True
except ImportError:  # pragma: no cover
    mlflow = None  # type: ignore
    _MLFLOW = False

from ..data import LendingClubLoader, build_fraud_labels, LabelConfig
from ..data.splitter import time_based_split, stratified_split
from ..features import FeatureBuilder
from ..models import (
    XGBFraudModel,
    LGBMFraudModel,
    CatBoostFraudModel,
    IsolationForestFraudModel,
    AutoencoderFraudModel,
    StackingFraudEnsemble,
)
from ..evaluation import binary_classification_metrics
from ..evaluation.business import cost_sensitive_evaluation, CostMatrix
from ..utils.config import load_config
from ..utils.io import save_joblib, save_json, ensure_dir
from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class Trainer:
    config: dict

    # Fitted state
    feature_builder_: FeatureBuilder | None = None
    base_models_: dict | None = None
    ensemble_: StackingFraudEnsemble | None = None

    def run(self, mlflow_run_name: str | None = None) -> dict:
        cfg = self.config
        seed = cfg["project"]["random_seed"]
        np.random.seed(seed)

        # ---------- 1. Load ---------- #
        loader = LendingClubLoader(
            raw_path=Path(cfg["paths"]["data_raw"]) / cfg["data"]["raw_filename"],
            sample_size=cfg["data"].get("sample_size"),
            random_seed=seed,
        )
        df = loader.load()

        # ---------- 2. Label ---------- #
        label_cfg = LabelConfig.from_dict(cfg["labels"])
        df = build_fraud_labels(df, label_cfg)

        # ---------- 3. Split ---------- #
        if cfg["data"]["split_strategy"] == "time":
            train, val, test = time_based_split(
                df, cfg["data"]["date_col"],
                cfg["data"]["val_size"], cfg["data"]["test_size"],
            )
        else:
            train, val, test = stratified_split(
                df, cfg["data"]["target_col"],
                cfg["data"]["val_size"], cfg["data"]["test_size"],
                random_seed=seed,
            )

        y_train = train["is_fraud"]
        y_val = val["is_fraud"]
        y_test = test["is_fraud"]

        # Drop outcome columns from the features
        drop_cols = ["is_fraud", "rule_fpd", "rule_income_anomaly",
                     "rule_debt_inconsist", "rule_address_ring",
                     "n_anomalies", "loan_status", "last_pymnt_d", "id"]
        X_train_raw = train.drop(columns=[c for c in drop_cols if c in train.columns])
        X_val_raw = val.drop(columns=[c for c in drop_cols if c in val.columns])
        X_test_raw = test.drop(columns=[c for c in drop_cols if c in test.columns])

        # ---------- 4. Feature engineering ---------- #
        self.feature_builder_ = FeatureBuilder()
        X_train = self.feature_builder_.fit_transform(X_train_raw, y_train)
        X_val = self.feature_builder_.transform(X_val_raw)
        X_test = self.feature_builder_.transform(X_test_raw)

        # ---------- 5. Train base models ---------- #
        self.base_models_ = {}
        mcfg = cfg["models"]
        if mcfg["xgboost"]["enabled"]:
            m = XGBFraudModel(
                params=mcfg["xgboost"]["params"],
                monotonic_constraints=mcfg["xgboost"].get("monotonic_constraints"),
            )
            m.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            self.base_models_["xgboost"] = m
        if mcfg["lightgbm"]["enabled"]:
            m = LGBMFraudModel(params=mcfg["lightgbm"]["params"])
            m.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            self.base_models_["lightgbm"] = m
        if mcfg["catboost"]["enabled"]:
            try:
                m = CatBoostFraudModel(params=mcfg["catboost"]["params"])
                m.fit(X_train, y_train, eval_set=[(X_val, y_val)])
                self.base_models_["catboost"] = m
            except ImportError:
                log.warning("CatBoost not installed — skipping.")
        if mcfg["isolation_forest"]["enabled"]:
            m = IsolationForestFraudModel(params=mcfg["isolation_forest"]["params"])
            m.fit(X_train)
            self.base_models_["isolation_forest"] = m
        if mcfg["autoencoder"]["enabled"]:
            m = AutoencoderFraudModel(params=mcfg["autoencoder"]["params"])
            m.fit(X_train, y_train)
            self.base_models_["autoencoder"] = m

        # ---------- 6. Stacking ensemble ---------- #
        if mcfg["ensemble"]["enabled"] and len(self.base_models_) >= 2:
            self.ensemble_ = StackingFraudEnsemble(
                base_models=list(self.base_models_.values()),
                n_folds=3,  # smaller for runtime; bump to 5 for full
                calibration=mcfg["ensemble"]["calibration"],
                random_seed=seed,
            )
            self.ensemble_.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        # ---------- 7. Evaluate on test ---------- #
        results: dict[str, Any] = {"per_model": {}, "ensemble": None}
        for name, m in self.base_models_.items():
            proba = m.predict_proba(X_test)
            results["per_model"][name] = binary_classification_metrics(y_test, proba)

        if self.ensemble_ is not None:
            proba = self.ensemble_.predict_proba(X_test)
            results["ensemble"] = binary_classification_metrics(y_test, proba)

            # Cost sweep
            cost_cfg = cfg["evaluation"]["cost_matrix"]
            cost = CostMatrix(
                false_negative=cost_cfg["false_negative"],
                false_positive=cost_cfg["false_positive"],
            )
            cost_df = cost_sensitive_evaluation(y_test, proba, cost=cost)
            best_t = cost_df.loc[cost_df["total_cost"].idxmin()]
            results["best_threshold"] = float(best_t["threshold"])
            results["expected_cost_per_applicant"] = float(best_t["cost_per_applicant"])

        # ---------- 8. Persist + log ---------- #
        artifacts_dir = ensure_dir(Path(cfg["paths"]["artifacts"]))
        save_joblib(self.feature_builder_, artifacts_dir / "feature_builder.joblib")
        for name, m in self.base_models_.items():
            save_joblib(m, artifacts_dir / f"model_{name}.joblib")
        if self.ensemble_ is not None:
            save_joblib(self.ensemble_, artifacts_dir / "model_ensemble.joblib")
        save_json(results, artifacts_dir / "results.json")

        # Save test-set predictions so the dashboard's Portfolio view works.
        if self.ensemble_ is not None:
            test_proba = self.ensemble_.predict_proba(X_test)
        else:
            first_model = next(iter(self.base_models_.values()))
            test_proba = first_model.predict_proba(X_test)
        pd.DataFrame({"y_true": y_test.to_numpy(), "proba": test_proba}).to_csv(
            artifacts_dir / "test_predictions.csv", index=False
        )

        # Fit + save a drift monitor on the training features (reference set).
        try:
            from ..monitoring.drift import DriftMonitor

            monitor = DriftMonitor(
                psi_alert=cfg["monitoring"]["drift"]["psi_alert_threshold"]
            ).fit(X_train)
            save_joblib(monitor, artifacts_dir / "drift_monitor.joblib")
        except Exception as exc:  # pragma: no cover
            log.warning(f"Could not build drift monitor: {exc}")

        if _MLFLOW and mlflow is not None:
            # MLflow logging is optional — never let it crash training.
            # Newer MLflow versions deprecated the file-store backend and
            # raise on use; we opt in via env var, and still wrap in
            # try/except so any other MLflow problem can't lose artifacts.
            import os as _os
            _os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
            try:
                self._log_mlflow(results, cfg, mlflow_run_name)
            except Exception as exc:
                log.warning(f"MLflow logging skipped (non-fatal): {exc}")

        log.info("Training pipeline complete.")
        log.info(f"Ensemble metrics: {results['ensemble']}")
        return results

    # ------------------------------------------------------------------ #
    def _log_mlflow(self, results: dict, cfg: dict, run_name: str | None) -> None:
        mlflow.set_tracking_uri(cfg["paths"]["mlflow_uri"])
        mlflow.set_experiment(cfg["project"]["name"])
        with mlflow.start_run(run_name=run_name or "training-run"):
            # Flatten + log params
            mlflow.log_params({"random_seed": cfg["project"]["random_seed"]})
            mlflow.log_param("sample_size", cfg["data"].get("sample_size"))
            mlflow.log_param("split_strategy", cfg["data"]["split_strategy"])

            # Log per-model metrics
            for name, metrics in results["per_model"].items():
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"{name}_{k}", v)
            if results["ensemble"]:
                for k, v in results["ensemble"].items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"ensemble_{k}", v)
            if "best_threshold" in results:
                mlflow.log_metric("best_threshold", results["best_threshold"])
                mlflow.log_metric("expected_cost_per_applicant", results["expected_cost_per_applicant"])

            # Log artifacts
            mlflow.log_artifacts(cfg["paths"]["artifacts"], "model_artifacts")


def train_from_config(config_path: str | Path = "config/config.yaml") -> dict:
    cfg = load_config(config_path)
    return Trainer(config=cfg).run()
