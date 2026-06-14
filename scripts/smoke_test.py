"""End-to-end smoke test.

Runs the entire pipeline on a small synthetic dataset and verifies:
  1. Data loads (synthetic fallback works)
  2. Labels build
  3. Split works
  4. FeatureBuilder fits and transforms consistently
  5. Each model fits and scores
  6. Ensemble stacks and calibrates
  7. Metrics compute
  8. Artifacts save & load
  9. ScoringService produces a decision

If this exits 0, you are safe to commit. Run with:

    python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def step(name: str):
    print(f"\n=== {name} ===", flush=True)


def fail(msg: str, exc: Exception | None = None) -> None:
    print(f"\n❌ FAIL: {msg}")
    if exc is not None:
        traceback.print_exc()
    sys.exit(1)


def main() -> None:
    from src.data import LendingClubLoader, build_fraud_labels, LabelConfig
    from src.data.splitter import time_based_split, stratified_split
    from src.features import FeatureBuilder
    from src.models import (
        XGBFraudModel,
        LGBMFraudModel,
        IsolationForestFraudModel,
        StackingFraudEnsemble,
    )
    from src.evaluation import binary_classification_metrics
    from src.evaluation.business import cost_sensitive_evaluation, CostMatrix
    from src.utils.io import save_joblib, load_joblib
    from src.api.service import ScoringService
    from src.api.schemas import LoanApplication
    from datetime import date

    # 1. Load synthetic data
    step("1. Load synthetic data")
    try:
        df = LendingClubLoader._synthetic(n=3000, seed=42)
        assert len(df) == 3000
        print(f"   loaded {len(df):,} rows, {df.shape[1]} cols")
    except Exception as e:
        fail("Synthetic loader broke", e)

    # 2. Build fraud labels
    step("2. Build fraud labels")
    try:
        df = build_fraud_labels(df, LabelConfig())
        rate = df["is_fraud"].mean()
        print(f"   fraud rate: {rate:.2%}")
        assert 0.01 < rate < 0.6, f"Fraud rate {rate:.2%} outside expected range"
    except Exception as e:
        fail("Label builder broke", e)

    # 3. Split
    step("3. Time-based split")
    try:
        train, val, test = stratified_split(df, "is_fraud", 0.15, 0.15, random_seed=1)
        print(f"   train={len(train)} val={len(val)} test={len(test)}")
    except Exception as e:
        fail("Splitter broke", e)

    drop = ["is_fraud", "rule_fpd", "rule_income_anomaly", "rule_debt_inconsist",
            "rule_address_ring", "n_anomalies", "loan_status", "last_pymnt_d", "id"]
    y_train, y_val, y_test = train["is_fraud"], val["is_fraud"], test["is_fraud"]
    X_train_raw = train.drop(columns=drop)
    X_val_raw = val.drop(columns=drop)
    X_test_raw = test.drop(columns=drop)

    # 4. Feature builder
    step("4. FeatureBuilder fit+transform")
    try:
        builder = FeatureBuilder(use_velocity=True, use_graph=True)
        X_train = builder.fit_transform(X_train_raw, y_train)
        X_val = builder.transform(X_val_raw)
        X_test = builder.transform(X_test_raw)
        print(f"   produced {X_train.shape[1]} features")
        assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1]
        assert not X_train.isna().any().any(), "NaNs in transformed training set"
    except Exception as e:
        fail("FeatureBuilder broke", e)

    # 5. Train each model
    step("5. Train base models")
    try:
        xgb = XGBFraudModel(params={"n_estimators": 50, "max_depth": 4}).fit(
            X_train, y_train, eval_set=[(X_val, y_val)]
        )
        lgb = LGBMFraudModel(params={"n_estimators": 50, "num_leaves": 15}).fit(
            X_train, y_train, eval_set=[(X_val, y_val)]
        )
        iso = IsolationForestFraudModel(params={"n_estimators": 50}).fit(X_train)
        print(f"   XGB AUC = {binary_classification_metrics(y_test, xgb.predict_proba(X_test))['roc_auc']:.3f}")
        print(f"   LGB AUC = {binary_classification_metrics(y_test, lgb.predict_proba(X_test))['roc_auc']:.3f}")
        print(f"   IF  AUC = {binary_classification_metrics(y_test, iso.predict_proba(X_test))['roc_auc']:.3f}")
    except Exception as e:
        fail("Model training broke", e)

    # 6. Stacking ensemble
    step("6. Stacking ensemble")
    try:
        ens = StackingFraudEnsemble(
            base_models=[xgb, lgb, iso],
            n_folds=3,
            calibration="isotonic",
        ).fit(X_train, y_train, X_val=X_val, y_val=y_val)
        proba = ens.predict_proba(X_test)
        metrics = binary_classification_metrics(y_test, proba)
        print(f"   ensemble: AUC={metrics['roc_auc']:.3f}, "
              f"PR-AUC={metrics['pr_auc']:.3f}, KS={metrics['ks']:.3f}")
        assert 0 <= proba.min() and proba.max() <= 1
    except Exception as e:
        fail("Ensemble broke", e)

    # 7. Cost sweep
    step("7. Cost-sensitive evaluation")
    try:
        cost_df = cost_sensitive_evaluation(y_test, proba, CostMatrix())
        best = cost_df.loc[cost_df["total_cost"].idxmin()]
        print(f"   optimal threshold = {best['threshold']:.2f}, "
              f"cost/applicant = ₹{best['cost_per_applicant']:.2f}")
    except Exception as e:
        fail("Cost evaluation broke", e)

    # 8. Save + reload artifacts
    step("8. Save & load artifacts")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            save_joblib(builder, tmp_path / "feature_builder.joblib")
            save_joblib(ens, tmp_path / "model_ensemble.joblib")

            builder2 = load_joblib(tmp_path / "feature_builder.joblib")
            ens2 = load_joblib(tmp_path / "model_ensemble.joblib")
            import numpy as np

            # (a) Builder roundtrip — re-loaded builder transforms the SAME
            #     full raw frame to identical features (velocity/graph are
            #     batch-context dependent, so we must transform the same input).
            Xb1 = builder.transform(X_test_raw)
            Xb2 = builder2.transform(X_test_raw)
            assert np.allclose(Xb1.values, Xb2.values, atol=1e-6), (
                "Re-loaded FeatureBuilder produces different features"
            )

            # (b) Model roundtrip — re-loaded ensemble scores identically on
            #     identical features.
            p1 = ens.predict_proba(X_test)
            p2 = ens2.predict_proba(X_test)
            assert np.allclose(p1, p2, atol=1e-6), "Re-loaded model produces different scores"
            print("   serialization roundtrip OK")

            # 9. Scoring service
            step("9. ScoringService end-to-end")
            svc = ScoringService(artifacts_dir=tmp_path)
            svc.load()
            assert svc.is_ready
            app = LoanApplication(
                loan_amnt=15000,
                term=36,
                int_rate=13.5,
                installment=509.0,
                grade="B",
                sub_grade="B3",
                emp_length=4.0,
                home_ownership="RENT",
                annual_inc=60000,
                verification_status="Verified",
                purpose="debt_consolidation",
                zip_code="941xx",
                addr_state="CA",
                dti=18.0,
                revol_util=45.0,
                revol_bal=5000,
                open_acc=8,
                total_acc=15,
                issue_d=date.today(),
            )
            result = svc.score_one(app)
            print(f"   score = {result.fraud_score:.4f}, decision = {result.decision}")
            assert 0 <= result.fraud_score <= 1
            assert result.decision in ("APPROVE", "REVIEW", "DECLINE")
    except Exception as e:
        fail("Serialization or scoring service broke", e)

    print("\n✅ ALL SMOKE CHECKS PASSED — safe to commit.\n")


if __name__ == "__main__":
    main()
