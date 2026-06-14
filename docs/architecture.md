# LoanGuard — Architecture

## System overview

LoanGuard is split into four loosely-coupled subsystems:

1. **Offline training pipeline** — reads raw data, builds features, trains models, registers artifacts in MLflow.
2. **Online scoring service (FastAPI)** — loads the latest model from the registry, exposes `/score` and `/score/batch`.
3. **Risk analyst dashboard (Streamlit)** — interactive triage and portfolio diagnostics.
4. **Monitoring (Evidently + Prometheus)** — drift detection on input features and predicted score distribution.

Every subsystem can be deployed independently. The dashboard talks to the API; the API and dashboard share the artifact directory (read-only).

## Why a stacking ensemble?

| Model | Strength | Weakness |
|---|---|---|
| XGBoost | Best raw signal on tabular features; supports monotonic constraints | Sensitive to imbalanced classes; can overfit small categories |
| LightGBM | Faster than XGB on high-dimensional sparse data; native categorical support | Slightly weaker on small-N problems |
| CatBoost | Best out-of-the-box on raw categoricals (no encoding needed) | Slower training; less ecosystem |
| Isolation Forest | Catches novel anomalies the supervised models haven't seen | No probability semantics; noisy |
| Autoencoder | Captures multivariate dependencies; learns the manifold of "normal" applications | Hyperparameter-sensitive; needs scaled features |

A logistic-regression stacker learns the right combination per business case. The isotonic calibrator ensures the final score is interpretable as a probability.

## Why weak-supervision labelling?

The LendingClub dataset (like any lender's earliest cohorts) has no clean fraud flag. We construct one using four signals that are independently defensible and individually weak; a row is labelled fraud only when it triggers at least one strong signal (FPD) or two weak signals (anomaly rules).

The label policy is **transparent**, lives in `config.yaml`, and is **reproducible**. Once a real labelled fraud panel is available, the same model pipeline retrains on it without code changes.

## Latency budget

| Hop | Budget |
|---|---|
| Pydantic validation | 1 ms |
| Feature build (one row) | 10–20 ms |
| Ensemble predict | 5–8 ms |
| SHAP explanation | 30–50 ms |
| **Total (single app)** | **~50–80 ms p50** |

Stays comfortably under a 200 ms SLA at the API edge. Batch endpoint is throughput-optimised — 5000-row batch in ~3 seconds.

## Data leakage protection

Three explicit barriers against leakage:

1. **Application-time columns only.** The loader's `APPLICATION_TIME_COLUMNS` allow-list explicitly excludes every post-funding field (`recoveries`, `total_pymnt`, etc.).
2. **Out-of-fold meta features.** The stacking ensemble builds the meta-learner's training matrix using K-fold OOF predictions — never with in-fold base-model predictions.
3. **Time-based split.** Default split strategy is chronological, not random. Train ends before val starts; val ends before test starts.

## Fairness scope

LendingClub has no protected-class attributes. The fairness scaffolding in `src/evaluation/fairness.py` computes per-group AUC / TPR / FPR for any column you pass; in production we'd pass the protected attribute(s) once available. The dashboard surfaces the report.
