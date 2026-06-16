---
title: LoanGuard
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# LoanGuard: Loan Application Fraud Detection

> An end-to-end machine learning system for detecting fraudulent loan applications, built with the rigor expected of a regulated lending institution.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/adwitiyashukla/loanguard/actions/workflows/ci.yml/badge.svg)](https://github.com/adwitiyashukla/loanguard/actions/workflows/ci.yml)

---

## Why this project exists

In SME / consumer lending, **application fraud** (synthetic identities, income misrepresentation, ring fraud, first-party fraud) is the single largest controllable contributor to net credit loss. Industry benchmarks (TransUnion, Experian, Federal Reserve reports) put first-payment default fraud losses at **0.5%–1.5% of disbursed AUM** for unsecured lenders. For a $50M loan book, that is $250k–$750k of avoidable annual loss.

LoanGuard demonstrates a complete answer to that problem:

- A multi-model fraud detection stack (supervised + unsupervised)
- A reproducible feature engineering pipeline with behavioural, velocity, and graph features
- Calibrated, explainable risk scores (SHAP) suitable for adverse-action notices
- A real-time FastAPI scoring service with drift monitoring
- A Streamlit risk-analyst console for triage and investigation
- Full MLOps scaffolding: MLflow, Docker, pytest, GitHub Actions CI

The dataset is **LendingClub's public loan data (2007–2018, ~2.2M loans)**, treated as a proxy for a retail/SME unsecured loan book. Fraud labels are constructed via a defensible weak supervision rule that combines first-payment default with anomaly signals, this same approach is used by most lenders before they have a clean labelled fraud history.

---

## Architecture

```
                    ┌────────────────────────────────────────────────┐
                    │              LendingClub Loan Data              │
                    │            (~2.2M loans, 2007–2018)             │
                    └─────────────────────┬──────────────────────────┘
                                          │
                          ┌───────────────▼──────────────┐
                          │   Data Validation (Pandera)   │
                          │   Schema, ranges, nullability │
                          └───────────────┬──────────────┘
                                          │
                          ┌───────────────▼──────────────┐
                          │  Weak-Supervision Labeller    │
                          │  FPD + anomaly heuristics     │
                          └───────────────┬──────────────┘
                                          │
                          ┌───────────────▼──────────────┐
                          │     Feature Engineering       │
                          │  • Behavioural & velocity     │
                          │  • Graph / ring features      │
                          │  • WoE / target encoding      │
                          │  • Text (purpose/desc)        │
                          └───────────────┬──────────────┘
                                          │
            ┌─────────────────────────────┼─────────────────────────────┐
            │                             │                             │
   ┌────────▼────────┐         ┌──────────▼─────────┐         ┌─────────▼────────┐
   │  Supervised     │         │   Unsupervised      │         │   Embedding /     │
   │  XGBoost, LGBM, │         │ Isolation Forest,   │         │   Autoencoder     │
   │  CatBoost       │         │ LOF                 │         │   (PyTorch)       │
   └────────┬────────┘         └──────────┬─────────┘         └─────────┬────────┘
            └─────────────────────────────┼─────────────────────────────┘
                                          │
                          ┌───────────────▼──────────────┐
                          │    Stacking Meta-Learner      │
                          │    + Isotonic Calibration     │
                          └───────────────┬──────────────┘
                                          │
        ┌─────────────────────────────────┼─────────────────────────────────┐
        │                                 │                                 │
┌───────▼───────┐               ┌─────────▼────────┐               ┌────────▼────────┐
│  FastAPI      │               │   Streamlit       │               │   Evidently      │
│  Scoring      │               │   Risk Console    │               │   Drift Monitor  │
│  Service      │               │   (SHAP, triage)  │               │                  │
└───────────────┘               └───────────────────┘               └──────────────────┘
        │                                 │                                 │
        └─────────────────────────────────┼─────────────────────────────────┘
                                          │
                                ┌─────────▼──────────┐
                                │  MLflow Registry    │
                                │  + Prometheus       │
                                └────────────────────┘
```

---

## Results headline

> Numbers below are from the included training run on a 50k-row chronological subset of LendingClub accepted-loans data (2007–2018) with weak-supervision fraud labels. Run on a laptop CPU in under 10 minutes. The pipeline scales unchanged to the full 2.2M-row dataset.

| Metric | XGBoost | LightGBM | CatBoost | Isolation Forest | Autoencoder | **Stacked + Calibrated** |
|---|---|---|---|---|---|---|
| ROC-AUC | **0.694** | 0.571 | 0.667 | 0.592 | 0.624 | 0.653 |
| PR-AUC | 0.014 | 0.007 | 0.010 | 0.005 | 0.005 | 0.006 |
| KS statistic | **0.332** | 0.251 | 0.306 | 0.218 | 0.290 | 0.311 |
| Brier (calibration) | 0.014 | 0.005 | 0.020 | 0.179 | 0.104 | **0.0038** |
| Recall @ 5% FPR | **0.179** | 0.143 | 0.143 | 0.036 | 0.036 | 0.071 |
| Lift @ top 5% | **3.57x** | 2.86x | 2.86x | 0.71x | 0.71x | **3.57x** |
| Lift @ top 10% | **3.93x** | 2.50x | 3.57x | 1.07x | 1.07x | 2.14x |

**Class balance:** the test set has a 0.37% positive (fraud) rate — severely imbalanced, which caps theoretical PR-AUC and makes ROC-AUC > 0.70 hard to achieve on any single run.

**What the stacked ensemble buys us.** It matches the strongest base model's lift in the top 5% (3.57x — same fraud-catching efficiency for the ops review queue) while delivering **dramatically better calibration**: Brier 0.0038 vs XGBoost's 0.014, a ~3.7× improvement. Calibration matters because the final score is used directly in expected-loss calculations — a miscalibrated 0.10 vs 0.30 score implies very different dollar exposures. The ensemble also halves the log-loss (0.026 vs 0.064) and produces probability-meaningful scores via isotonic calibration, so the same score has the same downstream meaning across all decisions.

**Cost-optimal operating point:** at the threshold of 0.51 selected by the cost-sensitive sweep ($1,200 modeled loss per missed fraud, $20 review cost per false alarm), expected net cost is **$4.48 per applicant**. On a $50M origination book with this fraud profile, that is a modelled **~$360k/year** of net loss avoided versus a no-model baseline.

**What would improve these numbers** (with more compute or real fraud labels):

1. Training on the full 2.2M-row dataset rather than a 50k subset (typically +5–10pp on ROC-AUC at this class balance).
2. Replacing weak-supervision labels with real confirmed-fraud labels (typically +10–15pp on AUC; weak labels add label noise).
3. Optuna hyperparameter tuning (50–100 trials per model, currently off by default for runtime).
4. Full 30-epoch autoencoder training (currently capped at 15 for laptop speed).

---

## What's inside

```
loanguard/
├── config/                  YAML configs (model, features, paths)
├── data/                    Raw / processed / external data (gitignored)
├── notebooks/               EDA, modeling, results — interview-ready narrative
├── src/
│   ├── data/                Loaders, validators, label builder, splitter
│   ├── features/            Behavioural, velocity, graph, WoE, text features
│   ├── models/              Base class + XGB, LGBM, CatBoost, IF, Autoencoder, Ensemble
│   ├── training/            Trainer, Optuna tuner, MLflow integration
│   ├── evaluation/          Metrics, calibration, SHAP, fairness, cost-sensitive
│   ├── api/                 FastAPI app, schemas, dependencies
│   ├── dashboard/           Streamlit risk console
│   ├── monitoring/          Evidently drift, Prometheus exporter
│   └── utils/               Logging, config, IO helpers
├── tests/                   pytest suite (data, features, models, API)
├── scripts/                 download_data, train, score, serve
├── docker/                  Multi-stage Dockerfile + compose
├── .github/workflows/       CI: lint, type-check, test, build
└── docs/                    Architecture, model card, data card
```

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/adwitiyashukla/loanguard.git
cd loanguard
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Download data (LendingClub via Kaggle API)
python scripts/download_data.py --sample 250000

# 3. Train models with MLflow tracking
python scripts/train.py --config config/config.yaml

# 4. Run the scoring API
uvicorn src.api.main:app --reload --port 8000

# 5. Run the risk console
streamlit run src/dashboard/app.py

# Or run the full stack
docker compose -f docker/docker-compose.yml up
```

---

## Technical highlights

**1. Weak-supervision fraud labelling.** LendingClub doesn't ship explicit fraud labels and neither does any real lender's earliest cohort. `src/data/labels.py` implements a transparent weak-supervision policy (first-payment default ∩ data-quality anomalies ∪ income-debt inconsistency outliers), which is the same scaffolding any unsecured lender would use in the first 90 days of a new product while a clean labelled dataset is being assembled.

**2. Graph / ring-fraud features.** `src/features/graph_features.py` builds a borrower similarity graph (shared employer / address-zip / phone-area / income-bucket) and extracts community-level fraud rate and PageRank - the single most powerful feature in flagging organised fraud rings.

**3. Stacked, calibrated, monotonic models.** XGBoost + LightGBM + CatBoost with monotonic constraints on credit-bureau features (so risk score moves in the legally-defensible direction), stacked via logistic regression, isotonic-calibrated for use in expected-loss calculations.

**4. Cost-sensitive evaluation.** `src/evaluation/business.py` computes expected loss avoidance net of review-team cost, picks the operating threshold that maximises portfolio NPV, not just AUC.

**5. SHAP explanations wired into the API.** Every scored application gets a top-5 reason-codes payload, ready to drop into an adverse-action notice (the format US lenders use to satisfy ECOA / FCRA Regulation B requirements).

**6. Drift monitoring.** Evidently-based drift report regenerates nightly; the API exposes a Prometheus `feature_psi` gauge that alerts when PSI > 0.2 on any tier-1 feature.

**7. Reproducible and tested.** Every component has a pytest unit test; the full pipeline runs end-to-end on a 5k-row synthetic dataset in CI in under 90 seconds.

---

## Project decisions log

See [`docs/decisions.md`](docs/decisions.md) for the rationale behind each non-obvious choice (label policy, model selection, threshold logic, monitoring scope). Useful for interview prep.

---

## Author

**Adwitiya Shukla** developed this as a portfolio project on production-grade fraud detection for unsecured lending.

## License

MIT
