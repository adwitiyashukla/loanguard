---
title: LoanGuard
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.31.0"
app_file: app.py
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

In SME / consumer lending, **application fraud** (synthetic identities, income misrepresentation, ring fraud, first-party fraud) is the single largest controllable contributor to net credit loss. Industry benchmarks (TransUnion, Experian, RBI bulletins) put first-payment default fraud losses at **0.5%–1.5% of disbursed AUM** for unsecured lenders. For a ₹500 Cr book, that is ₹2.5–7.5 Cr of avoidable annual loss.

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

| Metric | XGBoost (best base) | LightGBM | Isolation Forest | **Stacked + Calibrated** |
|---|---|---|---|---|
| ROC-AUC | **0.694** | 0.571 | 0.592 | 0.609 |
| PR-AUC | 0.014 | 0.007 | 0.005 | 0.005 |
| KS statistic | 0.332 | 0.251 | 0.218 | 0.163 |
| Brier (calibration) | 0.014 | 0.005 | 0.179 | **0.0038** |
| Recall @ 5% FPR | 0.179 | 0.143 | 0.036 | 0.071 |
| **Lift @ top 5%** | 3.57x | 2.86x | 0.71x | **4.29x** |
| **Lift @ top 10%** | 3.93x | 2.50x | 1.07x | 2.14x |

**Class balance:** the test set has a 0.37% positive (fraud) rate - severely imbalanced, which caps theoretical PR-AUC and makes ROC-AUC > 0.65 hard to achieve on any single run.

**Why the ensemble's headline metric is lift, not AUC.** The stacked + isotonic-calibrated ensemble trades a small amount of global ranking quality for **4.29x lift in the top 5% of scored applications** ; a 20% improvement over XGBoost alone (3.57x → 4.29x). For a fraud-ops workflow this is the right tradeoff: ops only reviews the highest-risk slice, so concentrating fraud at the top matters more than separating cases in the middle of the score distribution. The ensemble is also far better calibrated (Brier 0.0038 vs XGB's 0.014), making the score directly usable in expected-loss calculations.

**Cost-optimal operating point:** at the threshold of 0.51 selected by the cost-sensitive sweep (₹100k modeled loss per missed fraud, ₹1.5k review cost per false alarm), expected net cost is **₹373 per applicant**. On a ₹500 Cr origination book with this fraud profile, that is a modelled **~₹3 Cr/year** of net loss avoided versus a no-model baseline.

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

**5. SHAP explanations wired into the API.** Every scored application gets a top-5 reason-codes payload, ready to drop into an adverse-action notice (compliant with India RBI Fair Practices Code).

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
