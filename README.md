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

# LoanGuard

Loan application fraud detection on the LendingClub dataset (2007-2018, ~2.2M loans).

Live demo: https://huggingface.co/spaces/adwitiyashukla/loanguard

## Stack

- Data: LendingClub accepted-loans CSV
- Labels: weak-supervision (first-payment default + anomaly rules)
- Features: behavioural, velocity, graph (networkx), WoE + smoothed target encoding
- Models: XGBoost, LightGBM, CatBoost, Isolation Forest, denoising autoencoder (PyTorch)
- Ensemble: logistic-regression stacker with isotonic calibration
- Serving: FastAPI + Streamlit dashboard, packaged as Docker
- Monitoring: Evidently drift report, Prometheus metrics

## Results

Test metrics on a 50k-row time-split subset (0.37% positive rate).

| Metric | XGBoost | LightGBM | CatBoost | Isolation Forest | Autoencoder | Ensemble |
|---|---|---|---|---|---|---|
| ROC-AUC | 0.694 | 0.571 | 0.667 | 0.592 | 0.624 | 0.653 |
| PR-AUC | 0.014 | 0.007 | 0.010 | 0.005 | 0.005 | 0.006 |
| KS | 0.332 | 0.251 | 0.306 | 0.218 | 0.290 | 0.311 |
| Brier | 0.014 | 0.005 | 0.020 | 0.179 | 0.104 | 0.0038 |
| Lift @ top 5% | 3.57x | 2.86x | 2.86x | 0.71x | 0.71x | 3.57x |

Best threshold: 0.51. Expected net cost: $4.48 per applicant on a $50M origination book with the assumed cost matrix ($1,200 per missed fraud, $20 per false alarm).

## Layout

```
src/
  data/         loader, weak-supervision labels, splitter, schema validation
  features/     behavioural, velocity, graph, WoE and target encoders
  models/       base class, XGBoost, LightGBM, CatBoost, Isolation Forest, autoencoder, ensemble
  training/     trainer, Optuna tuner
  evaluation/   metrics, cost sweep, SHAP explainer, fairness report
  api/          FastAPI schemas, service, main
  dashboard/    Streamlit risk console
  monitoring/   drift monitor
  utils/        config, logging, IO
scripts/        download_data, train
config/         config.yaml
artifacts/      trained models (Git LFS)
```

## Run

```
git clone https://github.com/adwitiyashukla/loanguard.git
cd loanguard
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_data.py
python scripts/train.py --config config/config.yaml
streamlit run app.py
```

## Deploy

The repo doubles as a Docker Space on HuggingFace. Push to `huggingface.co/spaces/adwitiyashukla/loanguard` and the Space rebuilds automatically.
