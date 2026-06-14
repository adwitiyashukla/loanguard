# Model Card — LoanGuard Fraud Detection v0.1.0

Following [Mitchell et al. 2019](https://arxiv.org/abs/1810.03993) format.

## Model details

- **Authors**: Adwitiya Shukla
- **Date**: 2026
- **Version**: 0.1.0
- **Model type**: Stacked ensemble of gradient-boosted trees (XGBoost + LightGBM + CatBoost), Isolation Forest, and a denoising autoencoder; logistic-regression meta-learner; isotonic calibration.
- **Paper / docs**: see `README.md` and `docs/architecture.md`.

## Intended use

- **Primary**: real-time scoring of unsecured loan applications to assist fraud-ops triage.
- **Primary users**: lending product, risk-ops analysts.
- **Out of scope**: not a substitute for KYC/AML; not for credit-risk decisioning; not for setting interest rates.

## Factors

- **Relevant factors**: loan amount, borrower income, employment length, credit bureau attributes, application velocity per zip/employer, applicant similarity graph features.
- **Evaluation factors**: addr_state, home_ownership, verification_status (proxy fairness slices).

## Metrics

Headline test-set performance on time-split LendingClub data:

| Metric | Value |
|---|---|
| ROC-AUC | 0.89 |
| PR-AUC | 0.42 |
| KS | 0.62 |
| Brier | 0.029 |
| Recall @ 5% FPR | 0.69 |

Operating threshold (review): 0.30; operating threshold (decline): 0.70.

## Training data

- **Source**: LendingClub accepted-loans CSV (2007–2018), ~2.2M rows.
- **Label**: weak-supervision (FPD ∩ anomaly rules) — see `src/data/labels.py`.
- **Sample**: 250k chronological-ordered rows (configurable).

## Evaluation data

- Most-recent 15% of loans by `issue_d`, fully held out from feature fitting and model training.

## Ethical considerations

- **Bias risk**: any fraud model can disproportionately flag legitimate borrowers from underrepresented groups. Fairness reports are auto-generated every training run.
- **Disparate-impact mitigation**: monotonic constraints on bureau features; review threshold sits below decline threshold so humans get a second look before a decline.
- **Adverse-action support**: every decline emits top-5 SHAP reason codes; suitable for an adverse-action notice.

## Caveats and recommendations

- Labels are weak-supervised — performance numbers are upper bounds. Retrain on confirmed-fraud labels once available.
- Velocity and graph features are computed from training history at score time; in a deployed system, replace with a streaming feature store.
- Re-evaluate quarterly; trigger retraining when PSI > 0.25 on any tier-1 feature or AUC degrades by >3pp.
