# Project Decisions Log

This is the "why" of every non-obvious choice in LoanGuard — written so you can answer "why did you do X?" in an interview without rehearsing.

---

## 1. Why fraud detection over credit scoring?

On a young unsecured SME or consumer book, the single biggest controllable line item is application fraud, not interest-rate optimisation. Building a fraud-detection layer is a higher-leverage exercise than building yet another XGBoost PD model. It also lets the project showcase unsupervised learning, graph features, weak supervision, and cost-sensitive evaluation — all relevant DS skills that a credit-scoring project wouldn't naturally exercise.

## 2. Why LendingClub data?

There is no public, large-scale, real Indian SME-lending dataset. LendingClub is the closest analog publicly available: unsecured consumer/SME loans, full application-time fields, ~2.2M rows. The transfer of feature ideas (DTI, credit history length, employment verification) to an Indian SME book is direct. I documented this explicitly to avoid claiming we're modelling Indian SME fraud directly.

## 3. Why weak supervision for the fraud label?

LendingClub doesn't ship explicit fraud labels. Three options were on the table:

1. **Treat charge-off as fraud** — too noisy (most charge-offs are credit risk, not fraud).
2. **Buy / license a labelled dataset** — out of scope for a portfolio project.
3. **Weak-supervision rule-based labels** — what we did.

The weak-supervision policy is transparent, lives in config, and is the same scaffolding a real lender uses in their first year. The label policy can be retired the moment a clean panel of confirmed-fraud cases is available.

## 4. Why monotonic constraints on credit-bureau features?

Risk score must move *with* delinquency count, public records, and grade. Without monotonic constraints, the model can learn locally non-monotonic relationships that — while higher AUC on the training set — fail audit and customer dispute resolution under RBI's Fair Practices Code. Forcing monotonicity costs ~0.5pp AUC but is non-negotiable for a regulated product.

## 5. Why a stacking ensemble instead of a single XGBoost?

Two reasons:

1. The unsupervised models (IsolationForest, autoencoder) provide signal on **novel** patterns that the supervised models can't see — the meta-learner gives them ~5–15% weight, which moves the recall-at-low-FPR meaningfully.
2. Diversity across XGB/LGBM/CatBoost reduces variance in the OOF predictions, giving a more stable production model than any single GBM.

The cost is ~3x training time and ~2x inference latency, both well within the budget.

## 6. Why isotonic calibration and not Platt scaling?

The raw scores from a stacked model are not calibrated. We need calibrated probabilities for expected-loss calculations and threshold setting. Isotonic regression makes no parametric assumption about the calibration curve shape, which is important here because the stack's score distribution has a long tail. Platt would force a sigmoid that mis-fits the tails.

## 7. Why time-based split as default, not stratified random?

A credit / fraud model deployed today will score tomorrow's loans, not yesterday's. Stratified random splits hugely overestimate generalisation because they leak the future into the training set. Time-based splits give a realistic estimate of out-of-time performance. Stratified is only used in tests (where we need predictable class balance).

## 8. Why graph features?

Most application fraud is ring fraud: a single bad actor pushes 30–50 synthetic identities through the funnel in days, often sharing employer names or zip clusters. Graph features (component size, degree centrality, shared-key counts) are the single most powerful signal for catching this — far more than any single demographic feature. networkx is fast enough up to ~500k applications; for the full 2M-row dataset we'd switch to GraphTool or a Neo4j-backed pipeline.

## 9. Why cost-sensitive thresholding?

A 0.5 threshold is almost never the right answer. The cost matrix (₹100k per missed fraud vs ₹1.5k per false alarm, derived from Indian lending benchmarks) means the optimal threshold is much lower than 0.5 — we'd rather over-flag and let humans review than miss a fraud. The threshold is recomputed each training run and lives in the artifacts.

## 10. Why SHAP and not LIME?

SHAP TreeExplainer is exact, deterministic, and fast for tree-based models. LIME is sample-based and not as reproducible — bad property for a regulated product where you'll be asked to explain the same decision twice. SHAP also has a coherent global / local story (additive feature attributions) that maps directly onto adverse-action notice format.

## 11. Why three thresholds (APPROVE / REVIEW / DECLINE)?

A binary decision wastes information. Most lenders need a "send to review" lane because hard declines have CX consequences and false-positive rate at the high-score tail is non-zero. The two thresholds — `review_threshold` and `decline_threshold` — are tunable and surface on the dashboard.

## 12. Why log loss + Brier in addition to AUC?

AUC measures ranking; Brier and log-loss measure calibration. A model can have 0.90 AUC and still be miscalibrated, in which case the expected-loss math is wrong. Tracking both forces honest performance measurement.

## 13. Why ship a Streamlit dashboard and not just an API?

The end users at a lender are risk analysts and ops staff, not engineers. They need an interface to triage flagged applications, drill into reason codes, and watch drift. Streamlit gets us that in ~300 LoC; the same functionality in a React+FastAPI app would be 10x the effort.

## 14. Why Evidently for drift instead of rolling my own?

Drift detection done well is its own product (binning logic, statistical tests, report rendering). Evidently is the de-facto standard and integrates trivially with MLflow. Rolling our own would have been pointless undifferentiated work.

## 15. What I would do differently in a real production deployment

1. **Replace weak-supervision labels with the ops team's confirmed-fraud panel** as it grows.
2. **Add an online feature store** (Feast / Tecton) — at scale you can't afford to recompute velocity features at request time.
3. **Move the graph features into a streaming engine** (Flink + Neo4j) — networkx will not scale past ~500k.
4. **Champion / Challenger framework** — every quarter, retrain a fresh model and shadow-route 5% of traffic to it.
5. **Causal uplift modelling** — current model predicts probability of fraud; a real product would predict the marginal effect of declining vs. reviewing, which is what actually drives decisions.
