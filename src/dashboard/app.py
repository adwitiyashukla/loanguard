"""Streamlit risk analyst console.

Three views:
  1. Portfolio Overview  — fraud rate, score distribution, top features
  2. Application Triage  — score a single application, see reason codes
  3. Drift Monitor       — PSI per feature, training vs. production

Run with:
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure repo root on path even if running from elsewhere
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402
import plotly.express as px  # noqa: E402

from src.utils.io import load_joblib  # noqa: E402
from src.api.schemas import LoanApplication  # noqa: E402
from src.api.service import ScoringService  # noqa: E402


# ------------------------------------------------------------------ #
# Page config
# ------------------------------------------------------------------ #

st.set_page_config(
    page_title="LoanGuard | Risk Console",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("LoanGuard — Risk Analyst Console")
st.caption("Real-time fraud risk for loan applications • powered by an XGBoost / LGBM / AE stack")


# ------------------------------------------------------------------ #
# Cached resources
# ------------------------------------------------------------------ #

@st.cache_resource(show_spinner="Loading model…")
def get_service() -> ScoringService:
    svc = ScoringService(artifacts_dir=ROOT / "artifacts")
    svc.load()
    return svc


@st.cache_data
def load_test_predictions() -> pd.DataFrame | None:
    path = ROOT / "artifacts" / "test_predictions.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


svc = get_service()

# ------------------------------------------------------------------ #
# Sidebar
# ------------------------------------------------------------------ #

st.sidebar.header("Navigation")
view = st.sidebar.radio(
    "Choose a view",
    ["Portfolio Overview", "Application Triage", "Drift Monitor"],
)
st.sidebar.divider()
st.sidebar.write(f"**Service**: {'🟢 ready' if svc.is_ready else '🔴 not loaded'}")
st.sidebar.write(f"**Model version**: {svc.model_version}")


# ------------------------------------------------------------------ #
# View 1: Portfolio Overview
# ------------------------------------------------------------------ #

if view == "Portfolio Overview":
    st.subheader("Portfolio overview")
    preds = load_test_predictions()
    if preds is None:
        st.warning(
            "No `artifacts/test_predictions.csv` found yet. Run training first: "
            "`python scripts/train.py`"
        )
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Applications", f"{len(preds):,}")
    col2.metric("Fraud rate", f"{preds['y_true'].mean():.2%}")
    col3.metric("Avg score", f"{preds['proba'].mean():.3f}")
    col4.metric(
        "Flag rate @ 0.5",
        f"{(preds['proba'] >= 0.5).mean():.1%}",
    )

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Score distribution**")
        fig = px.histogram(preds, x="proba", color="y_true", nbins=40, opacity=0.7, barmode="overlay")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**Decile fraud-rate (lift)**")
        d = preds.copy()
        d["decile"] = pd.qcut(d["proba"], 10, labels=False, duplicates="drop") + 1
        decile_rates = d.groupby("decile")["y_true"].mean().reset_index()
        fig = px.bar(decile_rates, x="decile", y="y_true")
        fig.update_layout(yaxis_tickformat=".1%")
        st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------------ #
# View 2: Application Triage
# ------------------------------------------------------------------ #

elif view == "Application Triage":
    st.subheader("Score a single application")
    if not svc.is_ready:
        st.error("Service is not loaded — train the model first.")
        st.stop()

    with st.form("application_form", clear_on_submit=False):
        col1, col2, col3 = st.columns(3)
        loan_amnt = col1.number_input("Loan amount (₹)", 1000.0, 50000.0, 15000.0, step=500.0)
        term = col2.selectbox("Term (months)", [36, 60])
        int_rate = col3.number_input("Interest rate (%)", 5.0, 35.0, 13.0, step=0.1)

        col1, col2, col3 = st.columns(3)
        annual_inc = col1.number_input("Annual income (₹)", 5_000.0, 5_000_000.0, 60_000.0, step=1000.0)
        dti = col2.number_input("DTI", 0.0, 100.0, 18.0, step=0.5)
        emp_length = col3.number_input("Employment length (yrs)", 0.0, 10.0, 4.0, step=1.0)

        col1, col2, col3 = st.columns(3)
        grade = col1.selectbox("Grade", ["A", "B", "C", "D", "E", "F", "G"])
        sub_grade = col2.text_input("Sub-grade", value="B3")
        purpose = col3.selectbox(
            "Purpose",
            ["debt_consolidation", "credit_card", "home_improvement", "other",
             "major_purchase", "small_business", "car", "medical", "moving"],
        )

        col1, col2, col3 = st.columns(3)
        home = col1.selectbox("Home ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
        verif = col2.selectbox("Verification status", ["Verified", "Source Verified", "Not Verified"])
        addr_state = col3.text_input("State", value="CA")

        col1, col2, col3 = st.columns(3)
        zip_code = col1.text_input("ZIP code (5-digit or 3+xx)", value="941xx")
        delinq_2yrs = col2.number_input("Delinquencies (last 2 yrs)", 0, 50, 0)
        revol_util = col3.number_input("Revolving utilisation (%)", 0.0, 200.0, 50.0)

        submitted = st.form_submit_button("Score application", type="primary")

    if submitted:
        installment = loan_amnt * (int_rate / 1200) / (1 - (1 + int_rate / 1200) ** -term)
        app = LoanApplication(
            loan_amnt=loan_amnt,
            term=int(term),
            int_rate=int_rate,
            installment=round(installment, 2),
            grade=grade,
            sub_grade=sub_grade,
            emp_length=emp_length,
            home_ownership=home,
            annual_inc=annual_inc,
            verification_status=verif,
            purpose=purpose,
            zip_code=zip_code,
            addr_state=addr_state,
            dti=dti,
            delinq_2yrs=int(delinq_2yrs),
            revol_util=revol_util,
            issue_d=date.today(),
        )
        result = svc.score_one(app)

        c1, c2, c3 = st.columns([1, 1, 2])
        c1.metric("Fraud score", f"{result.fraud_score:.3f}")
        c2.metric(
            "Decision",
            result.decision,
            delta_color="off" if result.decision == "APPROVE" else "inverse",
        )
        c3.markdown(
            f"**Thresholds** — review ≥ {result.threshold_review:.2f}, "
            f"decline ≥ {result.threshold_decline:.2f}"
        )

        if result.reason_codes:
            st.markdown("**Top reason codes**")
            df_reasons = pd.DataFrame([rc.model_dump() for rc in result.reason_codes])
            df_reasons = df_reasons.sort_values("contribution", key=lambda s: s.abs(), ascending=False)
            fig = px.bar(
                df_reasons,
                x="contribution",
                y="feature",
                orientation="h",
                color="contribution",
                color_continuous_scale="RdBu_r",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df_reasons, use_container_width=True)


# ------------------------------------------------------------------ #
# View 3: Drift Monitor
# ------------------------------------------------------------------ #

elif view == "Drift Monitor":
    st.subheader("Feature drift (PSI)")
    monitor_path = ROOT / "artifacts" / "drift_monitor.joblib"
    if not monitor_path.exists():
        st.info(
            "Drift monitor not yet built. Run training, then a fit() of "
            "`DriftMonitor` is included in the artifacts."
        )
        st.stop()

    monitor = load_joblib(monitor_path)
    uploaded = st.file_uploader("Upload a recent batch (CSV) to compare", type="csv")
    if uploaded is not None:
        recent = pd.read_csv(uploaded)
        report = monitor.psi_report(recent)
        st.dataframe(report, use_container_width=True)
        st.markdown(
            f"**Features in alert (PSI > {monitor.psi_alert}):** "
            f"{report['alert'].sum()}"
        )
        fig = px.bar(report.head(20), x="feature", y="psi", color="alert")
        st.plotly_chart(fig, use_container_width=True)
