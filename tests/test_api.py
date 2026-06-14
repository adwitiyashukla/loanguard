"""API integration tests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def test_root():
    res = client.get("/")
    assert res.status_code == 200
    assert "endpoints" in res.json()


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] in ("ok", "degraded")


def test_score_returns_503_when_model_not_loaded():
    """Without artifacts on disk, /score should return 503 — not 500."""
    payload = {
        "loan_amnt": 10000,
        "term": 36,
        "int_rate": 12.5,
        "installment": 334.5,
        "grade": "B",
        "sub_grade": "B3",
        "annual_inc": 50000,
        "dti": 18.0,
        "purpose": "debt_consolidation",
        "zip_code": "941xx",
        "addr_state": "CA",
        "revol_util": 30.0,
        "revol_bal": 5000,
        "open_acc": 5,
        "total_acc": 12,
    }
    res = client.post("/score", json=payload)
    # Either 200 (if you've already trained and a model is loaded)
    # or 503 (no model). 500 = bug.
    assert res.status_code in (200, 503)


def test_metrics_endpoint_exposes_prometheus():
    res = client.get("/metrics")
    assert res.status_code == 200
    assert "loanguard_predict_requests_total" in res.text or "python_info" in res.text
