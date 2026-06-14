"""FastAPI application — real-time + batch scoring with Prometheus metrics."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

from ..utils.logging import get_logger, setup_logging
from .schemas import (
    BatchScoreRequest,
    BatchScoreResponse,
    HealthResponse,
    LoanApplication,
    ScoreResponse,
)
from .service import ScoringService


# ---------------------------------------------------------------------- #
# Metrics
# ---------------------------------------------------------------------- #

PREDICT_REQUESTS = Counter(
    "loanguard_predict_requests_total",
    "Number of scoring requests",
    ["endpoint", "decision"],
)
PREDICT_LATENCY = Histogram(
    "loanguard_predict_latency_seconds",
    "Latency of scoring requests",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
SCORE_HIST = Histogram(
    "loanguard_fraud_score",
    "Distribution of fraud scores",
    buckets=tuple(i / 20 for i in range(21)),
)
FEATURE_PSI = Gauge(
    "loanguard_feature_psi",
    "Population stability index for a feature vs. training distribution",
    ["feature"],
)


# ---------------------------------------------------------------------- #
# Lifespan
# ---------------------------------------------------------------------- #

service = ScoringService()
boot_time = time.time()
log = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: D401
    setup_logging()
    log.info("LoanGuard API booting...")
    service.load()
    yield
    log.info("LoanGuard API shutting down")


# ---------------------------------------------------------------------- #
# App
# ---------------------------------------------------------------------- #

app = FastAPI(
    title="LoanGuard Fraud Scoring API",
    description="Real-time fraud risk scoring for loan applications.",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------- #
# Routes
# ---------------------------------------------------------------------- #

@app.get("/", tags=["meta"])
def root():
    return {
        "service": "loanguard",
        "version": app.version,
        "endpoints": ["/health", "/score", "/score/batch", "/metrics", "/docs"],
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    status = "ok" if service.is_ready else "degraded"
    return HealthResponse(
        status=status,  # type: ignore[arg-type]
        model_loaded=service.is_ready,
        model_version=service.model_version if service.is_ready else None,
        uptime_seconds=time.time() - boot_time,
    )


@app.post("/score", response_model=ScoreResponse, tags=["scoring"])
def score(app_in: LoanApplication):
    if not service.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    with PREDICT_LATENCY.labels(endpoint="score").time():
        try:
            result = service.score_one(app_in)
        except Exception as exc:
            log.exception("Scoring failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    PREDICT_REQUESTS.labels(endpoint="score", decision=result.decision).inc()
    SCORE_HIST.observe(result.fraud_score)
    return result


@app.post("/score/batch", response_model=BatchScoreResponse, tags=["scoring"])
def score_batch(req: BatchScoreRequest):
    if not service.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    if len(req.applications) > 5000:
        raise HTTPException(status_code=413, detail="Batch too large (max 5000).")
    with PREDICT_LATENCY.labels(endpoint="score_batch").time():
        results = service.score_many(req.applications)
    for r in results:
        PREDICT_REQUESTS.labels(endpoint="score_batch", decision=r.decision).inc()
        SCORE_HIST.observe(r.fraud_score)
    return BatchScoreResponse(
        scored_at=datetime.now(timezone.utc),
        model_version=service.model_version,
        results=results,
    )


@app.post("/reload", tags=["meta"])
def reload_artifacts():
    """Reload model artifacts (e.g. after a new training run)."""
    service.load()
    return {"reloaded": True, "ready": service.is_ready}


@app.get("/metrics", tags=["meta"])
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
