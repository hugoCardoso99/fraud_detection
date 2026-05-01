"""
api.py — FastAPI REST API for the fraud detection model.

Endpoints:
    GET  /health              — Health check & model status
    GET  /model/info          — Model metadata, threshold, feature list
    POST /predict             — Predict fraud for a single transaction
    POST /predict/batch       — Predict fraud for multiple transactions
    PUT  /threshold           — Update the decision threshold at runtime

Run:
    uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
"""

import os
import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference import FraudDetector


# ─── Configuration via environment variables ─────────────────────────────────

MODEL_PATH = os.getenv("MODEL_PATH", "models/XGBoost.joblib")
SCALER_PATH = os.getenv("SCALER_PATH", "data/scaler.joblib")
DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", "0.5"))

# Global detector instance, initialised at startup
detector: FraudDetector | None = None


# ─── Lifespan: load model once on startup ────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model when the server starts, release on shutdown."""
    global detector
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run `python src/preprocess.py && python src/train.py --train-all` first."
        )
    detector = FraudDetector(MODEL_PATH, SCALER_PATH, DEFAULT_THRESHOLD)
    yield
    detector = None


app = FastAPI(
    title="Fraud Detection API",
    description=(
        "REST API for real-time credit card fraud detection. "
        "Serves a trained model with configurable decision threshold."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Pydantic schemas ───────────────────────────────────────────────────────

class Transaction(BaseModel):
    """A single credit card transaction.

    V1–V28 are PCA-transformed features (already scaled).
    Time and Amount are raw and will be scaled by the API.
    """
    Time: float = Field(..., description="Seconds elapsed since first transaction in dataset")
    V1: float = 0.0
    V2: float = 0.0
    V3: float = 0.0
    V4: float = 0.0
    V5: float = 0.0
    V6: float = 0.0
    V7: float = 0.0
    V8: float = 0.0
    V9: float = 0.0
    V10: float = 0.0
    V11: float = 0.0
    V12: float = 0.0
    V13: float = 0.0
    V14: float = 0.0
    V15: float = 0.0
    V16: float = 0.0
    V17: float = 0.0
    V18: float = 0.0
    V19: float = 0.0
    V20: float = 0.0
    V21: float = 0.0
    V22: float = 0.0
    V23: float = 0.0
    V24: float = 0.0
    V25: float = 0.0
    V26: float = 0.0
    V27: float = 0.0
    V28: float = 0.0
    Amount: float = Field(..., description="Transaction amount in euros")

    model_config = {"json_schema_extra": {
        "examples": [{
            "Time": 0.0,
            "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
            "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10,
            "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
            "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
            "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
            "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
            "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02,
            "Amount": 149.62,
        }]
    }}


class PredictionResponse(BaseModel):
    fraud_probability: float = Field(..., description="Model confidence that the transaction is fraudulent (0–1)")
    is_fraud: bool = Field(..., description="Binary prediction at the current threshold")
    risk_level: str = Field(..., description="Low / Medium / High risk category")
    threshold_used: float = Field(..., description="Decision threshold that was applied")


class BatchPredictionRequest(BaseModel):
    transactions: list[Transaction]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    summary: dict


class ThresholdUpdate(BaseModel):
    threshold: float = Field(..., ge=0.0, le=1.0, description="New decision threshold (0–1)")


class ModelInfo(BaseModel):
    model_type: str
    threshold: float
    feature_count: int
    features: list[str]


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Check if the API and model are operational."""
    return {
        "status": "healthy" if detector else "model_not_loaded",
        "model_loaded": detector is not None,
        "model_type": type(detector.model).__name__ if detector else None,
        "threshold": detector.threshold if detector else None,
    }


@app.get("/model/info", response_model=ModelInfo)
async def model_info():
    """Return metadata about the loaded model."""
    if not detector:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ModelInfo(
        model_type=type(detector.model).__name__,
        threshold=detector.threshold,
        feature_count=len(detector.feature_names) if detector.feature_names else 30,
        features=detector.feature_names or [],
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(transaction: Transaction):
    """
    Predict fraud for a single transaction.

    Returns the fraud probability, binary label, and risk level.
    The decision threshold can be changed via PUT /threshold.
    """
    if not detector:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    result = detector.predict_single(transaction.model_dump())
    latency_ms = (time.perf_counter() - start) * 1000

    return PredictionResponse(
        fraud_probability=round(result["fraud_probability"], 6),
        is_fraud=result["is_fraud"],
        risk_level=result["risk_level"],
        threshold_used=detector.threshold,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict fraud for a batch of transactions.

    More efficient than calling /predict in a loop — the model
    scores all transactions in a single vectorized pass.
    """
    if not detector:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(request.transactions) == 0:
        raise HTTPException(status_code=400, detail="Empty transaction list")

    if len(request.transactions) > 10_000:
        raise HTTPException(status_code=400, detail="Batch size limited to 10,000 transactions")

    # Convert to DataFrame for vectorized prediction
    records = [t.model_dump() for t in request.transactions]
    df = pd.DataFrame(records)

    start = time.perf_counter()
    results_df = detector.predict(df)
    latency_ms = (time.perf_counter() - start) * 1000

    predictions = []
    for _, row in results_df.iterrows():
        predictions.append(PredictionResponse(
            fraud_probability=round(float(row["fraud_probability"]), 6),
            is_fraud=bool(row["is_fraud"]),
            risk_level=str(row["risk_level"]),
            threshold_used=detector.threshold,
        ))

    n_fraud = sum(1 for p in predictions if p.is_fraud)
    summary = {
        "total": len(predictions),
        "flagged_fraud": n_fraud,
        "flagged_rate": round(n_fraud / len(predictions), 4),
        "latency_ms": round(latency_ms, 2),
    }

    return BatchPredictionResponse(predictions=predictions, summary=summary)


@app.put("/threshold")
async def update_threshold(update: ThresholdUpdate):
    """
    Update the decision threshold at runtime.

    This is useful for A/B testing different thresholds or adjusting
    the precision/recall tradeoff without restarting the server.

    Interview talking point: In production, you'd want to log threshold
    changes and track how they affect precision/recall over time.
    """
    if not detector:
        raise HTTPException(status_code=503, detail="Model not loaded")

    old_threshold = detector.threshold
    detector.threshold = update.threshold

    return {
        "old_threshold": old_threshold,
        "new_threshold": detector.threshold,
        "message": f"Threshold updated from {old_threshold} to {detector.threshold}",
    }
