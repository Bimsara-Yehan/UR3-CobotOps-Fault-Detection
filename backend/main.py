"""FastAPI backend: loads models/final_pipeline.joblib once at startup, serves /health and /predict."""

from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import HealthResponse, PredictionResponse, SensorReading
from src.config import MODELS_DIR

MODEL_PATH = MODELS_DIR / "final_pipeline.joblib"

model_bundle = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if MODEL_PATH.exists():
        bundle = joblib.load(MODEL_PATH)
        model_bundle["pipeline"] = bundle["pipeline"]
        model_bundle["threshold"] = bundle["threshold"]
        model_bundle["features"] = bundle["features"]
    yield
    model_bundle.clear()


app = FastAPI(title="UR3 CobotOps - Protective Stop Risk API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RISK_MESSAGES = {
    "high": "Elevated protective-stop risk detected. Reduce load/speed and inspect before continuing.",
    "medium": "Moderate protective-stop risk. Monitor closely.",
    "low": "Normal operating range.",
}


def _build_feature_row(reading: SensorReading) -> pd.DataFrame:
    data = reading.model_dump()
    current_cols = [f"Current_J{i}" for i in range(6)]
    data["total_joint_current"] = sum(abs(data[c]) for c in current_cols)
    # A single API call has no prior readings to diff against; the trained
    # imputer fills this with the training median (see src/pipeline.py).
    data["total_joint_current_delta3"] = None
    return pd.DataFrame([data])


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_loaded="pipeline" in model_bundle)


@app.post("/predict", response_model=PredictionResponse)
def predict(reading: SensorReading):
    if "pipeline" not in model_bundle:
        raise HTTPException(status_code=503, detail="Model not loaded: models/final_pipeline.joblib is missing.")

    pipeline = model_bundle["pipeline"]
    threshold = model_bundle["threshold"]
    features = model_bundle["features"]

    X = _build_feature_row(reading)[features]
    probability = float(pipeline.predict_proba(X)[0, 1])
    label = int(probability >= threshold)

    if probability >= threshold:
        risk_band = "high"
    elif probability >= threshold * 0.5:
        risk_band = "medium"
    else:
        risk_band = "low"

    return PredictionResponse(
        label=label,
        probability=probability,
        risk_band=risk_band,
        message=f"{RISK_MESSAGES[risk_band]} This is a decision-support estimate, not a certified safety system.",
    )
