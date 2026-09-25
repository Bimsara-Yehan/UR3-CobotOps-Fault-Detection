"""FastAPI backend: loads models/final_pipeline.joblib once at startup, serves /health, /model-info and /predict."""

from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.schemas import (
    Factor,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
    Thresholds,
)
from src.config import MODELS_DIR
from src.features import CURRENT_COLS, LAG_ROWS, WINDOW_ROWS, add_engineered_features

MODEL_PATH = MODELS_DIR / "final_pipeline.joblib"

# Inputs the API can supply: the tool current and joint currents it is sent, plus the engineered columns.
RAW_FEATURES = CURRENT_COLS + ["Tool_current"]
ENGINEERED_FEATURES = ["total_joint_current", "total_joint_current_delta3"]

# Start of the "medium" band: the threshold at which row-level recall reaches 0.80 on out-of-fold train
# predictions (notebook 07, section 7). The "high" band starts at the model's own alert threshold.
# Recompute this if the final model is retrained.
WATCH_THRESHOLD = 0.39

TOP_FACTORS = 3

RISK_MESSAGES = {
    "high": "The readings match the pattern seen when the robot protective-stopped. Check the robot and the work cell before continuing.",
    "medium": "The readings are drifting toward the pattern seen before and during protective stops. Keep watching.",
    "low": "The readings look like normal operation.",
}
DISCLAIMER = "This is a decision-support estimate, not a certified safety system."

model_bundle = {}


def _validate_bundle(bundle):
    """Fail at startup, not on the first request, if the saved model needs inputs this API cannot build."""
    unknown = set(bundle["features"]) - set(RAW_FEATURES) - set(ENGINEERED_FEATURES)
    if unknown:
        raise RuntimeError(f"final_pipeline.joblib needs features the API cannot supply: {sorted(unknown)}")
    if not WATCH_THRESHOLD < bundle["threshold"]:
        raise RuntimeError(f"WATCH_THRESHOLD ({WATCH_THRESHOLD}) must be below the model threshold ({bundle['threshold']:.4f}).")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if MODEL_PATH.exists():
        bundle = joblib.load(MODEL_PATH)
        _validate_bundle(bundle)
        model_bundle["pipeline"] = bundle["pipeline"]
        model_bundle["threshold"] = bundle["threshold"]
        model_bundle["features"] = bundle["features"]
        model_bundle["explainer"] = shap.TreeExplainer(bundle["pipeline"].named_steps["classifier"])
    yield
    model_bundle.clear()


app = FastAPI(title="UR3 CobotOps - Protective Stop Risk API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Report where each input is wrong and why, without echoing the bad value back.

    The default 422 body includes the rejected input; for NaN or infinity that cannot be written as JSON
    and turned a validation error into a 500.
    """
    detail = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": detail})


def build_feature_row(request: PredictionRequest, features) -> pd.DataFrame:
    """One row of model inputs from the current reading and the earlier joint currents.

    The features come from src.features.add_engineered_features, the same code the training data went
    through. It works on a window of WINDOW_ROWS readings; delta3 only uses the first (earlier) and last
    (current) of them, so the readings in between are left empty.
    """
    window = pd.DataFrame(
        [request.earlier.model_dump()] + [{}] * (WINDOW_ROWS - 2) + [request.current.model_dump()]
    )
    return add_engineered_features(window).iloc[[-1]][features].reset_index(drop=True)


def risk_band(probability: float, threshold: float) -> str:
    if probability >= threshold:
        return "high"
    if probability >= WATCH_THRESHOLD:
        return "medium"
    return "low"


def top_factors(pipeline, explainer, X: pd.DataFrame, n: int = TOP_FACTORS) -> list[Factor]:
    """The n inputs that moved this prediction most, from SHAP values on the model's preprocessed inputs."""
    preprocessing = pipeline.named_steps["preprocessing"]
    names = [name.split("__", 1)[1] for name in preprocessing.get_feature_names_out()]
    contributions = explainer.shap_values(preprocessing.transform(X))[0]
    order = np.argsort(-np.abs(contributions))[:n]
    return [
        Factor(
            feature=names[i],
            value=float(X.iloc[0][names[i]]),
            effect="raises risk" if contributions[i] > 0 else "lowers risk",
            contribution=float(contributions[i]),
        )
        for i in order
    ]


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_loaded="pipeline" in model_bundle)


def _require_model():
    if "pipeline" not in model_bundle:
        raise HTTPException(status_code=503, detail="Model not loaded: models/final_pipeline.joblib is missing.")


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    _require_model()
    return ModelInfoResponse(
        features=model_bundle["features"],
        thresholds=Thresholds(watch=WATCH_THRESHOLD, alert=model_bundle["threshold"]),
        lag_rows=LAG_ROWS,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    _require_model()

    pipeline = model_bundle["pipeline"]
    threshold = model_bundle["threshold"]

    X = build_feature_row(request, model_bundle["features"])
    probability = float(pipeline.predict_proba(X)[0, 1])
    band = risk_band(probability, threshold)

    return PredictionResponse(
        label=int(probability >= threshold),
        probability=probability,
        risk_band=band,
        message=f"{RISK_MESSAGES[band]} {DISCLAIMER}",
        thresholds=Thresholds(watch=WATCH_THRESHOLD, alert=threshold),
        top_factors=top_factors(pipeline, model_bundle["explainer"], X),
    )
