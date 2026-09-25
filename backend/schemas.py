"""Pydantic request/response schemas for the prediction API."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.features import LAG_ROWS

# Sensor readings must be real numbers: NaN and infinity are rejected instead of being imputed silently.
Reading = Annotated[float, Field(allow_inf_nan=False)]


class CurrentReading(BaseModel):
    """The reading to score: the six joint currents and the tool current at the latest moment."""

    model_config = ConfigDict(extra="forbid")

    Current_J0: Reading
    Current_J1: Reading
    Current_J2: Reading
    Current_J3: Reading
    Current_J4: Reading
    Current_J5: Reading
    Tool_current: Reading


class EarlierCurrents(BaseModel):
    """The six joint currents from an earlier reading, needed for the current-trend feature."""

    model_config = ConfigDict(extra="forbid")

    Current_J0: Reading
    Current_J1: Reading
    Current_J2: Reading
    Current_J3: Reading
    Current_J4: Reading
    Current_J5: Reading


class PredictionRequest(BaseModel):
    """A reading plus the joint currents from `LAG_ROWS` readings earlier, from the same working cycle.

    The model uses how much the total joint current changed over the last few readings, so a single
    reading is not enough. The API cannot check that the two readings really are consecutive and from
    the same cycle; that is the caller's responsibility.
    """

    model_config = ConfigDict(extra="forbid")

    current: CurrentReading
    earlier: EarlierCurrents = Field(
        description=f"Joint currents of the reading {LAG_ROWS} readings before `current` (about {LAG_ROWS} seconds earlier)."
    )


class Factor(BaseModel):
    """How one input pushed this prediction (SHAP value, in log-odds of a protective stop)."""

    feature: str
    value: float
    effect: Literal["raises risk", "lowers risk"]
    contribution: float


class Thresholds(BaseModel):
    """Probability cut-offs: `alert` is the model's stop threshold, `watch` starts the medium band."""

    watch: float
    alert: float


class PredictionResponse(BaseModel):
    label: int
    probability: float
    risk_band: Literal["low", "medium", "high"]
    message: str
    thresholds: Thresholds
    top_factors: list[Factor]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    features: list[str]
    thresholds: Thresholds
    lag_rows: int
