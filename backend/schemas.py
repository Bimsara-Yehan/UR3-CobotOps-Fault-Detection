"""Pydantic request/response schemas for the prediction API."""

from pydantic import BaseModel


class SensorReading(BaseModel):
    """Raw single-row sensor snapshot, matching the eligible-feature list minus the engineered columns."""

    Current_J0: float
    Current_J1: float
    Current_J2: float
    Current_J3: float
    Current_J4: float
    Current_J5: float
    Temperature_T0: float
    Temperature_J1: float
    Temperature_J2: float
    Temperature_J3: float
    Temperature_J4: float
    Temperature_J5: float
    Tool_current: float
    grip_lost: bool = False


class PredictionResponse(BaseModel):
    label: int
    probability: float
    risk_band: str
    message: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
