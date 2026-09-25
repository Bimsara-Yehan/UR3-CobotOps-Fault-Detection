"""Pydantic request/response schemas for the prediction API."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.features import LAG_ROWS

# Physical sanity bounds — wide headroom around the observed training range so typos and unit mistakes
# are caught before reaching the model.  These are not from a UR3 datasheet; they are round numbers
# chosen to sit well outside the data (joint currents -6.25 to 6.47 A, tool current 0.07 to 0.60 A).
# The UI warns the operator when values are outside the training range.
JointCurrent = Annotated[float, Field(ge=-10.0, le=10.0, allow_inf_nan=False, description="Joint motor current in amps")]
ToolCurrent = Annotated[float, Field(ge=0.0, le=3.0, allow_inf_nan=False, description="Tool current in amps")]


class CurrentReading(BaseModel):
    """The reading to score: the six joint currents and the tool current at the latest moment."""

    model_config = ConfigDict(extra="forbid")

    Current_J0: JointCurrent
    Current_J1: JointCurrent
    Current_J2: JointCurrent
    Current_J3: JointCurrent
    Current_J4: JointCurrent
    Current_J5: JointCurrent
    Tool_current: ToolCurrent


class EarlierCurrents(BaseModel):
    """The six joint currents from an earlier reading, needed for the current-trend feature."""

    model_config = ConfigDict(extra="forbid")

    Current_J0: JointCurrent
    Current_J1: JointCurrent
    Current_J2: JointCurrent
    Current_J3: JointCurrent
    Current_J4: JointCurrent
    Current_J5: JointCurrent


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
