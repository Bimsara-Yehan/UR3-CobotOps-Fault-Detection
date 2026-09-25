"""API tests: parity with the notebook, input validation, risk bands and explanations."""

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import backend.main as main
from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.features import CURRENT_COLS

RAW_FEATURES = CURRENT_COLS + ["Tool_current"]


@pytest.fixture(scope="module")
def client():
    with TestClient(main.app) as c:
        yield c


@pytest.fixture(scope="module")
def bundle():
    return joblib.load(MODELS_DIR / "final_pipeline.joblib")


@pytest.fixture(scope="module")
def test_df():
    return pd.read_parquet(PROCESSED_DATA_DIR / "test.parquet").reset_index(drop=True)


def request_for(df, i):
    """The API request for row i of `df`: that reading plus the joint currents of the row 3 earlier."""
    return {
        "current": {k: float(df.loc[i, k]) for k in RAW_FEATURES},
        "earlier": {k: float(df.loc[i - 3, k]) for k in CURRENT_COLS},
    }


def sample_rows(df, stop, n):
    """First n rows with the given label that have 3 earlier readings in the same cycle."""
    position = df.groupby("cycle").cumcount().to_numpy()
    return [i for i in range(len(df)) if position[i] >= 3 and df.loc[i, "Robot_ProtectiveStop"] == stop][:n]


VALID = {
    "current": {"Current_J0": 0.1, "Current_J1": -0.2, "Current_J2": 0.3, "Current_J3": 0.0,
                "Current_J4": 0.5, "Current_J5": -0.6, "Tool_current": 0.7},
    "earlier": {f"Current_J{i}": 0.1 for i in range(6)},
}


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_model_info(client, bundle):
    info = client.get("/model-info").json()
    assert info["features"] == bundle["features"]
    assert info["lag_rows"] == 3
    assert info["thresholds"]["alert"] == pytest.approx(bundle["threshold"])
    assert info["thresholds"]["watch"] < info["thresholds"]["alert"]


@pytest.mark.parametrize("stop", [1, 0])
def test_matches_notebook_prediction(client, bundle, test_df, stop):
    """Same probability as the saved pipeline gives on the stored test features (3 stop rows, 3 normal rows)."""
    rows = sample_rows(test_df, stop, 3)
    assert len(rows) == 3
    expected = bundle["pipeline"].predict_proba(test_df.loc[rows, bundle["features"]])[:, 1]
    for i, p in zip(rows, expected):
        response = client.post("/predict", json=request_for(test_df, i))
        assert response.status_code == 200
        body = response.json()
        assert body["probability"] == pytest.approx(p, abs=1e-6)
        assert body["label"] == int(p >= bundle["threshold"])


def test_known_stop_rows_are_flagged(client, test_df):
    for i in sample_rows(test_df, 1, 3):
        assert client.post("/predict", json=request_for(test_df, i)).json()["risk_band"] == "high"


def test_history_changes_the_prediction(client, test_df):
    """The earlier readings matter: a stop row scored against a very different earlier reading changes."""
    i = sample_rows(test_df, 1, 1)[0]
    request = request_for(test_df, i)
    base = client.post("/predict", json=request).json()["probability"]
    request["earlier"] = {k: v + 5.0 for k, v in request["earlier"].items()}
    assert client.post("/predict", json=request).json()["probability"] != pytest.approx(base)


def test_response_shape_and_disclaimer(client):
    body = client.post("/predict", json=VALID).json()
    assert set(body) == {"label", "probability", "risk_band", "message", "thresholds", "top_factors"}
    assert 0.0 <= body["probability"] <= 1.0
    assert body["risk_band"] in {"low", "medium", "high"}
    assert "not a certified safety system" in body["message"]


def test_top_factors_are_the_three_largest(client):
    factors = client.post("/predict", json=VALID).json()["top_factors"]
    assert len(factors) == 3
    magnitudes = [abs(f["contribution"]) for f in factors]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert all((f["contribution"] > 0) == (f["effect"] == "raises risk") for f in factors)


def test_explanation_is_consistent_with_the_prediction(bundle, test_df):
    """SHAP values plus the base value must add up to the model's own output, so the factors describe this prediction."""
    pipeline = bundle["pipeline"]
    explainer = main.shap.TreeExplainer(pipeline.named_steps["classifier"])
    X = test_df.loc[sample_rows(test_df, 1, 2), bundle["features"]]
    contributions = explainer.shap_values(pipeline.named_steps["preprocessing"].transform(X))
    margin = explainer.expected_value + contributions.sum(axis=1)
    np.testing.assert_allclose(1 / (1 + np.exp(-margin)), pipeline.predict_proba(X)[:, 1], atol=1e-5)


@pytest.mark.parametrize("probability, band", [
    (0.0, "low"), (0.389, "low"), (0.39, "medium"), (0.6564, "medium"), (0.6566, "high"), (1.0, "high"),
])
def test_risk_band_edges(bundle, probability, band):
    assert main.risk_band(probability, 0.6565) == band


def test_boundary_inputs_are_accepted(client):
    """Values at zero and exactly at the physical limits should be accepted (not out-of-range)."""
    all_zeros = {"current": {"Current_J0": 0.0, "Current_J1": 0.0, "Current_J2": 0.0,
                             "Current_J3": 0.0, "Current_J4": 0.0, "Current_J5": 0.0,
                             "Tool_current": 0.0},
                 "earlier": {f"Current_J{i}": 0.0 for i in range(6)}}
    at_joint_max = {"current": {"Current_J0": 10.0, "Current_J1": -10.0, "Current_J2": 10.0,
                                "Current_J3": -10.0, "Current_J4": 10.0, "Current_J5": -10.0,
                                "Tool_current": 3.0},
                   "earlier": {f"Current_J{i}": 10.0 for i in range(6)}}
    for body in [all_zeros, at_joint_max]:
        response = client.post("/predict", json=body)
        assert response.status_code == 200
        assert 0.0 <= response.json()["probability"] <= 1.0


def _without(body, section, field):
    copy = {"current": dict(body["current"]), "earlier": dict(body["earlier"])}
    del copy[section][field]
    return copy


def _with(body, section, field, value):
    copy = {"current": dict(body["current"]), "earlier": dict(body["earlier"])}
    copy[section][field] = value
    return copy


@pytest.mark.parametrize("body", [
    _without(VALID, "current", "Current_J2"),
    _without(VALID, "earlier", "Current_J5"),
    {"current": VALID["current"]},
    {"earlier": VALID["earlier"]},
    {},
    _with(VALID, "current", "Current_J0", "high"),
    _with(VALID, "current", "Tool_current", None),
    _with(VALID, "earlier", "Current_J1", [1.0]),
    _with(VALID, "current", "Temperature_J1", 30.0),   # an input the model no longer takes
    VALID["current"],                                  # the old single-reading format
])
def test_invalid_or_missing_inputs_give_a_readable_422(client, body):
    response = client.post("/predict", json=body)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail and all("loc" in d and "msg" in d for d in detail)


@pytest.mark.parametrize("raw", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_numbers_are_rejected(client, raw):
    text = '{"current": {"Current_J0": %s, "Current_J1": 0, "Current_J2": 0, "Current_J3": 0, "Current_J4": 0, ' \
           '"Current_J5": 0, "Tool_current": 0}, "earlier": {"Current_J0": 0, "Current_J1": 0, "Current_J2": 0, ' \
           '"Current_J3": 0, "Current_J4": 0, "Current_J5": 0}}' % raw
    response = client.post("/predict", content=text, headers={"Content-Type": "application/json"})
    assert response.status_code == 422


def test_empty_body_and_bad_json_do_not_crash(client):
    assert client.post("/predict").status_code == 422
    assert client.post("/predict", content="not json", headers={"Content-Type": "application/json"}).status_code == 422


@pytest.mark.parametrize("section", ["current", "earlier"])
@pytest.mark.parametrize("value, ok", [(10.0, True), (-10.0, True), (10.01, False), (-10.01, False)])
def test_joint_current_bounds(client, section, value, ok):
    """Joint current: exactly at ±10.0 A is accepted; beyond that is rejected with 422."""
    body = _with(VALID, section, "Current_J2", value)
    response = client.post("/predict", json=body)
    assert (response.status_code == 200) is ok
    if not ok:
        detail = response.json()["detail"]
        # Error must name the field; it must NOT echo the invalid numeric value in the message text or detail keys.
        locs = [d["loc"] for d in detail]
        assert any("Current_J2" in loc for loc in locs)
        assert all("input" not in d for d in detail)


@pytest.mark.parametrize("value, ok", [(0.0, True), (3.0, True), (3.01, False), (-0.01, False)])
def test_tool_current_bounds(client, value, ok):
    """Tool current: [0.0, 3.0] A is accepted; outside that range is rejected with 422."""
    body = _with(VALID, "current", "Tool_current", value)
    response = client.post("/predict", json=body)
    assert (response.status_code == 200) is ok


def test_missing_model_gives_503(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "MODEL_PATH", tmp_path / "missing.joblib")
    monkeypatch.setattr(main, "model_bundle", {})  # the module-scoped client above has already loaded the model
    with TestClient(main.app) as c:
        assert c.get("/health").json()["model_loaded"] is False
        assert c.post("/predict", json=VALID).status_code == 503
        assert c.get("/model-info").status_code == 503


def test_bundle_with_unsupported_features_fails_at_startup(bundle):
    bad = dict(bundle, features=bundle["features"] + ["Temperature_J1"])
    with pytest.raises(RuntimeError, match="cannot supply"):
        main._validate_bundle(bad)
