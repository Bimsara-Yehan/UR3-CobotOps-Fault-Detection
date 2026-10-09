"""Export system-test evidence for the technical report (section 13).

Exercises the real /predict endpoint (via FastAPI's TestClient, no server needed) with the
categories the project plan calls for -- valid, invalid, missing, boundary, and known-positive
cases -- and records expected vs actual for each. Known-positive/negative cases are drawn from
real rows of the held-out test split and checked against the saved model bundle directly, so
"expected" is never hand-typed for those rows.

Run from the repo root:
    python scripts/export_test_evidence.py

Writes reports/test_evidence.md (the report-ready table) and reports/test_evidence.csv (for an
appendix or spreadsheet).
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
from fastapi.testclient import TestClient

import backend.main as backend_main
from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.features import CURRENT_COLS

RAW_FEATURES = CURRENT_COLS + ["Tool_current"]
rows = []


def error_summary(response):
    """A short, report-friendly rendering of a 422/503 error body: "field: message" per error."""
    detail = response.json().get("detail")
    if isinstance(detail, list):
        return "; ".join(f"{'.'.join(str(p) for p in e['loc'][1:])}: {e['msg']}" for e in detail)
    return str(detail)


def record(category, description, input_summary, expected, status, body, ok):
    rows.append({
        "Category": category,
        "Case": description,
        "Input": input_summary,
        "Expected": expected,
        "Actual": f"HTTP {status}: {body}",
        "Result": "PASS" if ok else "FAIL",
    })
    flag = "PASS" if ok else "FAIL *** FAIL ***"
    print(f"[{flag}] {category:14s} {description}")


def request_for(df, i, dp=None):
    """Build a /predict body from row i. dp=None sends exact values (for strict notebook-parity
    checks); dp=3 rounds to 3 decimals (what a UI reasonably sends) for the other test categories.
    """
    rnd = (lambda v: v) if dp is None else (lambda v: round(v, dp))
    return {
        "current": {k: rnd(float(df.loc[i, k])) for k in RAW_FEATURES},
        "earlier": {k: rnd(float(df.loc[i - 3, k])) for k in CURRENT_COLS},
    }


def main():
    bundle = joblib.load(MODELS_DIR / "final_pipeline.joblib")
    test_df = pd.read_parquet(PROCESSED_DATA_DIR / "test.parquet").reset_index(drop=True)
    position = test_df.groupby("cycle").cumcount()
    eligible = test_df[position >= 3].copy()  # rows with 3 prior readings in the same cycle

    with TestClient(backend_main.app) as client:
        # ---- known-positive / known-negative: real rows, checked against the saved model ----
        stop_rows = eligible[eligible["Robot_ProtectiveStop"] == 1].index[:3]
        normal_rows = eligible[eligible["Robot_ProtectiveStop"] == 0].index[:2]
        for i in list(stop_rows) + list(normal_rows):
            body = request_for(test_df, i)  # exact values: this category asserts exact notebook parity
            notebook_p = float(bundle["pipeline"].predict_proba(test_df.loc[[i], bundle["features"]])[0, 1])
            notebook_label = int(notebook_p >= bundle["threshold"])
            r = client.post("/predict", json=body)
            api = r.json()
            ok = r.status_code == 200 and api["label"] == notebook_label and abs(api["probability"] - notebook_p) < 1e-6
            truth = "stop" if i in stop_rows else "normal"
            record(
                "Known-positive" if truth == "stop" else "Valid",
                f"real test-set row, ground truth = {truth} (row {i})",
                f"reading + history from test.parquet row {i}",
                f"200, label={notebook_label}, p={notebook_p:.6f} (matches notebook pipeline)",
                r.status_code,
                f"label={api.get('label')}, p={api.get('probability', 0):.6f}, band={api.get('risk_band')}",
                ok,
            )

        valid_body = request_for(test_df, eligible.index[0], dp=3)  # realistic rounded input for the rest

        # ---- invalid ----
        bad = {**valid_body, "current": {**valid_body["current"], "Current_J0": 99.0}}
        r = client.post("/predict", json=bad)
        record("Invalid", "joint current beyond physical range (99 A)", "Current_J0 = 99.0", "422, field named, value not echoed",
               r.status_code, error_summary(r), r.status_code == 422 and "99" not in error_summary(r))

        nan_body = {**valid_body, "current": {**valid_body["current"], "Current_J0": None}}
        import json as _json
        nan_text = _json.dumps(nan_body).replace("null", "NaN")
        r = client.post("/predict", content=nan_text, headers={"Content-Type": "application/json"})
        record("Invalid", "non-finite value (NaN)", "Current_J0 = NaN", "422 (not a 500 crash)", r.status_code,
               error_summary(r), r.status_code == 422)

        # ---- missing ----
        missing_field = {"current": {k: v for k, v in valid_body["current"].items() if k != "Current_J2"},
                         "earlier": valid_body["earlier"]}
        r = client.post("/predict", json=missing_field)
        record("Missing", "required field omitted (Current_J2)", "current.Current_J2 absent", "422, names the missing field",
               r.status_code, error_summary(r), r.status_code == 422)

        r = client.post("/predict", json={})
        record("Missing", "empty request body", "{}", "422", r.status_code, error_summary(r), r.status_code == 422)

        # ---- boundary ----
        at_max = {**valid_body, "current": {**valid_body["current"], "Current_J0": 10.0}}
        r = client.post("/predict", json=at_max)
        record("Boundary", "joint current exactly at +10 A limit", "Current_J0 = 10.0", "200 (accepted)",
               r.status_code, f"label={r.json().get('label') if r.status_code == 200 else r.json().get('detail')}", r.status_code == 200)

        at_min = {**valid_body, "current": {**valid_body["current"], "Current_J0": -10.0}}
        r = client.post("/predict", json=at_min)
        record("Boundary", "joint current exactly at -10 A limit", "Current_J0 = -10.0", "200 (accepted)",
               r.status_code, f"label={r.json().get('label') if r.status_code == 200 else r.json().get('detail')}", r.status_code == 200)

        beyond = {**valid_body, "current": {**valid_body["current"], "Current_J0": 10.01}}
        r = client.post("/predict", json=beyond)
        record("Boundary", "joint current just beyond limit (10.01 A)", "Current_J0 = 10.01", "422 (rejected)",
               r.status_code, error_summary(r), r.status_code == 422)

    # ---- model-not-loaded failure mode ----
    import importlib
    importlib.reload(backend_main)
    backend_main.MODEL_PATH = Path("models/does_not_exist.joblib")
    with TestClient(backend_main.app) as client:
        r = client.get("/health")
        health_ok = r.json()["model_loaded"] is False
        r2 = client.post("/predict", json=valid_body)
        record("Missing model", "model file absent at startup", "N/A (MODEL_PATH points nowhere)", "503, health reports model_loaded=false",
               r2.status_code, f"health.model_loaded={r.json()['model_loaded']}, predict -> {r2.status_code}", r2.status_code == 503 and health_ok)

    # ---- write out ----
    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "test_evidence.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["Category", "Case", "Input", "Expected", "Actual", "Result"])
        w.writeheader()
        w.writerows(rows)

    n_fail = sum(1 for r in rows if r["Result"] == "FAIL")
    with open(out_dir / "test_evidence.md", "w", encoding="utf-8") as f:
        f.write("# System test evidence\n\n")
        f.write(f"Generated by `scripts/export_test_evidence.py` against the live FastAPI app (in-process, "
                f"via TestClient) and the saved `models/final_pipeline.joblib`. {len(rows)} cases, "
                f"{len(rows) - n_fail} passed, {n_fail} failed.\n\n")
        f.write("| Category | Case | Input | Expected | Actual | Result |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['Category']} | {r['Case']} | {r['Input']} | {r['Expected']} | {r['Actual']} | **{r['Result']}** |\n")

    print(f"\n{len(rows)} cases, {len(rows) - n_fail} passed, {n_fail} failed.")
    print(f"Wrote {out_dir / 'test_evidence.md'} and {out_dir / 'test_evidence.csv'}")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
