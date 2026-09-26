# UR3 CobotOps Fault Detection

A protective-stop classifier for the UR3 cobot, built on the UCI CobotOps dataset (7,409 rows, 3.78% positive class). The project covers the full data mining pipeline — data understanding, leakage-aware preprocessing, and model selection with cross-validation — and ships the final model behind a FastAPI backend and a React frontend so operators can get a live risk prediction from sensor readings. Built for the IT3051 group project (4 members, one lane each); see [CLAUDE.md](CLAUDE.md) for the full workflow, ownership model, and schedule.

## Where the project stands (25 Sep 2026)

- **Final model:** XGBoost on 9 features (the six joint currents, the tool current, their summed absolute value, and how that sum changed over the last 3 readings), alert threshold 0.6565. On cross-validated PR-AUC it is a close call with Random Forest (notebook 07).
- **Held-out result** (later cycles, evaluated once): 84.3% of the 51 stop episodes flagged at least once, recall 0.648, precision 0.519, about 27 false-alarm runs per hour. It detects stops as they begin and gives little advance warning (notebook 07, section 10).
- **Not a safety system:** a decision-support tool built from one robot and about 2 hours of readings. It does not replace the robot's certified safety functions.
- **Where the reasoning is:** every notebook ends with a decision log (what we decided, the evidence, the alternative, why it was rejected).

## Setup

**Prerequisites:** Python **3.11** (tested from a clean install; the pinned `numpy==1.26.4` has no prebuilt wheel for Python 3.13) and Node.js 18 or newer for the frontend. On Windows, clone into a short path (for example `C:\dev\`): a very long folder path can make `pip install` fail with a long-path error.

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt
```

## Running the notebooks

```bash
jupyter notebook notebooks/
```

Run the notebooks in order: 03 saves the split, 04 saves the processed train/test data and the final feature list, 05 and 06 use them, and 07 saves `models/final_pipeline.joblib`. Everything is seeded (`RANDOM_STATE = 42`), so a full re-run reproduces the committed results. Run each notebook top to bottom (*Restart & Run All*) before committing. Shared logic (data loading, feature engineering, the preprocessing pipeline, CV/metrics) lives in `src/` and is imported by the notebooks, not copy-pasted.

## Running the backend

```bash
uvicorn backend.main:app --reload
```

Serves `/predict`, `/model-info` and `/health`, loading `models/final_pipeline.joblib` once at startup.

The model uses how much the total joint current changed over the last 3 readings, so `/predict` takes the latest reading plus the joint currents from the reading 3 steps earlier in the same working cycle (the API cannot check that the two are consecutive):

```json
{
  "current": {"Current_J0": 0.084, "Current_J1": -1.692, "Current_J2": -0.724, "Current_J3": -0.694,
              "Current_J4": -0.031, "Current_J5": 0.068, "Tool_current": 0.082},
  "earlier": {"Current_J0": 0.129, "Current_J1": -1.975, "Current_J2": -0.999, "Current_J3": -0.574,
              "Current_J4": -0.004, "Current_J5": -0.01}
}
```

It returns the label, stop probability, risk band (low / medium / high), a message, the two thresholds, and the three inputs that moved the prediction most (SHAP). Invalid or missing inputs get a `422` that names the field.

Tests (parity with the notebook on real test rows, validation, risk bands, explanations):

```bash
python -m pytest backend/tests
```

## Running the frontend

```bash
cd frontend
npm install
npm run dev
```

Vite dev server on `http://localhost:5173`, calling the backend `/predict` endpoint (`http://localhost:8000` by default — see `frontend/.env.example`) and displaying a risk level and explanation. The backend must allow CORS from the dev server origin.

## Project layout

```
data/raw/          # original dataset, tracked in git as evidence
data/processed/    # generated train/test splits, not evidence-critical
notebooks/         # 01-07, one per pipeline phase
src/                # shared code: config, features, pipeline, evaluate
models/            # saved final_pipeline.joblib
backend/           # FastAPI app
frontend/          # React (Vite) app
reports/figures/   # figures reused by the report and slides
```
