# UR3 CobotOps Fault Detection

A protective-stop classifier for the UR3 cobot, built on the UCI CobotOps dataset (7,409 rows, 3.78% positive class). The project covers the full data mining pipeline — data understanding, leakage-aware preprocessing, and model selection with cross-validation — and ships the final model behind a FastAPI backend and a React frontend so operators can get a live risk prediction from sensor readings. Built for the IT3051 group project (4 members, one lane each); see [CLAUDE.md](CLAUDE.md) for the full workflow, ownership model, and schedule.

## Setup

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

Run each notebook top to bottom (*Restart & Run All*) before committing. Shared logic (data loading, feature engineering, the preprocessing pipeline, CV/metrics) lives in `src/` and is imported by the notebooks, not copy-pasted.

## Running the backend

```bash
uvicorn backend.main:app --reload
```

Serves `/predict` and `/health`, loading `models/final_pipeline.joblib` once at startup.

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
