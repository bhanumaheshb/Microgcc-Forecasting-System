"""FastAPI service for multi-state weekly sales forecasts.

Endpoints
- GET  /health                           liveness check + model count
- GET  /states                           list of supported states + chosen model
- GET  /forecast/{state}?horizon=8       forecast next N weeks for a state
- POST /forecast                         batch forecast (list of states)
- GET  /metrics                          per-state validation MAPE table

The service loads `artifacts/models.joblib` once at startup. Forecasts are
produced by replaying the chosen model's `forecast(n)` method.
"""
from __future__ import annotations

import os
# Silence the harmless "Importing plotly failed" notice from prophet, and avoid
# libomp double-loading between torch and xgboost on macOS.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import logging
logging.getLogger("prophet.plot").setLevel(logging.CRITICAL)

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ART = ROOT / "artifacts" / "models.joblib"
SUMMARY = ROOT / "artifacts" / "training_summary.csv"


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load()
    yield


app = FastAPI(
    title="Sales Forecasting API",
    version="1.0.0",
    description="Per-state weekly Beverages sales forecasts. "
                "Best of {SARIMA, Prophet, XGBoost, LSTM} chosen per state by MAPE.",
    lifespan=lifespan,
)


class ForecastPoint(BaseModel):
    week_start: str
    yhat: float


class ForecastResponse(BaseModel):
    state: str
    model: str
    horizon_weeks: int
    last_train_week: str
    forecast: List[ForecastPoint]


class BatchForecastRequest(BaseModel):
    states: List[str] = Field(..., min_length=1, description="State names to forecast.")
    horizon: int = Field(8, ge=1, le=52, description="Number of weeks to forecast.")


class BatchForecastResponse(BaseModel):
    horizon_weeks: int
    results: List[ForecastResponse]


_state_results: dict = {}
_summary_df: pd.DataFrame | None = None
_history_df: pd.DataFrame | None = None
HISTORY_PATH = ROOT / "data" / "weekly_clean.csv"


def _load():
    global _state_results, _summary_df, _history_df
    if not ART.exists():
        raise RuntimeError(
            f"Artifacts not found at {ART}. Run `python -m backend.train` first."
        )
    _state_results = joblib.load(ART)
    if SUMMARY.exists():
        _summary_df = pd.read_csv(SUMMARY)
    if HISTORY_PATH.exists():
        _history_df = pd.read_csv(HISTORY_PATH, parse_dates=["ds"])


def _forecast_state(state: str, horizon: int) -> ForecastResponse:
    if state not in _state_results:
        raise HTTPException(
            status_code=404,
            detail=f"State '{state}' not in trained model registry. "
                   f"Available: {sorted(_state_results.keys())[:5]}...",
        )
    res = _state_results[state]
    yhat = res.artifact.forecast(horizon)
    yhat = np.clip(np.asarray(yhat, dtype=float), a_min=0.0, a_max=None)
    last_ds = pd.to_datetime(res.last_train_ds)
    points = [
        ForecastPoint(
            week_start=(last_ds + pd.Timedelta(weeks=i + 1)).date().isoformat(),
            yhat=round(float(v), 2),
        )
        for i, v in enumerate(yhat)
    ]
    return ForecastResponse(
        state=state,
        model=res.best_model,
        horizon_weeks=horizon,
        last_train_week=last_ds.date().isoformat(),
        forecast=points,
    )


@app.get("/health")
def health():
    return {"status": "ok", "states_loaded": len(_state_results)}


@app.get("/states")
def states():
    return [
        {"state": s, "best_model": r.best_model,
         "validation_mape": None if r.best_mape != r.best_mape else round(r.best_mape, 3)}
        for s, r in sorted(_state_results.items())
    ]


@app.get("/metrics")
def metrics():
    if _summary_df is None:
        raise HTTPException(404, "training_summary.csv missing")
    return _summary_df.replace({np.nan: None}).to_dict(orient="records")


@app.get("/forecast/{state}", response_model=ForecastResponse)
def forecast_state(state: str, horizon: int = Query(8, ge=1, le=52)):
    return _forecast_state(state, horizon)


@app.get("/history/{state}")
def history(state: str, weeks: int = Query(52, ge=4, le=400)):
    if _history_df is None:
        raise HTTPException(404, "history not loaded")
    sub = _history_df[_history_df["state"] == state].sort_values("ds").tail(weeks)
    if sub.empty:
        raise HTTPException(404, f"no history for state {state!r}")
    return {
        "state": state,
        "weeks": int(len(sub)),
        "history": [
            {"week_start": d.date().isoformat(), "y": round(float(v), 2)}
            for d, v in zip(sub["ds"], sub["y"])
        ],
    }


@app.post("/forecast", response_model=BatchForecastResponse)
def forecast_batch(req: BatchForecastRequest):
    results = [_forecast_state(s, req.horizon) for s in req.states]
    return BatchForecastResponse(horizon_weeks=req.horizon, results=results)


FRONTEND_DIR = ROOT / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(str(FRONTEND_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
