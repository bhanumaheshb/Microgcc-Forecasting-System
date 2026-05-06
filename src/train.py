"""Training driver. For each state we fit all four models, score on the last 8
weeks of the training data, and persist the winner under `artifacts/`."""
from __future__ import annotations

import sys
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data_prep
from src.features import make_features, time_split, FEATURE_COLS
from src.models import (
    SarimaModel, ProphetModel, XgbModel, LstmModel,
    mape, StateResult,
)

ART_DIR = ROOT / "artifacts"
ART_DIR.mkdir(exist_ok=True)
VAL_WEEKS = 8


def _safe_fit_score(model_cls, fit_input, val_y) -> tuple[object, float]:
    try:
        m = model_cls()
        m.fit(fit_input)
        pred = m.forecast(len(val_y))
        return m, mape(val_y.values, pred)
    except Exception as e:
        print(f"      {model_cls.__name__} failed: {e}")
        return None, float("inf")


def train_state(state: str, panel_state: pd.DataFrame) -> StateResult:
    """Train all 4 models on `panel_state` (already feature-engineered for XGB)."""
    panel_state = panel_state.sort_values("ds").reset_index(drop=True)
    if len(panel_state) <= VAL_WEEKS + 12:
        # Not enough rows to hold out; skip validation, fit ARIMA only.
        train_y = panel_state["y"]
        m = SarimaModel().fit(train_y)
        return StateResult(
            state=state, metrics={"sarima": float("nan")},
            best_model="sarima", best_mape=float("nan"),
            artifact=m, last_train_ds=panel_state["ds"].max(),
        )

    train_part = panel_state.iloc[:-VAL_WEEKS]
    val_part = panel_state.iloc[-VAL_WEEKS:]

    train_y = train_part["y"].reset_index(drop=True)
    val_y = val_part["y"].reset_index(drop=True)

    metrics = {}
    candidates = {}

    # SARIMA
    m, sc = _safe_fit_score(SarimaModel, train_y, val_y)
    metrics["sarima"] = sc; candidates["sarima"] = m

    # Prophet
    def _prophet_fit_input(): return train_part[["ds", "y"]]
    try:
        mp = ProphetModel().fit(_prophet_fit_input())
        pred = mp.forecast(len(val_y))
        metrics["prophet"] = mape(val_y.values, pred); candidates["prophet"] = mp
    except Exception as e:
        print(f"      prophet failed: {e}")
        metrics["prophet"] = float("inf"); candidates["prophet"] = None

    # XGBoost (needs feature panel)
    try:
        mx = XgbModel().fit(train_part)
        pred = mx.forecast(len(val_y))
        metrics["xgboost"] = mape(val_y.values, pred); candidates["xgboost"] = mx
    except Exception as e:
        print(f"      xgboost failed: {e}")
        metrics["xgboost"] = float("inf"); candidates["xgboost"] = None

    # LSTM
    m, sc = _safe_fit_score(LstmModel, train_y, val_y)
    metrics["lstm"] = sc; candidates["lstm"] = m

    # pick winner
    best_name = min(metrics, key=lambda k: metrics[k] if not np.isnan(metrics[k]) else float("inf"))
    best_mape = metrics[best_name]

    # Refit winner on the FULL series so production forecasts use all data.
    full_panel = panel_state
    full_y = full_panel["y"].reset_index(drop=True)
    if best_name == "sarima":
        final = SarimaModel().fit(full_y)
    elif best_name == "prophet":
        final = ProphetModel().fit(full_panel[["ds", "y"]])
    elif best_name == "xgboost":
        final = XgbModel().fit(full_panel)
    else:
        final = LstmModel().fit(full_y)

    return StateResult(
        state=state,
        metrics=metrics,
        best_model=best_name,
        best_mape=best_mape,
        artifact=final,
        last_train_ds=full_panel["ds"].max(),
    )


def main(states_subset: list[str] | None = None):
    df = pd.read_csv(data_prep.CLEAN_PATH, parse_dates=["ds"])
    panel = make_features(df)

    states = states_subset or sorted(panel["state"].unique())
    print(f"Training on {len(states)} states; validating on last {VAL_WEEKS} weeks.")
    print("-" * 70)

    summary_rows = []
    artifacts: dict[str, StateResult] = {}
    t0 = time.time()
    for i, st in enumerate(states, 1):
        sub = panel[panel["state"] == st].copy()
        t_start = time.time()
        res = train_state(st, sub)
        artifacts[st] = res
        elapsed = time.time() - t_start
        print(
            f"[{i:2d}/{len(states)}] {st:15s} -> best={res.best_model:8s} "
            f"mape={res.best_mape:6.2f}%  "
            f"(sarima={res.metrics.get('sarima', float('nan')):.2f} "
            f"prophet={res.metrics.get('prophet', float('nan')):.2f} "
            f"xgb={res.metrics.get('xgboost', float('nan')):.2f} "
            f"lstm={res.metrics.get('lstm', float('nan')):.2f}) "
            f"[{elapsed:.1f}s]"
        )
        row = {"state": st, "best_model": res.best_model, "best_mape": res.best_mape}
        row.update({f"mape_{k}": v for k, v in res.metrics.items()})
        summary_rows.append(row)

    print("-" * 70)
    print(f"Done in {time.time() - t0:.1f}s")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(ART_DIR / "training_summary.csv", index=False)
    joblib.dump(artifacts, ART_DIR / "models.joblib")
    print(f"Saved: {ART_DIR / 'models.joblib'} + training_summary.csv")
    print("\nModel-selection counts:")
    print(summary["best_model"].value_counts().to_string())
    print(f"\nMean MAPE across states: {summary['best_mape'].mean():.2f}%")
    print(f"Median MAPE: {summary['best_mape'].median():.2f}%")


if __name__ == "__main__":
    subset = None
    if len(sys.argv) > 1:
        subset = sys.argv[1].split(",")
    main(subset)
