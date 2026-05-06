"""Forecasting models: ARIMA/SARIMA, Prophet, XGBoost, LSTM.

Each model is wrapped in a uniform interface so the training driver can fit,
score (MAPE on a fixed validation window), and produce an `n_steps` forecast
without knowing the model internals.

Design notes
- Each model is fit *per state* to keep the case study scope tractable. The
  XGBoost and LSTM models could be trained globally with state as a feature,
  but per-state fitting eliminates the dominance issue caused by California
  being ~10x larger than smaller states.
- We forecast in log-space and exponentiate, which keeps multiplicative growth
  reasonable and ensures non-negative predictions.
"""
from __future__ import annotations

import os
# Must be set BEFORE importing torch / xgboost — both ship their own libomp
# on macOS and double-loading triggers a SIGSEGV. Setting these here makes
# the package safe to import in any order.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import warnings
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any
import contextlib

# Silence prophet's noisy stdout/stderr.
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

from . import features as feat_mod  # type: ignore  # noqa
from .features import FEATURE_COLS, LAGS, ROLL_WINDOWS, make_features

# ---------- metric ----------

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > 1e-9
    if not mask.any():
        return float("inf")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# ---------- ARIMA / SARIMA ----------

class SarimaModel:
    name = "sarima"

    def __init__(self):
        self.fit_obj = None
        self.last_y = None

    def fit(self, train_y: pd.Series):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        # weekly data: yearly seasonality m=52 is too costly; use m=4 (monthly-ish)
        # plus a non-seasonal ARIMA(1,1,1) backbone.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = SARIMAX(
                    train_y.values,
                    order=(1, 1, 1),
                    seasonal_order=(1, 0, 1, 4),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                self.fit_obj = model.fit(disp=False, maxiter=80)
            except Exception:
                model = SARIMAX(train_y.values, order=(1, 1, 1))
                self.fit_obj = model.fit(disp=False, maxiter=80)
        self.last_y = train_y
        return self

    def forecast(self, n: int) -> np.ndarray:
        f = self.fit_obj.forecast(steps=n)
        return np.asarray(f, dtype=float)


# ---------- Prophet ----------

class ProphetModel:
    name = "prophet"

    def __init__(self):
        self.model = None

    def fit(self, train_df: pd.DataFrame):
        from prophet import Prophet
        df = train_df[["ds", "y"]].copy()
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.1,
        )
        # Note: do NOT use contextlib.redirect_stdout here — it conflicts with
        # cmdstanpy's subprocess handling on macOS and can deadlock when
        # combined with torch in the same process.
        m.fit(df)
        self.model = m
        self.last_ds = df["ds"].max()
        return self

    def forecast(self, n: int) -> np.ndarray:
        future = self.model.make_future_dataframe(periods=n, freq="W-MON", include_history=False)
        fcst = self.model.predict(future)
        return fcst["yhat"].values.astype(float)


# ---------- XGBoost (recursive) ----------

class XgbModel:
    name = "xgboost"

    def __init__(self):
        self.model = None
        self.history = None  # full panel up to train cutoff (for recursive forecast)

    def fit(self, train_panel: pd.DataFrame):
        import xgboost as xgb
        # Drop rows with NaN lags (warmup period)
        train = train_panel.dropna(subset=FEATURE_COLS).copy()
        X = train[FEATURE_COLS].values
        y = np.log1p(train["y"].values)
        self.model = xgb.XGBRegressor(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            verbosity=0,
            n_jobs=2,
        )
        self.model.fit(X, y)
        self.history = train_panel.copy()
        return self

    def forecast(self, n: int) -> np.ndarray:
        """Recursive multi-step forecast for ONE state's panel."""
        history = self.history.copy().sort_values("ds").reset_index(drop=True)
        preds = []
        last_ds = history["ds"].iloc[-1]
        state = history["state"].iloc[-1]
        for step in range(n):
            new_ds = last_ds + pd.Timedelta(weeks=1)
            new_row = {"state": state, "ds": new_ds, "y": np.nan}
            history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)
            history = make_features(history)
            row = history.iloc[[-1]]
            x = row[FEATURE_COLS].values
            yhat_log = self.model.predict(x)[0]
            yhat = float(np.expm1(yhat_log))
            history.loc[history.index[-1], "y"] = yhat
            preds.append(yhat)
            last_ds = new_ds
        return np.array(preds, dtype=float)


try:
    import torch.nn as _nn

    class _LstmNet(_nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = _nn.LSTM(input_size=1, hidden_size=32, num_layers=1, batch_first=True)
            self.fc = _nn.Linear(32, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])
except ImportError:
    _LstmNet = None  # type: ignore


# ---------- LSTM ----------

class LstmModel:
    name = "lstm"
    SEQ_LEN = 12

    def __init__(self):
        self.net = None
        self.mu = 0.0
        self.sd = 1.0
        self.history_log = None  # 1D np.array of log-y train values

    def fit(self, train_y: pd.Series):
        import torch
        torch.set_num_threads(1)

        y_log = np.log1p(train_y.values.astype(float))
        if len(y_log) < self.SEQ_LEN + 4:
            self.net = None  # too short — fallback handled by trainer
            return self
        self.mu = float(y_log.mean())
        self.sd = float(y_log.std() + 1e-9)
        z = (y_log - self.mu) / self.sd

        Xs, Ys = [], []
        for i in range(len(z) - self.SEQ_LEN):
            Xs.append(z[i : i + self.SEQ_LEN])
            Ys.append(z[i + self.SEQ_LEN])
        X = torch.tensor(np.array(Xs), dtype=torch.float32).unsqueeze(-1)
        Y = torch.tensor(np.array(Ys), dtype=torch.float32).unsqueeze(-1)

        torch.manual_seed(13)
        net = _LstmNet()
        import torch.nn as nn
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        loss_fn = nn.MSELoss()
        net.train()
        for epoch in range(120):
            opt.zero_grad()
            pred = net(X)
            loss = loss_fn(pred, Y)
            loss.backward()
            opt.step()
        net.eval()
        self.net = net
        self.history_log = y_log
        return self

    def forecast(self, n: int) -> np.ndarray:
        import torch
        if self.net is None:
            return np.full(n, float(np.expm1(self.history_log[-1] if self.history_log is not None else 0.0)))
        seq = list(self.history_log[-self.SEQ_LEN :])
        preds = []
        for _ in range(n):
            z = [(v - self.mu) / self.sd for v in seq[-self.SEQ_LEN :]]
            x = torch.tensor(z, dtype=torch.float32).view(1, self.SEQ_LEN, 1)
            with torch.no_grad():
                z_hat = self.net(x).item()
            y_log_hat = z_hat * self.sd + self.mu
            seq.append(y_log_hat)
            preds.append(float(np.expm1(y_log_hat)))
        return np.array(preds, dtype=float)


@dataclass
class StateResult:
    state: str
    metrics: dict[str, float] = field(default_factory=dict)
    best_model: str = ""
    best_mape: float = float("inf")
    artifact: Any = None  # the chosen fitted model object
    last_train_ds: pd.Timestamp | None = None
