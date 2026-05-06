"""Feature engineering for the supervised models (XGBoost, LSTM).

The case study asks for `t-1, t-7, t-30` lags. Our series is weekly, so we map
those calendar-day intervals to the nearest weekly equivalents:
  t-1   -> lag 1 week
  t-7   -> lag 1 week (kept once)
  t-30  -> lag 4 weeks
We additionally include t-2, t-12, t-52 for short-term, quarterly and yearly
seasonality. Rolling mean / std (4 and 12 week windows) capture trend/volatility.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import holidays

US_HOLIDAYS = holidays.country_holidays("US")
LAGS = [1, 2, 4, 8, 12, 52]
ROLL_WINDOWS = [4, 12]
FEATURE_COLS: list[str] = []  # populated by make_features


def _calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ds = pd.to_datetime(df["ds"])
    df["dow"] = ds.dt.dayofweek
    df["month"] = ds.dt.month
    df["quarter"] = ds.dt.quarter
    df["weekofyear"] = ds.dt.isocalendar().week.astype(int)
    df["year"] = ds.dt.year
    # holiday flag = 1 if any day of the ISO week falls on a US federal holiday
    def week_has_holiday(d):
        for k in range(7):
            if (d + pd.Timedelta(days=k)) in US_HOLIDAYS:
                return 1
        return 0
    df["is_holiday_week"] = ds.apply(week_has_holiday).astype(int)
    return df


def make_features(panel: pd.DataFrame) -> pd.DataFrame:
    """`panel` columns: state, ds, y. Returns enriched panel with lag/roll/cal feats.

    Lags and rolls are computed per-state to avoid information leakage across
    series. Rolling features use shift(1) so they only see the past.
    """
    panel = panel.sort_values(["state", "ds"]).copy()
    g = panel.groupby("state", group_keys=False)
    for lag in LAGS:
        panel[f"lag_{lag}"] = g["y"].shift(lag)
    for w in ROLL_WINDOWS:
        shifted = g["y"].shift(1)
        panel[f"rmean_{w}"] = shifted.groupby(panel["state"]).transform(
            lambda s: s.rolling(window=w, min_periods=1).mean()
        )
        panel[f"rstd_{w}"] = shifted.groupby(panel["state"]).transform(
            lambda s: s.rolling(window=w, min_periods=2).std()
        )
    panel = _calendar_features(panel)
    feature_cols = (
        [f"lag_{l}" for l in LAGS]
        + [f"rmean_{w}" for w in ROLL_WINDOWS]
        + [f"rstd_{w}" for w in ROLL_WINDOWS]
        + ["dow", "month", "quarter", "weekofyear", "year", "is_holiday_week"]
    )
    FEATURE_COLS[:] = feature_cols
    return panel


def time_split(panel: pd.DataFrame, val_weeks: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-state holdout: last `val_weeks` rows are validation, earlier rows train."""
    panel = panel.sort_values(["state", "ds"])
    train, val = [], []
    for _, g in panel.groupby("state"):
        if len(g) <= val_weeks + 12:
            train.append(g)
            continue
        train.append(g.iloc[:-val_weeks])
        val.append(g.iloc[-val_weeks:])
    train_df = pd.concat(train).reset_index(drop=True)
    val_df = pd.concat(val).reset_index(drop=True) if val else panel.iloc[0:0]
    return train_df, val_df


if __name__ == "__main__":
    import data_prep  # noqa
    df = pd.read_csv(data_prep.CLEAN_PATH, parse_dates=["ds"])
    feats = make_features(df)
    tr, vl = time_split(feats, 8)
    print(f"train rows={len(tr)}, val rows={len(vl)}, features={len(FEATURE_COLS)}")
    print(FEATURE_COLS)
