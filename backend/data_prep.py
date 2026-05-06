"""Data ingestion + preprocessing.

Raw file is irregular per-state weekly-ish sales (Beverages). We resample to a
strict weekly grid (W-MON anchor) per state, log-transform totals to stabilise
variance, and impute remaining gaps via time-aware linear interpolation.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw_sales.xlsx"
CLEAN_PATH = Path(__file__).resolve().parents[1] / "data" / "weekly_clean.csv"

WEEK_RULE = "W-MON"


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_excel(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "State", "Total"]).copy()
    df["Total"] = df["Total"].astype(float)
    return df


def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Per-state weekly resample with linear interpolation + edge ffill/bfill."""
    out_frames = []
    for state, g in df.groupby("State"):
        g = g.sort_values("Date").set_index("Date")
        weekly = g["Total"].resample(WEEK_RULE).sum(min_count=1)
        # Replace zeros (caused by sum of empty bins) with NaN, then interpolate.
        weekly = weekly.where(weekly > 0)
        weekly = weekly.interpolate(method="time", limit_direction="both")
        weekly = weekly.ffill().bfill()
        sub = weekly.reset_index().rename(columns={"Total": "y", "Date": "ds"})
        sub["state"] = state
        out_frames.append(sub)
    out = pd.concat(out_frames, ignore_index=True)
    out = out.sort_values(["state", "ds"]).reset_index(drop=True)
    return out


def build_clean(write: bool = True) -> pd.DataFrame:
    raw = load_raw()
    weekly = to_weekly(raw)
    if write:
        CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
        weekly.to_csv(CLEAN_PATH, index=False)
    return weekly


if __name__ == "__main__":
    df = build_clean()
    print(f"Built weekly dataset: {df.shape}, states={df['state'].nunique()}, "
          f"date range {df['ds'].min().date()} -> {df['ds'].max().date()}")
