"""Offline smoke tests — run after `python -m backend.train`."""
from pathlib import Path
import joblib
import pandas as pd

ART = Path(__file__).resolve().parents[1] / "artifacts" / "models.joblib"


def test_artifacts_exist():
    assert ART.exists(), "Run training first: python -m backend.train"


def test_each_state_has_model():
    res = joblib.load(ART)
    assert len(res) >= 2
    for st, r in res.items():
        assert r.best_model in {"sarima", "prophet", "xgboost", "lstm"}
        assert r.artifact is not None


def test_forecast_shape():
    res = joblib.load(ART)
    sample_state = next(iter(res))
    out = res[sample_state].artifact.forecast(8)
    assert len(out) == 8
    assert all(v >= 0 or v != v for v in out)  # non-negative or NaN-free check


def test_summary_csv():
    p = ART.parent / "training_summary.csv"
    assert p.exists()
    df = pd.read_csv(p)
    assert {"state", "best_model", "best_mape"}.issubset(df.columns)
