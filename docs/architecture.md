# Architecture

```
              raw_sales.xlsx
                    │
                    ▼
          ┌──────────────────┐
          │  data_prep.py    │  weekly resample + interp
          └─────────┬────────┘
                    ▼
            weekly_clean.csv
                    │
                    ▼
          ┌──────────────────┐
          │  features.py     │  lag/roll/calendar/holiday
          └─────────┬────────┘
                    ▼
          ┌──────────────────────────────────────────┐
          │          train.py (per state)            │
          │                                          │
          │  ┌─────────┐ ┌────────┐ ┌────────┐ ┌────┐│
          │  │ SARIMA  │ │Prophet │ │XGBoost │ │LSTM││
          │  └────┬────┘ └────┬───┘ └────┬───┘ └─┬──┘│
          │       └───────────┴──────────┴───────┘   │
          │            validate (last 8 weeks)        │
          │            pick min(MAPE)                 │
          │            refit on full series           │
          └──────────────────────┬───────────────────┘
                                 ▼
                       artifacts/models.joblib
                                 │
                                 ▼
                       ┌─────────────────┐
                       │  api/main.py    │  FastAPI
                       │  /forecast/...  │
                       └─────────────────┘
```

## Data contract

- Cleaned weekly panel: `state` (str), `ds` (datetime, week-start Monday), `y` (float).
- 43 states × 256 weekly observations after resampling/imputation.

## Model interface contract

```python
class Model:
    name: str
    def fit(self, fit_input) -> "Model": ...
    def forecast(self, n: int) -> np.ndarray: ...
```

`fit_input` differs per model (Series for SARIMA/LSTM, DataFrame[ds,y] for Prophet, full feature panel for XGBoost) but they all return a 1-D `np.ndarray` of length `n` from `forecast(n)`.

## Selection criterion

MAPE (mean absolute percentage error) on a fixed 8-week holdout per state. We use MAPE because:
1. The forecast horizon is the same as our validation horizon (8 weeks).
2. State-level scales differ by 10× — MAPE is scale-invariant.
3. Stakeholders intuitively read percentage errors.

## Inference

The API does not retrain. Each request calls `model.forecast(horizon)`; XGBoost runs the recursive feature-rebuild loop. Latency is <1 s for `horizon=8` on commodity hardware.

## Failure modes & mitigations

| Failure | Mitigation |
|---------|------------|
| SARIMA non-convergence | fall back to ARIMA(1,1,1); if that also fails, model is dropped from candidates |
| Prophet NaN forecast | set MAPE = inf so it cannot win selection |
| XGBoost lag warmup nulls | dropped during training; recursive loop fills them at inference |
| Series too short (<20 weeks) | only SARIMA is trained, no validation, marked NaN |
| State unknown to API | 404 with the available state list |
