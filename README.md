# End-to-End Time Series Forecasting System

<p align="center">
  <em>A production-ready forecasting system that trains multiple algorithms, selects the best model per state, and serves predictions via a REST API and an Interactive Dashboard.</em>
</p>

## 📖 About the Project

This project implements an end-to-end machine learning pipeline to forecast the **next 8 weeks of sales for US states**. It is designed with clean architecture principles, treating the machine learning models as a scalable backend service rather than just a set of notebooks.

**Key Features:**
- **Automated Model Selection**: Trains 4 distinct model families per state (ARIMA/SARIMA, Facebook Prophet, XGBoost, LSTM) and automatically selects the champion model based on validation MAPE.
- **Robust Feature Engineering**: Implements leakage-free lag features (t-1 to t-52), rolling statistics, calendar metadata, and US federal holiday flags.
- **REST API Serving**: Exposes the trained models and their predictions via a FastAPI-powered REST interface.
- **Interactive Console**: A modern Node.js/Vite frontend dashboard to visually explore the forecast and evaluation metrics per state.

## 🏗️ Architecture

The system follows a modular, clean architecture separating data processing, model training, and API serving.

```text
forecasting_system/
├── data/                  # Raw datasets and processed outputs
├── src/                   # Core Business Logic & ML Pipeline
│   ├── data_prep.py       # Ingestion, weekly resampling, and imputation
│   ├── features.py        # Leakage-free feature engineering (lags, rolling stats)
│   ├── models.py          # Wrappers for SARIMA, Prophet, XGBoost, and LSTM
│   └── train.py           # Training loop, evaluation, and model selection
├── api/                   # Presentation Layer (API & Serving)
│   └── main.py            # FastAPI service exposing predictions
├── frontend/              # User Interface (Node.js / Vite)
│   ├── package.json       # Frontend dependencies and scripts
│   └── src/               # React/Vue source code for the dashboard
├── artifacts/             # Serialized best models and training summary
└── tests/                 # Automated testing
```

## 🚀 Setup Instructions

### Prerequisites
- Python 3.10+
- Node.js (v18+) and npm
- macOS / Linux / Windows (macOS users may need `brew install libomp` for XGBoost support)

### Backend Installation
1. Navigate to the project directory.
2. Install the required Python dependencies:
```bash
pip install -r requirements.txt
```

### Frontend Installation
1. Navigate to the frontend directory:
```bash
cd frontend
```
2. Install Node dependencies:
```bash
npm install
```

## ⚙️ How to Run

### 1. Data Preparation
Prepare the raw data by cleaning, resampling to a weekly grid, and imputing missing values:
```bash
python -m src.data_prep
```

### 2. Model Training
Train the models across all states. This script holds out the last 8 weeks for validation, selects the best model, and serializes the artifacts:
```bash
python -m src.train
```
*(Tip: To test the pipeline quickly, run `python -m src.train Alabama,California` to train on a subset of states.)*

### 3. Start the Backend API Service
Serve the predictions using FastAPI from the root directory:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
*(Alternatively: `python -m api.main`)*

### 4. Start the Frontend Dashboard
In a new terminal window, navigate to the `frontend/` directory and start the Vite development server:
```bash
cd frontend
npm run dev
```

Once running, open the local URL provided by Vite (typically [http://localhost:5173/](http://localhost:5173/)) in your browser to interact with the forecasting dashboard.

## 🌐 API Endpoints

The API is fully documented. When the server is running, access the interactive OpenAPI documentation at [http://localhost:8000/docs](http://localhost:8000/docs).

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/`      | Interactive frontend console |
| `GET`  | `/health` | Liveness check & loaded state count |
| `GET`  | `/states` | List of states with their chosen model and validation MAPE |
| `GET`  | `/forecast/{state}?horizon=8` | Forecast the next N weeks for a specific state |
| `POST` | `/forecast` | Batch forecast for multiple states |

**Example API Request:**
```bash
curl -s 'http://localhost:8000/forecast/California?horizon=8'
```

**Example JSON Response:**
```json
{
  "state": "California",
  "model": "prophet",
  "horizon_weeks": 8,
  "last_train_week": "2023-12-04",
  "forecast": [
    {"week_start": "2023-12-11", "yhat": 487123450.71},
    {"week_start": "2023-12-18", "yhat": 472098101.50}
  ]
}
```

## 🔬 Methodology

1. **Data Preparation**: Resamples irregularly spaced observations into a strict weekly grid (`W-MON`). Zero-bins are treated as missing and imputed via time-aware linear interpolation.
2. **Feature Engineering**: Calculates lags (1, 2, 4, 8, 12, 52 weeks), rolling means and standard deviations (4 and 12-week windows) on shifted data to prevent data leakage, alongside calendar and US holiday boolean flags.
3. **Training & Validation**: Trains SARIMA, Prophet, XGBoost, and an LSTM network per state. A holdout validation set of the last 8 weeks evaluates the performance (MAPE). The winning model is then refitted on the full dataset so production forecasts utilize the most recent trends.
4. **Serving**: The serialized `joblib` models are loaded into memory by FastAPI, which dynamically handles incoming prediction requests, recursively applies feature engineering for step-ahead forecasting (XGBoost), and serves results.

AUTHOR 
BHANU MAHESH B
