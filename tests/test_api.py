"""End-to-end API tests using FastAPI TestClient."""
from fastapi.testclient import TestClient
from api.main import app


def test_health():
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["states_loaded"] >= 1


def test_states_list():
    with TestClient(app) as c:
        r = c.get("/states")
        assert r.status_code == 200
        rows = r.json()
        assert any(row["state"] == "California" for row in rows)


def test_forecast_single():
    with TestClient(app) as c:
        r = c.get("/forecast/California?horizon=8")
        assert r.status_code == 200
        body = r.json()
        assert body["state"] == "California"
        assert len(body["forecast"]) == 8
        for pt in body["forecast"]:
            assert pt["yhat"] >= 0


def test_forecast_unknown_state():
    with TestClient(app) as c:
        assert c.get("/forecast/Atlantis").status_code == 404


def test_forecast_batch():
    with TestClient(app) as c:
        r = c.post("/forecast", json={"states": ["Alabama", "Texas"], "horizon": 4})
        assert r.status_code == 200
        body = r.json()
        assert body["horizon_weeks"] == 4
        assert len(body["results"]) == 2


def test_frontend_served():
    with TestClient(app) as c:
        r = c.get("/")
        assert r.status_code == 200
        assert "Sales Forecasting Dashboard" in r.text
