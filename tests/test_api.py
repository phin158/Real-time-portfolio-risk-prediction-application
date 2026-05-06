"""
tests/test_api.py

Unit tests for FastAPI endpoints.
"""
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.services.risk_service import RiskService
from data_pipeline.schemas import ValidatedTick
from datetime import datetime, timezone

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_get_metrics_starting_up():
    response = client.get("/api/metrics")
    assert response.status_code == 200
    assert response.json() == {"status": "starting_up"}

def test_portfolio_no_predictions_yet():
    payload = {"weights": {"AAPL": 0.5, "MSFT": 0.5}}
    response = client.post("/api/portfolio", json=payload)
    # Because risk_service is None in TestClient (lifespan is bypassed or not fully mocked here)
    assert response.status_code == 503
    assert response.json()["detail"] == "Service starting up"

# In a real environment, we would use pytest-asyncio and properly mock the RiskService
# to test the logic in /api/portfolio and /api/metrics when risk_service is active.
