"""Tests for FastAPI health endpoint."""

from fastapi.testclient import TestClient

from app.api.main import create_app


def test_health_endpoint() -> None:
    """Health endpoint should return ok status."""
    app = create_app()
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["app"] == "tradequant-engine"
