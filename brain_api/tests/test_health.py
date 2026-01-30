"""API-level tests for health endpoints."""

from fastapi.testclient import TestClient

from brain_api.main import app

client = TestClient(app)


def test_root_returns_hello():
    """GET / returns hello world payload."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Hello from Brain API"
    assert data["service"] == "brain-api"
    assert "version" in data


def test_health_check():
    """GET /health returns healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_liveness():
    """GET /health/live returns alive status."""
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_readiness():
    """GET /health/ready returns ready status with checks."""
    response = client.get("/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    # Readiness check now includes detailed checks
    assert "checks" in data
    assert "data_dir_exists" in data["checks"]
    assert "data_dir_writable" in data["checks"]
    assert "models_dir_exists" in data["checks"]
