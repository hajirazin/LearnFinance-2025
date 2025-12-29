"""Health check endpoints."""

from fastapi import APIRouter

router = APIRouter()


@router.get("")
def health_check() -> dict:
    """Generic health check."""
    return {"status": "healthy"}


@router.get("/live")
def liveness() -> dict:
    """Liveness probe - is the process running?"""
    return {"status": "alive"}


@router.get("/ready")
def readiness() -> dict:
    """Readiness probe - is the service ready to accept traffic?

    For now always returns ready. Later can check DB/storage connectivity.
    """
    return {"status": "ready"}

