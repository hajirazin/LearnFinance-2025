"""Root endpoint (hello world)."""

from fastapi import APIRouter

router = APIRouter(tags=["root"])


@router.get("/")
def read_root() -> dict:
    """Hello world endpoint."""
    return {"message": "Hello from Brain API", "service": "brain-api", "version": "0.1.0"}



