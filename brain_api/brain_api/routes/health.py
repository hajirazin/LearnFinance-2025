"""Health check endpoints."""

from pathlib import Path

from fastapi import APIRouter, HTTPException

from brain_api.storage.base import DEFAULT_DATA_PATH

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

    Verifies storage connectivity:
    - Data directory exists
    - Data directory is writable
    - Models directory exists

    Returns 503 if any check fails (Kubernetes will not route traffic).
    """
    checks: dict[str, bool] = {}
    errors: list[str] = []

    # Check 1: Data directory exists
    data_path = Path(DEFAULT_DATA_PATH)
    checks["data_dir_exists"] = data_path.exists()
    if not checks["data_dir_exists"]:
        errors.append(f"Data directory does not exist: {data_path.absolute()}")

    # Check 2: Data directory is writable
    if checks["data_dir_exists"]:
        try:
            # Try to create a temp file in the data directory
            test_file = data_path / ".health_check_test"
            test_file.touch()
            test_file.unlink()
            checks["data_dir_writable"] = True
        except (OSError, PermissionError) as e:
            checks["data_dir_writable"] = False
            errors.append(f"Data directory not writable: {e}")
    else:
        checks["data_dir_writable"] = False

    # Check 3: Models directory exists
    models_path = data_path / "models"
    checks["models_dir_exists"] = models_path.exists()
    if not checks["models_dir_exists"]:
        # Not an error - models dir may not exist yet on first run
        # but we note it for visibility
        pass

    # Overall status
    all_critical_passed = checks.get("data_dir_exists", False) and checks.get(
        "data_dir_writable", False
    )

    if not all_critical_passed:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "checks": checks,
                "errors": errors,
            },
        )

    return {
        "status": "ready",
        "checks": checks,
    }
