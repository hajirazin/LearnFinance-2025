"""Training routes for ppo_discovery."""

from fastapi import APIRouter

from .full import router as full_router
from .preflight import router as preflight_router
from .promote import router as promote_router

router = APIRouter()
router.include_router(preflight_router)
router.include_router(full_router)
router.include_router(promote_router)

__all__ = ["router"]
