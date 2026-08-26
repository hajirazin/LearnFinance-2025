"""Signal endpoints for SAC price inputs and ppo_discovery state."""

from fastapi import APIRouter

from brain_api.routes.signals.endpoints import router as endpoints_router
from brain_api.routes.signals.ppo_discovery import router as ppo_discovery_router

router = APIRouter()
router.include_router(endpoints_router)
router.include_router(ppo_discovery_router)

__all__ = ["router"]
