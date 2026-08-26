"""News HTTP surface."""

from fastapi import APIRouter

from brain_api.routes.news.endpoints import router as windows_router

router = APIRouter()
router.include_router(windows_router)
