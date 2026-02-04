"""Email endpoints for sending notifications and reports."""

from fastapi import APIRouter

from .models import (
    TrainingSummaryEmailRequest,
    TrainingSummaryEmailResponse,
)
from .training_summary import router as training_summary_router

# Create combined router
router = APIRouter()

# Include sub-routers
router.include_router(training_summary_router)

__all__ = [
    "TrainingSummaryEmailRequest",
    "TrainingSummaryEmailResponse",
    "router",
]
