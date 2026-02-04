"""Email endpoints for sending notifications and reports."""

from fastapi import APIRouter

from .models import (
    AlgorithmOrderResult,
    OrderResultsData,
    TrainingSummaryEmailRequest,
    TrainingSummaryEmailResponse,
    WeeklyReportEmailRequest,
    WeeklyReportEmailResponse,
)
from .training_summary import router as training_summary_router
from .weekly_report import router as weekly_report_router

# Create combined router
router = APIRouter()

# Include sub-routers
router.include_router(training_summary_router)
router.include_router(weekly_report_router)

__all__ = [
    "AlgorithmOrderResult",
    "OrderResultsData",
    "TrainingSummaryEmailRequest",
    "TrainingSummaryEmailResponse",
    "WeeklyReportEmailRequest",
    "WeeklyReportEmailResponse",
    "router",
]
