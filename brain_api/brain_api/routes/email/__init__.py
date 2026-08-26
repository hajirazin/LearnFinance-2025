"""Email endpoints for sending notifications and reports."""

from fastapi import APIRouter

from .models import (
    AlgorithmOrderResult,
    AlphaHRPEmailRequest,
    ForecastersTrainingSummaryEmailRequest,
    IndiaAlphaHRPEmailRequest,
    OrderResultsData,
    SACTrainingSummaryEmailRequest,
    SACWeeklyReportEmailRequest,
    TrainingSummaryEmailResponse,
    USAlphaHRPEmailRequest,
    WeeklyReportEmailResponse,
)
from .ppo_discovery import router as ppo_discovery_router
from .training_summary import router as training_summary_router
from .weekly_report import router as weekly_report_router

# Create combined router
router = APIRouter()

# Include sub-routers
router.include_router(training_summary_router)
router.include_router(weekly_report_router)
router.include_router(ppo_discovery_router)

__all__ = [
    "AlgorithmOrderResult",
    "AlphaHRPEmailRequest",
    "ForecastersTrainingSummaryEmailRequest",
    "IndiaAlphaHRPEmailRequest",
    "OrderResultsData",
    "SACTrainingSummaryEmailRequest",
    "SACWeeklyReportEmailRequest",
    "TrainingSummaryEmailResponse",
    "USAlphaHRPEmailRequest",
    "WeeklyReportEmailResponse",
    "router",
]
