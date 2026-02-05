"""Pydantic models for brain_api requests and responses."""

from flows.models.email import TrainingSummaryEmailResponse
from flows.models.etl import RefreshTrainingDataRequest, RefreshTrainingDataResponse
from flows.models.forecast_email import (
    AlpacaPortfolioResponse,
    FundamentalsResponse,
    GenerateOrdersResponse,
    HRPAllocationResponse,
    LSTMInferenceResponse,
    NewsSignalResponse,
    OrderHistoryItem,
    OrderModel,
    OrderSubmitResult,
    PatchTSTInferenceResponse,
    PositionModel,
    PPOInferenceResponse,
    SACInferenceResponse,
    SkippedAllocation,
    SkippedOrdersResponse,
    SkippedSubmitResponse,
    StoreExperienceResponse,
    SubmitOrdersResponse,
    UpdateExecutionResponse,
    WeeklyReportEmailResponse,
    WeeklySummaryResponse,
)
from flows.models.llm import TrainingSummaryResponse
from flows.models.training import TrainingResponse
from flows.models.universe import HalalUniverseResponse

__all__ = [
    # Universe
    "HalalUniverseResponse",
    # ETL
    "RefreshTrainingDataRequest",
    "RefreshTrainingDataResponse",
    # Training
    "TrainingResponse",
    "TrainingSummaryEmailResponse",
    "TrainingSummaryResponse",
    # Forecast Email Flow
    "AlpacaPortfolioResponse",
    "FundamentalsResponse",
    "GenerateOrdersResponse",
    "HRPAllocationResponse",
    "LSTMInferenceResponse",
    "NewsSignalResponse",
    "OrderHistoryItem",
    "OrderModel",
    "OrderSubmitResult",
    "PatchTSTInferenceResponse",
    "PositionModel",
    "PPOInferenceResponse",
    "SACInferenceResponse",
    "SkippedAllocation",
    "SkippedOrdersResponse",
    "SkippedSubmitResponse",
    "StoreExperienceResponse",
    "SubmitOrdersResponse",
    "UpdateExecutionResponse",
    "WeeklyReportEmailResponse",
    "WeeklySummaryResponse",
]
