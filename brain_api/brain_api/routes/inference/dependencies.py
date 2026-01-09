"""Shared dependency injection for inference endpoints."""

from datetime import date
from pathlib import Path

from brain_api.core.lstm import compute_week_boundaries, load_prices_yfinance
from brain_api.storage.local import (
    LocalModelStorage,
    PatchTSTModelStorage,
    PPOLSTMLocalStorage,
    PPOPatchTSTLocalStorage,
    SACLSTMLocalStorage,
    SACPatchTSTLocalStorage,
)

from .models import (
    LSTMInferenceRequest,
    PatchTSTInferenceRequest,
    PPOLSTMInferenceRequest,
    PPOPatchTSTInferenceRequest,
    SACLSTMInferenceRequest,
    SACPatchTSTInferenceRequest,
)

# Type aliases for dependency injection
PriceLoader = type(load_prices_yfinance)
WeekBoundaryComputer = type(compute_week_boundaries)


# ============================================================================
# LSTM dependencies
# ============================================================================


def get_storage() -> LocalModelStorage:
    """Get the model storage instance."""
    return LocalModelStorage()


def get_as_of_date(request: LSTMInferenceRequest) -> date:
    """Get the as-of date from request or default to today."""
    if request.as_of_date:
        return date.fromisoformat(request.as_of_date)
    return date.today()


def get_price_loader() -> PriceLoader:
    """Get the price loading function."""
    return load_prices_yfinance


def get_week_boundary_computer() -> WeekBoundaryComputer:
    """Get the week boundary computation function."""
    return compute_week_boundaries


# ============================================================================
# PatchTST dependencies
# ============================================================================


def get_patchtst_storage() -> PatchTSTModelStorage:
    """Get the PatchTST model storage instance."""
    return PatchTSTModelStorage()


def get_patchtst_as_of_date(request: PatchTSTInferenceRequest) -> date:
    """Get the as-of date from request or default to today."""
    if request.as_of_date:
        return date.fromisoformat(request.as_of_date)
    return date.today()


def get_sentiment_parquet_path() -> Path:
    """Get the path to the historical sentiment parquet file."""
    project_root = Path(__file__).parent.parent.parent.parent
    return project_root / "data" / "output" / "daily_sentiment.parquet"


# ============================================================================
# PPO dependencies
# ============================================================================


def get_ppo_lstm_storage() -> PPOLSTMLocalStorage:
    """Get the PPO + LSTM storage instance."""
    return PPOLSTMLocalStorage()


def get_ppo_lstm_as_of_date(request: PPOLSTMInferenceRequest) -> date:
    """Get the as-of date from request or default to today."""
    if request.as_of_date:
        return date.fromisoformat(request.as_of_date)
    return date.today()


def get_ppo_patchtst_storage() -> PPOPatchTSTLocalStorage:
    """Get the PPO + PatchTST storage instance."""
    return PPOPatchTSTLocalStorage()


def get_ppo_patchtst_as_of_date(request: PPOPatchTSTInferenceRequest) -> date:
    """Get the as-of date from request or default to today."""
    if request.as_of_date:
        return date.fromisoformat(request.as_of_date)
    return date.today()


# ============================================================================
# SAC dependencies
# ============================================================================


def get_sac_lstm_storage() -> SACLSTMLocalStorage:
    """Get the SAC + LSTM storage instance."""
    return SACLSTMLocalStorage()


def get_sac_lstm_as_of_date(request: SACLSTMInferenceRequest) -> date:
    """Get the as-of date from request or default to today."""
    if request.as_of_date:
        return date.fromisoformat(request.as_of_date)
    return date.today()


def get_sac_patchtst_storage() -> SACPatchTSTLocalStorage:
    """Get the SAC + PatchTST storage instance."""
    return SACPatchTSTLocalStorage()


def get_sac_patchtst_as_of_date(request: SACPatchTSTInferenceRequest) -> date:
    """Get the as-of date from request or default to today."""
    if request.as_of_date:
        return date.fromisoformat(request.as_of_date)
    return date.today()
