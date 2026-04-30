"""Shared dependency injection for inference endpoints."""

from datetime import date
from pathlib import Path

from brain_api.core.config import resolve_cutoff_date
from brain_api.core.lstm import compute_week_boundaries, load_prices_yfinance
from brain_api.storage.local import (
    LocalModelStorage,
    PatchTSTIndiaModelStorage,
    PatchTSTModelStorage,
)
from brain_api.storage.sac import SACLocalStorage

from .models import (
    LSTMInferenceRequest,
    PatchTSTInferenceRequest,
    SACInferenceRequest,
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
    """Get cutoff date (always Friday) from request or computed from today."""
    reference = date.fromisoformat(request.as_of_date) if request.as_of_date else None
    return resolve_cutoff_date(reference)


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


def get_patchtst_india_storage() -> PatchTSTIndiaModelStorage:
    """Get the India PatchTST model storage instance.

    India PatchTST artifacts live under ``data/models/patchtst_india/`` --
    distinct from US weights/scalers because they are trained on a
    different price distribution (Nifty Shariah 500). Reusing US weights
    on Indian symbols would be mathematically wrong.
    """
    return PatchTSTIndiaModelStorage()


def get_patchtst_as_of_date(request: PatchTSTInferenceRequest) -> date:
    """Get cutoff date (always Friday) from request or computed from today."""
    reference = date.fromisoformat(request.as_of_date) if request.as_of_date else None
    return resolve_cutoff_date(reference)


def get_sentiment_parquet_path() -> Path:
    """Get the path to the historical sentiment parquet file."""
    project_root = Path(__file__).parent.parent.parent.parent
    return project_root / "data" / "output" / "daily_sentiment.parquet"


# ============================================================================
# SAC dependencies (unified with dual forecasts)
# ============================================================================


def get_sac_storage() -> SACLocalStorage:
    """Get the SAC storage instance."""
    return SACLocalStorage()


def get_sac_as_of_date(request: SACInferenceRequest) -> date:
    """Get cutoff date (always Friday) from request or computed from today."""
    reference = date.fromisoformat(request.as_of_date) if request.as_of_date else None
    return resolve_cutoff_date(reference)
