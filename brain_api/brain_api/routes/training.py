"""Training endpoints for ML models."""

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from brain_api.core.config import resolve_training_window
from brain_api.core.lstm import (
    DEFAULT_CONFIG,
    DatasetResult,
    LSTMConfig,
    TrainingResult,
    build_dataset,
    compute_version,
    evaluate_for_promotion,
    load_prices_yfinance,
    train_model_pytorch,
)
from brain_api.storage.local import LocalModelStorage, create_metadata
from brain_api.universe import get_halal_universe

router = APIRouter()


class LSTMTrainResponse(BaseModel):
    """Response model for LSTM training endpoint."""

    version: str
    data_window_start: str  # YYYY-MM-DD
    data_window_end: str  # YYYY-MM-DD
    metrics: dict[str, Any]
    promoted: bool
    prior_version: str | None = None


# ============================================================================
# Dependency injection for testability
# ============================================================================


def get_storage() -> LocalModelStorage:
    """Get the model storage instance."""
    return LocalModelStorage()


def get_symbols() -> list[str]:
    """Get symbols for training from halal universe."""
    universe = get_halal_universe()
    return [stock["symbol"] for stock in universe["stocks"]]


def get_config() -> LSTMConfig:
    """Get LSTM training configuration."""
    return DEFAULT_CONFIG


# Type aliases for dependency injection
PriceLoader = Callable[[list[str], Any, Any], dict]
DatasetBuilder = Callable[[dict, LSTMConfig], DatasetResult]
Trainer = Callable[[Any, Any, Any, LSTMConfig], TrainingResult]  # X, y, scaler, config


def get_price_loader() -> PriceLoader:
    """Get the price loading function."""
    return load_prices_yfinance


def get_dataset_builder() -> DatasetBuilder:
    """Get the dataset building function."""
    return build_dataset


def get_trainer() -> Trainer:
    """Get the training function."""
    return train_model_pytorch


# ============================================================================
# Endpoint
# ============================================================================


@router.post("/lstm", response_model=LSTMTrainResponse)
def train_lstm(
    storage: LocalModelStorage = Depends(get_storage),
    symbols: list[str] = Depends(get_symbols),
    config: LSTMConfig = Depends(get_config),
    price_loader: PriceLoader = Depends(get_price_loader),
    dataset_builder: DatasetBuilder = Depends(get_dataset_builder),
    trainer: Trainer = Depends(get_trainer),
) -> LSTMTrainResponse:
    """Train the shared LSTM model for weekly return prediction.

    The model predicts weekly returns (Mon open → Fri close) to align with
    the RL agent's weekly decision horizon. This naturally handles holidays
    as a "week" is simply the first-to-last trading day of each ISO week.

    Uses API config for data window (default: last 10 years).
    Fetches price data from yfinance for the halal universe.
    Writes versioned artifacts and promotes if evaluation passes.

    Returns:
        Training result including version, metrics, and promotion status.
    """
    # Resolve window from API config
    start_date, end_date = resolve_training_window()

    # Compute deterministic version
    version = compute_version(start_date, end_date, symbols, config)

    # Check if this version already exists (idempotent)
    if storage.version_exists(version):
        # Return existing metadata
        existing_metadata = storage.read_metadata(version)
        if existing_metadata:
            return LSTMTrainResponse(
                version=version,
                data_window_start=existing_metadata["data_window"]["start"],
                data_window_end=existing_metadata["data_window"]["end"],
                metrics=existing_metadata["metrics"],
                promoted=existing_metadata["promoted"],
                prior_version=existing_metadata.get("prior_version"),
            )

    # Load price data
    prices = price_loader(symbols, start_date, end_date)

    # Build dataset (returns DatasetResult with X, y (weekly returns), feature_scaler)
    dataset = dataset_builder(prices, config)

    # Train model
    result = trainer(
        dataset.X,
        dataset.y,
        dataset.feature_scaler,
        config,
    )

    # Get prior version info for promotion decision
    prior_version = storage.read_current_version()
    prior_val_loss = None
    if prior_version:
        prior_metadata = storage.read_metadata(prior_version)
        if prior_metadata:
            prior_val_loss = prior_metadata["metrics"].get("val_loss")

    # Decide on promotion
    promoted = evaluate_for_promotion(
        val_loss=result.val_loss,
        baseline_loss=result.baseline_loss,
        prior_val_loss=prior_val_loss,
    )

    # Create metadata
    metadata = create_metadata(
        version=version,
        data_window_start=start_date.isoformat(),
        data_window_end=end_date.isoformat(),
        symbols=symbols,
        config=config,
        train_loss=result.train_loss,
        val_loss=result.val_loss,
        baseline_loss=result.baseline_loss,
        promoted=promoted,
        prior_version=prior_version,
    )

    # Write artifacts
    storage.write_artifacts(
        version=version,
        model=result.model,
        feature_scaler=result.feature_scaler,
        config=config,
        metadata=metadata,
    )

    # Promote if passed evaluation, or if this is the first model (so inference has something)
    if promoted or prior_version is None:
        storage.promote_version(version)

    return LSTMTrainResponse(
        version=version,
        data_window_start=start_date.isoformat(),
        data_window_end=end_date.isoformat(),
        metrics={
            "train_loss": result.train_loss,
            "val_loss": result.val_loss,
            "baseline_loss": result.baseline_loss,
        },
        promoted=promoted,
        prior_version=prior_version,
    )
