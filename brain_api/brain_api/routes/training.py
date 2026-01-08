"""Training endpoints for ML models."""

import logging
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from brain_api.core.config import (
    get_hf_model_repo,
    get_storage_backend,
    resolve_training_window,
)
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
from brain_api.core.patchtst import (
    DEFAULT_CONFIG as PATCHTST_DEFAULT_CONFIG,
    DatasetResult as PatchTSTDatasetResult,
    PatchTSTConfig,
    TrainingResult as PatchTSTTrainingResult,
    align_multivariate_data,
    build_dataset as patchtst_build_dataset,
    compute_version as patchtst_compute_version,
    evaluate_for_promotion as patchtst_evaluate_for_promotion,
    load_historical_fundamentals,
    load_historical_news_sentiment,
    load_prices_yfinance as patchtst_load_prices,
    train_model_pytorch as patchtst_train_model,
)
from brain_api.storage.local import (
    LocalModelStorage,
    PatchTSTModelStorage,
    create_metadata,
    create_patchtst_metadata,
)
from brain_api.universe import get_halal_universe

logger = logging.getLogger(__name__)

router = APIRouter()


class LSTMTrainResponse(BaseModel):
    """Response model for LSTM training endpoint."""

    version: str
    data_window_start: str  # YYYY-MM-DD
    data_window_end: str  # YYYY-MM-DD
    metrics: dict[str, Any]
    promoted: bool
    prior_version: str | None = None
    hf_repo: str | None = None  # HuggingFace repo if uploaded
    hf_url: str | None = None  # URL to model on HuggingFace


class PatchTSTTrainResponse(BaseModel):
    """Response model for PatchTST training endpoint."""

    version: str
    data_window_start: str  # YYYY-MM-DD
    data_window_end: str  # YYYY-MM-DD
    metrics: dict[str, Any]
    promoted: bool
    prior_version: str | None = None
    hf_repo: str | None = None  # HuggingFace repo if uploaded
    hf_url: str | None = None  # URL to model on HuggingFace
    # PatchTST-specific fields
    num_input_channels: int  # Number of feature channels used
    signals_used: list[str]  # List of signal types included


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

    Uses API config for data window (default: last 15 years).
    Fetches price data from yfinance for the halal universe.
    Writes versioned artifacts and promotes if evaluation passes.

    Returns:
        Training result including version, metrics, and promotion status.
    """
    import time

    # Resolve window from API config
    start_date, end_date = resolve_training_window()
    logger.info(f"[LSTM] Starting training for {len(symbols)} symbols")
    logger.info(f"[LSTM] Data window: {start_date} to {end_date}")
    logger.info(f"[LSTM] Symbols: {symbols}")

    # Compute deterministic version
    version = compute_version(start_date, end_date, symbols, config)
    logger.info(f"[LSTM] Computed version: {version}")

    # Check if this version already exists (idempotent)
    if storage.version_exists(version):
        logger.info(f"[LSTM] Version {version} already exists (idempotent), returning cached result")
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
    logger.info(f"[LSTM] Loading price data for {len(symbols)} symbols...")
    t0 = time.time()
    prices = price_loader(symbols, start_date, end_date)
    t_prices = time.time() - t0
    logger.info(f"[LSTM] Loaded prices for {len(prices)}/{len(symbols)} symbols in {t_prices:.1f}s")

    if len(prices) == 0:
        logger.error("[LSTM] No price data loaded - cannot train model")
        raise ValueError("No price data available for training")

    # Build dataset (returns DatasetResult with X, y (weekly returns), feature_scaler)
    logger.info("[LSTM] Building dataset...")
    t0 = time.time()
    dataset = dataset_builder(prices, config)
    t_dataset = time.time() - t0
    logger.info(f"[LSTM] Dataset built in {t_dataset:.1f}s: {len(dataset.X)} samples")

    if len(dataset.X) == 0:
        logger.error("[LSTM] Dataset is empty - cannot train model")
        raise ValueError("No training samples could be built from price data")

    # Train model
    logger.info("[LSTM] Starting model training...")
    t0 = time.time()
    result = trainer(
        dataset.X,
        dataset.y,
        dataset.feature_scaler,
        config,
    )
    t_train = time.time() - t0
    logger.info(f"[LSTM] Training complete in {t_train:.1f}s")
    logger.info(f"[LSTM] Metrics: train_loss={result.train_loss:.6f}, val_loss={result.val_loss:.6f}, baseline={result.baseline_loss:.6f}")

    # Get prior version info for promotion decision
    prior_version = storage.read_current_version()
    prior_val_loss = None
    if prior_version:
        logger.info(f"[LSTM] Prior version: {prior_version}")
        prior_metadata = storage.read_metadata(prior_version)
        if prior_metadata:
            prior_val_loss = prior_metadata["metrics"].get("val_loss")
            logger.info(f"[LSTM] Prior val_loss: {prior_val_loss}")
    else:
        logger.info("[LSTM] No prior version exists (first model)")

    # Decide on promotion
    promoted = evaluate_for_promotion(
        val_loss=result.val_loss,
        baseline_loss=result.baseline_loss,
        prior_val_loss=prior_val_loss,
    )
    logger.info(f"[LSTM] Promotion decision: {'PROMOTED' if promoted else 'NOT promoted'}")

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

    # Write artifacts locally
    logger.info(f"[LSTM] Writing artifacts for version {version}...")
    storage.write_artifacts(
        version=version,
        model=result.model,
        feature_scaler=result.feature_scaler,
        config=config,
        metadata=metadata,
    )
    logger.info("[LSTM] Artifacts written successfully")

    # Promote if passed evaluation, or if this is the first model (so inference has something)
    if promoted or prior_version is None:
        storage.promote_version(version)
        logger.info(f"[LSTM] Version {version} promoted to current")

    # Optionally push to HuggingFace Hub
    hf_repo = None
    hf_url = None
    storage_backend = get_storage_backend()
    hf_model_repo = get_hf_model_repo()

    if storage_backend == "hf" and hf_model_repo:
        try:
            from brain_api.storage.huggingface import HuggingFaceModelStorage

            hf_storage = HuggingFaceModelStorage(repo_id=hf_model_repo)
            hf_info = hf_storage.upload_model(
                version=version,
                model=result.model,
                feature_scaler=result.feature_scaler,
                config=config,
                metadata=metadata,
                make_current=(promoted or prior_version is None),
            )
            hf_repo = hf_info.repo_id
            hf_url = f"https://huggingface.co/{hf_info.repo_id}/tree/{version}"
            logger.info(f"Model uploaded to HuggingFace: {hf_url}")
        except Exception as e:
            logger.error(f"Failed to upload model to HuggingFace: {e}")
            # Don't fail the training request if HF upload fails

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
        hf_repo=hf_repo,
        hf_url=hf_url,
    )


# ============================================================================
# PatchTST Training Endpoint
# ============================================================================

# Type aliases for PatchTST dependency injection
PatchTSTPriceLoader = Callable[[list[str], Any, Any], dict]
PatchTSTNewsLoader = Callable[[list[str], Any, Any], dict]
PatchTSTFundamentalsLoader = Callable[[list[str], Any, Any], dict]
PatchTSTDataAligner = Callable[[dict, dict, dict, PatchTSTConfig], dict]
PatchTSTDatasetBuilder = Callable[[dict, dict, PatchTSTConfig], PatchTSTDatasetResult]
PatchTSTTrainer = Callable[[Any, Any, Any, PatchTSTConfig], PatchTSTTrainingResult]


def get_patchtst_storage() -> PatchTSTModelStorage:
    """Get the PatchTST model storage instance."""
    return PatchTSTModelStorage()


def get_patchtst_config() -> PatchTSTConfig:
    """Get PatchTST training configuration."""
    return PATCHTST_DEFAULT_CONFIG


def get_patchtst_price_loader() -> PatchTSTPriceLoader:
    """Get the price loading function for PatchTST."""
    return patchtst_load_prices


def get_patchtst_news_loader() -> PatchTSTNewsLoader:
    """Get the news sentiment loading function."""
    return load_historical_news_sentiment


def get_patchtst_fundamentals_loader() -> PatchTSTFundamentalsLoader:
    """Get the fundamentals loading function."""
    return load_historical_fundamentals


def get_patchtst_data_aligner() -> PatchTSTDataAligner:
    """Get the data alignment function."""
    return align_multivariate_data


def get_patchtst_dataset_builder() -> PatchTSTDatasetBuilder:
    """Get the dataset building function."""
    return patchtst_build_dataset


def get_patchtst_trainer() -> PatchTSTTrainer:
    """Get the training function."""
    return patchtst_train_model


@router.post("/patchtst", response_model=PatchTSTTrainResponse)
def train_patchtst(
    storage: PatchTSTModelStorage = Depends(get_patchtst_storage),
    symbols: list[str] = Depends(get_symbols),
    config: PatchTSTConfig = Depends(get_patchtst_config),
    price_loader: PatchTSTPriceLoader = Depends(get_patchtst_price_loader),
    news_loader: PatchTSTNewsLoader = Depends(get_patchtst_news_loader),
    fundamentals_loader: PatchTSTFundamentalsLoader = Depends(get_patchtst_fundamentals_loader),
    data_aligner: PatchTSTDataAligner = Depends(get_patchtst_data_aligner),
    dataset_builder: PatchTSTDatasetBuilder = Depends(get_patchtst_dataset_builder),
    trainer: PatchTSTTrainer = Depends(get_patchtst_trainer),
) -> PatchTSTTrainResponse:
    """Train the multi-signal PatchTST model for weekly return prediction.

    PatchTST uses OHLCV + external signals (news sentiment, fundamentals) to predict
    weekly returns. This contrasts with the pure-price LSTM baseline.

    Input channels (11 total):
    - OHLCV log returns (5)
    - News sentiment (1)
    - Fundamentals: gross_margin, operating_margin, net_margin, current_ratio, debt_to_equity (5)

    Uses API config for data window (default: last 15 years).
    Writes versioned artifacts and promotes if evaluation passes.

    Returns:
        Training result including version, metrics, and promotion status.
    """
    import time

    # Resolve window from API config
    start_date, end_date = resolve_training_window()
    logger.info(f"[PatchTST] Starting training for {len(symbols)} symbols")
    logger.info(f"[PatchTST] Data window: {start_date} to {end_date}")
    logger.info(f"[PatchTST] Symbols: {symbols}")
    logger.info(f"[PatchTST] Config: {config.num_input_channels} channels, {config.epochs} epochs")

    # Compute deterministic version
    version = patchtst_compute_version(start_date, end_date, symbols, config)
    logger.info(f"[PatchTST] Computed version: {version}")

    # Check if this version already exists (idempotent)
    if storage.version_exists(version):
        logger.info(f"[PatchTST] Version {version} already exists (idempotent), returning cached result")
        existing_metadata = storage.read_metadata(version)
        if existing_metadata:
            return PatchTSTTrainResponse(
                version=version,
                data_window_start=existing_metadata["data_window"]["start"],
                data_window_end=existing_metadata["data_window"]["end"],
                metrics=existing_metadata["metrics"],
                promoted=existing_metadata["promoted"],
                prior_version=existing_metadata.get("prior_version"),
                num_input_channels=config.num_input_channels,
                signals_used=["ohlcv", "news_sentiment", "fundamentals"],
            )

    # Load price data
    logger.info(f"[PatchTST] Loading price data for {len(symbols)} symbols...")
    t0 = time.time()
    prices = price_loader(symbols, start_date, end_date)
    t_prices = time.time() - t0
    logger.info(f"[PatchTST] Loaded prices for {len(prices)}/{len(symbols)} symbols in {t_prices:.1f}s")

    if len(prices) == 0:
        logger.error("[PatchTST] No price data loaded - cannot train model")
        raise ValueError("No price data available for training")

    # Load news sentiment
    logger.info("[PatchTST] Loading historical news sentiment...")
    t0 = time.time()
    news_sentiment = news_loader(symbols, start_date, end_date)
    t_news = time.time() - t0
    logger.info(f"[PatchTST] Loaded news sentiment for {len(news_sentiment)}/{len(symbols)} symbols in {t_news:.1f}s")

    # Load fundamentals
    logger.info("[PatchTST] Loading historical fundamentals...")
    t0 = time.time()
    fundamentals = fundamentals_loader(symbols, start_date, end_date)
    t_fund = time.time() - t0
    logger.info(f"[PatchTST] Loaded fundamentals for {len(fundamentals)}/{len(symbols)} symbols in {t_fund:.1f}s")

    # Align all data into multi-channel features
    logger.info("[PatchTST] Aligning multivariate data...")
    t0 = time.time()
    aligned_features = data_aligner(prices, news_sentiment, fundamentals, config)
    t_align = time.time() - t0
    logger.info(f"[PatchTST] Aligned data for {len(aligned_features)}/{len(prices)} symbols in {t_align:.1f}s")

    # Free intermediate data no longer needed (prices still needed for dataset_builder)
    del news_sentiment, fundamentals

    if len(aligned_features) == 0:
        logger.error("[PatchTST] No aligned features - cannot train model")
        raise ValueError("No aligned features could be built from available data")

    # Build dataset
    logger.info("[PatchTST] Building dataset...")
    t0 = time.time()
    dataset = dataset_builder(aligned_features, prices, config)
    t_dataset = time.time() - t0
    logger.info(f"[PatchTST] Dataset built in {t_dataset:.1f}s: {len(dataset.X)} samples")

    # Free aligned features and prices - no longer needed after dataset is built
    del aligned_features, prices

    if len(dataset.X) == 0:
        logger.error("[PatchTST] Dataset is empty - cannot train model")
        raise ValueError("No training samples could be built from aligned features")

    # Train model
    logger.info("[PatchTST] Starting model training...")
    t0 = time.time()
    result = trainer(
        dataset.X,
        dataset.y,
        dataset.feature_scaler,
        config,
    )
    t_train = time.time() - t0
    logger.info(f"[PatchTST] Training complete in {t_train:.1f}s")
    logger.info(f"[PatchTST] Metrics: train_loss={result.train_loss:.6f}, val_loss={result.val_loss:.6f}, baseline={result.baseline_loss:.6f}")

    # Get prior version info for promotion decision
    prior_version = storage.read_current_version()
    prior_val_loss = None
    if prior_version:
        logger.info(f"[PatchTST] Prior version: {prior_version}")
        prior_metadata = storage.read_metadata(prior_version)
        if prior_metadata:
            prior_val_loss = prior_metadata["metrics"].get("val_loss")
            logger.info(f"[PatchTST] Prior val_loss: {prior_val_loss}")
    else:
        logger.info("[PatchTST] No prior version exists (first model)")

    # Decide on promotion
    promoted = patchtst_evaluate_for_promotion(
        val_loss=result.val_loss,
        baseline_loss=result.baseline_loss,
        prior_val_loss=prior_val_loss,
    )
    logger.info(f"[PatchTST] Promotion decision: {'PROMOTED' if promoted else 'NOT promoted'}")

    # Create metadata
    metadata = create_patchtst_metadata(
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

    # Write artifacts locally
    logger.info(f"[PatchTST] Writing artifacts for version {version}...")
    storage.write_artifacts(
        version=version,
        model=result.model,
        feature_scaler=result.feature_scaler,
        config=config,
        metadata=metadata,
    )
    logger.info("[PatchTST] Artifacts written successfully")

    # Promote if passed evaluation, or if this is the first model
    if promoted or prior_version is None:
        storage.promote_version(version)
        logger.info(f"[PatchTST] Version {version} promoted to current")

    # Optionally push to HuggingFace Hub
    hf_repo = None
    hf_url = None
    storage_backend = get_storage_backend()
    hf_model_repo = get_hf_model_repo()

    if storage_backend == "hf" and hf_model_repo:
        try:
            from brain_api.storage.huggingface import PatchTSTHuggingFaceModelStorage

            hf_storage = PatchTSTHuggingFaceModelStorage(repo_id=hf_model_repo)
            hf_info = hf_storage.upload_model(
                version=version,
                model=result.model,
                feature_scaler=result.feature_scaler,
                config=config,
                metadata=metadata,
                make_current=(promoted or prior_version is None),
            )
            hf_repo = hf_info.repo_id
            hf_url = f"https://huggingface.co/{hf_info.repo_id}/tree/{version}"
            logger.info(f"[PatchTST] Model uploaded to HuggingFace: {hf_url}")
        except Exception as e:
            logger.error(f"[PatchTST] Failed to upload model to HuggingFace: {e}")
            # Don't fail the training request if HF upload fails

    return PatchTSTTrainResponse(
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
        hf_repo=hf_repo,
        hf_url=hf_url,
        num_input_channels=config.num_input_channels,
        signals_used=["ohlcv", "news_sentiment", "fundamentals"],
    )
