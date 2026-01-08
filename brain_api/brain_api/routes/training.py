"""Training endpoints for ML models."""

import logging
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from brain_api.core.config import (
    get_hf_lstm_model_repo,
    get_hf_ppo_lstm_model_repo,
    get_hf_ppo_patchtst_model_repo,
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
    PPOLSTMLocalStorage,
    PPOPatchTSTLocalStorage,
    create_metadata,
    create_patchtst_metadata,
    create_ppo_lstm_metadata,
    create_ppo_patchtst_metadata,
)
from brain_api.universe import get_halal_universe

# Import PPO LSTM training components
from brain_api.core.ppo_lstm import (
    DEFAULT_PPO_LSTM_CONFIG,
    PPOLSTMConfig,
    PPOFinetuneConfig,
    build_training_data,
    compute_version as ppo_lstm_compute_version,
    train_ppo_lstm,
    finetune_ppo_lstm,
)

# Import PPO PatchTST training components
from brain_api.core.ppo_patchtst import (
    DEFAULT_PPO_PATCHTST_CONFIG,
    PPOPatchTSTConfig,
    compute_version as ppo_patchtst_compute_version,
    train_ppo_patchtst,
    finetune_ppo_patchtst,
)

# Import SAC LSTM training components
from brain_api.core.sac_lstm import (
    DEFAULT_SAC_LSTM_CONFIG,
    SACLSTMConfig,
    build_training_data as sac_build_training_data,
    compute_version as sac_lstm_compute_version,
    train_sac_lstm,
    finetune_sac_lstm,
)

# Import SAC PatchTST training components
from brain_api.core.sac_patchtst import (
    DEFAULT_SAC_PATCHTST_CONFIG,
    SACPatchTSTConfig,
    compute_version as sac_patchtst_compute_version,
    train_sac_patchtst,
    finetune_sac_patchtst,
)

# Import SAC storage
from brain_api.storage.local import (
    SACLSTMLocalStorage,
    SACPatchTSTLocalStorage,
    create_sac_lstm_metadata,
    create_sac_patchtst_metadata,
)

# Import SAC config utilities
from brain_api.core.config import (
    get_hf_sac_lstm_model_repo,
    get_hf_sac_patchtst_model_repo,
)
from brain_api.core.portfolio_rl.sac_config import SACFinetuneConfig

# Import shared data loading for RL training
from brain_api.core.portfolio_rl.data_loading import build_rl_training_signals
from brain_api.core.portfolio_rl.walkforward import build_forecast_features
from brain_api.storage.forecaster_snapshots import SnapshotLocalStorage


def _snapshots_available(forecaster_type: str) -> bool:
    """Check if forecaster snapshots are available for walk-forward inference.

    Args:
        forecaster_type: "lstm" or "patchtst"

    Returns:
        True if at least one snapshot exists
    """
    storage = SnapshotLocalStorage(forecaster_type)
    snapshots = storage.list_snapshots()
    return len(snapshots) > 0

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
    skip_snapshot: bool = Query(False, description="Skip saving snapshot (by default saves snapshot for current + all historical years)"),
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

    By default, also saves snapshots for all historical years (for walk-forward
    forecast generation in RL training). Use skip_snapshot=true to disable.

    Args:
        skip_snapshot: If True, skips saving snapshots. By default (False),
                      saves snapshot for current training window + all historical years.

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
    hf_model_repo = get_hf_lstm_model_repo()

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

    # Save snapshots (unless skip_snapshot=True)
    if not skip_snapshot:
        from brain_api.storage.forecaster_snapshots import (
            SnapshotLocalStorage,
            create_snapshot_metadata,
        )
        snapshot_storage = SnapshotLocalStorage("lstm")

        # Save snapshot for current training window
        if not snapshot_storage.snapshot_exists(end_date):
            snapshot_metadata = create_snapshot_metadata(
                forecaster_type="lstm",
                cutoff_date=end_date,
                data_window_start=start_date.isoformat(),
                data_window_end=end_date.isoformat(),
                symbols=list(prices.keys()),
                config=config,
                train_loss=result.train_loss,
                val_loss=result.val_loss,
            )
            snapshot_storage.write_snapshot(
                cutoff_date=end_date,
                model=result.model,
                feature_scaler=result.feature_scaler,
                config=config,
                metadata=snapshot_metadata,
            )
            logger.info(f"[LSTM] Saved snapshot for cutoff {end_date}")

            # Upload to HuggingFace if in HF mode
            if storage_backend == "hf":
                try:
                    snapshot_storage.upload_snapshot_to_hf(end_date)
                    logger.info(f"[LSTM] Uploaded snapshot {end_date} to HuggingFace")
                except Exception as e:
                    logger.error(f"[LSTM] Failed to upload snapshot to HF: {e}")

        # Also backfill all historical snapshots
        logger.info("[LSTM] Backfilling historical snapshots...")
        _backfill_lstm_snapshots(symbols, config, start_date, end_date, snapshot_storage, storage_backend)

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


def _backfill_lstm_snapshots(
    symbols: list[str],
    config: LSTMConfig,
    start_date: "date",
    end_date: "date",
    snapshot_storage: "SnapshotLocalStorage",
    storage_backend: str = "local",
) -> None:
    """Backfill LSTM snapshots for all historical years.

    For each year from (start_year + 4) to end_year-1, trains a snapshot
    on data up to Dec 31 of that year. Uses 4-year bootstrap period.

    Args:
        symbols: List of stock symbols
        config: LSTM configuration
        start_date: Training data start date
        end_date: Training data end date
        snapshot_storage: Storage instance
        storage_backend: "local" or "hf" - if "hf", uploads to HuggingFace
    """
    import time
    from datetime import date
    from brain_api.storage.forecaster_snapshots import create_snapshot_metadata

    start_year = start_date.year
    end_year = end_date.year
    bootstrap_years = 4

    for year in range(start_year + bootstrap_years, end_year):
        cutoff_date = date(year, 12, 31)

        if snapshot_storage.snapshot_exists(cutoff_date):
            logger.info(f"[LSTM Backfill] Snapshot for {cutoff_date} already exists, skipping")
            continue

        logger.info(f"[LSTM Backfill] Training snapshot for cutoff {cutoff_date}")

        t0 = time.time()
        prices = load_prices_yfinance(symbols, start_date, cutoff_date)
        if len(prices) == 0:
            logger.warning(f"[LSTM Backfill] No price data for cutoff {cutoff_date}, skipping")
            continue

        dataset = build_dataset(prices, config)
        if len(dataset.X) == 0:
            logger.warning(f"[LSTM Backfill] Empty dataset for cutoff {cutoff_date}, skipping")
            continue

        result = train_model_pytorch(dataset.X, dataset.y, dataset.feature_scaler, config)

        metadata = create_snapshot_metadata(
            forecaster_type="lstm",
            cutoff_date=cutoff_date,
            data_window_start=start_date.isoformat(),
            data_window_end=cutoff_date.isoformat(),
            symbols=list(prices.keys()),
            config=config,
            train_loss=result.train_loss,
            val_loss=result.val_loss,
        )

        snapshot_storage.write_snapshot(
            cutoff_date=cutoff_date,
            model=result.model,
            feature_scaler=result.feature_scaler,
            config=config,
            metadata=metadata,
        )
        logger.info(f"[LSTM Backfill] Saved snapshot for {cutoff_date} in {time.time() - t0:.1f}s")

        # Upload to HuggingFace if in HF mode
        if storage_backend == "hf":
            try:
                snapshot_storage.upload_snapshot_to_hf(cutoff_date)
                logger.info(f"[LSTM Backfill] Uploaded snapshot {cutoff_date} to HuggingFace")
            except Exception as e:
                logger.error(f"[LSTM Backfill] Failed to upload snapshot to HF: {e}")


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
    skip_snapshot: bool = Query(False, description="Skip saving snapshot (by default saves snapshot for current + all historical years)"),
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

    By default, also saves snapshots for all historical years (for walk-forward
    forecast generation in RL training). Use skip_snapshot=true to disable.

    Args:
        skip_snapshot: If True, skips saving snapshots. By default (False),
                      saves snapshot for current training window + all historical years.

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

    # Save symbols list before freeing prices (needed for snapshot)
    available_symbols = list(prices.keys())

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
    hf_model_repo = get_hf_lstm_model_repo()

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

    # Save snapshots (unless skip_snapshot=True)
    if not skip_snapshot:
        from brain_api.storage.forecaster_snapshots import (
            SnapshotLocalStorage,
            create_snapshot_metadata,
        )
        snapshot_storage = SnapshotLocalStorage("patchtst")

        # Save snapshot for current training window
        if not snapshot_storage.snapshot_exists(end_date):
            snapshot_metadata = create_snapshot_metadata(
                forecaster_type="patchtst",
                cutoff_date=end_date,
                data_window_start=start_date.isoformat(),
                data_window_end=end_date.isoformat(),
                symbols=available_symbols,
                config=config,
                train_loss=result.train_loss,
                val_loss=result.val_loss,
            )
            snapshot_storage.write_snapshot(
                cutoff_date=end_date,
                model=result.model,
                feature_scaler=result.feature_scaler,
                config=config,
                metadata=snapshot_metadata,
            )
            logger.info(f"[PatchTST] Saved snapshot for cutoff {end_date}")

            # Upload to HuggingFace if in HF mode
            if storage_backend == "hf":
                try:
                    snapshot_storage.upload_snapshot_to_hf(end_date)
                    logger.info(f"[PatchTST] Uploaded snapshot {end_date} to HuggingFace")
                except Exception as e:
                    logger.error(f"[PatchTST] Failed to upload snapshot to HF: {e}")

        # Also backfill all historical snapshots
        logger.info("[PatchTST] Backfilling historical snapshots...")
        _backfill_patchtst_snapshots(symbols, config, start_date, end_date, snapshot_storage, storage_backend)

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


def _backfill_patchtst_snapshots(
    symbols: list[str],
    config: "PatchTSTConfig",
    start_date: "date",
    end_date: "date",
    snapshot_storage: "SnapshotLocalStorage",
    storage_backend: str = "local",
) -> None:
    """Backfill PatchTST snapshots for all historical years.

    For each year from (start_year + 4) to end_year-1, trains a snapshot
    on data up to Dec 31 of that year. Uses 4-year bootstrap period.

    Args:
        symbols: List of stock symbols
        config: PatchTST configuration
        start_date: Training data start date
        end_date: Training data end date
        snapshot_storage: Storage instance
        storage_backend: "local" or "hf" - if "hf", uploads to HuggingFace
    """
    import time
    from datetime import date
    from brain_api.storage.forecaster_snapshots import create_snapshot_metadata

    start_year = start_date.year
    end_year = end_date.year
    bootstrap_years = 4

    for year in range(start_year + bootstrap_years, end_year):
        cutoff_date = date(year, 12, 31)

        if snapshot_storage.snapshot_exists(cutoff_date):
            logger.info(f"[PatchTST Backfill] Snapshot for {cutoff_date} already exists, skipping")
            continue

        logger.info(f"[PatchTST Backfill] Training snapshot for cutoff {cutoff_date}")

        t0 = time.time()
        prices = patchtst_load_prices(symbols, start_date, cutoff_date)
        if len(prices) == 0:
            logger.warning(f"[PatchTST Backfill] No price data for cutoff {cutoff_date}, skipping")
            continue

        news_sentiment = load_historical_news_sentiment(symbols, start_date, cutoff_date)
        fundamentals = load_historical_fundamentals(symbols, start_date, cutoff_date)
        aligned_features = align_multivariate_data(prices, news_sentiment, fundamentals, config)

        if len(aligned_features) == 0:
            logger.warning(f"[PatchTST Backfill] No aligned features for cutoff {cutoff_date}, skipping")
            continue

        dataset = patchtst_build_dataset(aligned_features, prices, config)
        if len(dataset.X) == 0:
            logger.warning(f"[PatchTST Backfill] Empty dataset for cutoff {cutoff_date}, skipping")
            continue

        result = patchtst_train_model(dataset.X, dataset.y, dataset.feature_scaler, config)

        metadata = create_snapshot_metadata(
            forecaster_type="patchtst",
            cutoff_date=cutoff_date,
            data_window_start=start_date.isoformat(),
            data_window_end=cutoff_date.isoformat(),
            symbols=list(prices.keys()),
            config=config,
            train_loss=result.train_loss,
            val_loss=result.val_loss,
        )

        snapshot_storage.write_snapshot(
            cutoff_date=cutoff_date,
            model=result.model,
            feature_scaler=result.feature_scaler,
            config=config,
            metadata=metadata,
        )
        logger.info(f"[PatchTST Backfill] Saved snapshot for {cutoff_date} in {time.time() - t0:.1f}s")

        # Upload to HuggingFace if in HF mode
        if storage_backend == "hf":
            try:
                snapshot_storage.upload_snapshot_to_hf(cutoff_date)
                logger.info(f"[PatchTST Backfill] Uploaded snapshot {cutoff_date} to HuggingFace")
            except Exception as e:
                logger.error(f"[PatchTST Backfill] Failed to upload snapshot to HF: {e}")


# ============================================================================
# PPO + LSTM Training Endpoint
# ============================================================================


class PPOLSTMTrainResponse(BaseModel):
    """Response model for PPO + LSTM training endpoint."""

    version: str
    data_window_start: str  # YYYY-MM-DD
    data_window_end: str  # YYYY-MM-DD
    metrics: dict[str, Any]
    promoted: bool
    prior_version: str | None = None
    symbols_used: list[str]
    hf_repo: str | None = None  # HuggingFace repo if uploaded
    hf_url: str | None = None  # URL to model on HuggingFace


def get_ppo_lstm_storage() -> PPOLSTMLocalStorage:
    """Get the PPO + LSTM storage instance."""
    return PPOLSTMLocalStorage()


def get_ppo_lstm_config() -> PPOLSTMConfig:
    """Get PPO + LSTM configuration."""
    return DEFAULT_PPO_LSTM_CONFIG


def get_top15_symbols() -> list[str]:
    """Get top 15 symbols by liquidity from halal universe."""
    universe = get_halal_universe()
    stocks = universe["stocks"][:15]
    return [stock["symbol"] for stock in stocks]


@router.post("/ppo_lstm/full", response_model=PPOLSTMTrainResponse)
def train_ppo_lstm_endpoint(
    storage: PPOLSTMLocalStorage = Depends(get_ppo_lstm_storage),
    symbols: list[str] = Depends(get_top15_symbols),
    config: PPOLSTMConfig = Depends(get_ppo_lstm_config),
) -> PPOLSTMTrainResponse:
    """Train PPO portfolio allocator using LSTM forecasts.

    This endpoint:
    1. Loads historical price data and signals
    2. Generates LSTM forecast features (from pre-trained LSTM)
    3. Trains PPO policy for portfolio allocation
    4. Evaluates on held-out data
    5. Promotes if first model or beats prior

    Returns:
        Training result including version, metrics, and promotion status.
    """
    import time

    import numpy as np

    # Resolve window from API config
    start_date, end_date = resolve_training_window()
    logger.info(f"[PPO_LSTM] Starting training for {len(symbols)} symbols")
    logger.info(f"[PPO_LSTM] Data window: {start_date} to {end_date}")
    logger.info(f"[PPO_LSTM] Symbols: {symbols}")

    # Compute deterministic version
    version = ppo_lstm_compute_version(start_date, end_date, symbols, config)
    logger.info(f"[PPO_LSTM] Computed version: {version}")

    # Check if this version already exists (idempotent)
    if storage.version_exists(version):
        logger.info(f"[PPO_LSTM] Version {version} already exists (idempotent)")
        existing_metadata = storage.read_metadata(version)
        if existing_metadata:
            return PPOLSTMTrainResponse(
                version=version,
                data_window_start=existing_metadata["data_window"]["start"],
                data_window_end=existing_metadata["data_window"]["end"],
                metrics=existing_metadata["metrics"],
                promoted=existing_metadata["promoted"],
                prior_version=existing_metadata.get("prior_version"),
                symbols_used=existing_metadata["symbols"],
            )

    # Load price data
    logger.info("[PPO_LSTM] Loading price data...")
    t0 = time.time()
    prices_dict = load_prices_yfinance(symbols, start_date, end_date)
    t_prices = time.time() - t0
    logger.info(f"[PPO_LSTM] Loaded prices for {len(prices_dict)}/{len(symbols)} symbols in {t_prices:.1f}s")

    if len(prices_dict) == 0:
        raise ValueError("No price data available for training")

    # Filter symbols to those with price data
    available_symbols = [s for s in symbols if s in prices_dict]
    if len(available_symbols) < 5:
        raise ValueError(f"Need at least 5 symbols with data, got {len(available_symbols)}")

    # Resample prices to weekly (Friday close)
    logger.info("[PPO_LSTM] Resampling prices to weekly...")
    weekly_prices = {}
    for symbol in available_symbols:
        df = prices_dict[symbol]
        if df is not None and len(df) > 0:
            weekly = df["close"].resample("W-FRI").last().dropna()
            weekly_prices[symbol] = weekly.values

    # Determine minimum length across all symbols
    min_weeks = min(len(weekly_prices[s]) for s in available_symbols if s in weekly_prices)
    logger.info(f"[PPO_LSTM] Using {min_weeks} weeks of data")

    # Get weekly date index for walk-forward forecasts
    first_symbol = available_symbols[0]
    weekly_df = prices_dict[first_symbol]["close"].resample("W-FRI").last().dropna()
    weekly_dates = weekly_df.index[-min_weeks:]

    # Align all price series
    for symbol in available_symbols:
        if symbol in weekly_prices:
            weekly_prices[symbol] = weekly_prices[symbol][-min_weeks:]

    # Load REAL historical signals (news sentiment, fundamentals)
    logger.info("[PPO_LSTM] Loading historical signals (news, fundamentals)...")
    signals = build_rl_training_signals(
        prices_dict=prices_dict,
        symbols=available_symbols,
        start_date=start_date,
        end_date=end_date,
    )

    # Align signals to the common week count
    for symbol in available_symbols:
        if symbol in signals:
            for signal_name in signals[symbol]:
                signal_arr = signals[symbol][signal_name]
                if len(signal_arr) >= min_weeks:
                    signals[symbol][signal_name] = signal_arr[-min_weeks + 1:]
                else:
                    # Pad with zeros if not enough data
                    padded = np.zeros(min_weeks - 1)
                    padded[-len(signal_arr):] = signal_arr[:min_weeks - 1] if len(signal_arr) > 0 else 0
                    signals[symbol][signal_name] = padded
        else:
            # No signals for this symbol, use zeros
            signals[symbol] = {
                "news_sentiment": np.zeros(min_weeks - 1),
                "gross_margin": np.zeros(min_weeks - 1),
                "operating_margin": np.zeros(min_weeks - 1),
                "net_margin": np.zeros(min_weeks - 1),
                "current_ratio": np.zeros(min_weeks - 1),
                "debt_to_equity": np.zeros(min_weeks - 1),
                "fundamental_age": np.ones(min_weeks - 1),
            }

    # Generate walk-forward forecast features (use snapshots if available)
    use_snapshots = _snapshots_available("lstm")
    logger.info(f"[PPO_LSTM] Generating walk-forward forecast features (snapshots={use_snapshots})...")
    lstm_predictions = build_forecast_features(
        weekly_prices=weekly_prices,
        weekly_dates=weekly_dates,
        symbols=available_symbols,
        forecaster_type="lstm",
        use_model_snapshots=use_snapshots,
    )

    # Align forecast features to common week count
    for symbol in available_symbols:
        if symbol in lstm_predictions:
            pred_arr = lstm_predictions[symbol]
            if len(pred_arr) >= min_weeks - 1:
                lstm_predictions[symbol] = pred_arr[-(min_weeks - 1):]
            else:
                padded = np.zeros(min_weeks - 1)
                padded[-len(pred_arr):] = pred_arr
                lstm_predictions[symbol] = padded
        else:
            lstm_predictions[symbol] = np.zeros(min_weeks - 1)

    # Build training data
    training_data = build_training_data(
        prices=weekly_prices,
        signals=signals,
        lstm_predictions=lstm_predictions,
        symbol_order=available_symbols,
    )

    logger.info(f"[PPO_LSTM] Training data: {training_data.n_weeks} weeks, {training_data.n_stocks} stocks")
    logger.info(f"[PPO_LSTM] Signals loaded for {len([s for s in signals if 'news_sentiment' in signals[s]])} symbols")

    # Train PPO
    logger.info("[PPO_LSTM] Starting PPO training...")
    t0 = time.time()
    result = train_ppo_lstm(training_data, config)
    t_train = time.time() - t0
    logger.info(f"[PPO_LSTM] Training complete in {t_train:.1f}s")
    logger.info(f"[PPO_LSTM] Eval sharpe: {result.eval_sharpe:.4f}, CAGR: {result.eval_cagr*100:.2f}%")

    # Get prior version info
    prior_version = storage.read_current_version()
    prior_sharpe = None
    if prior_version:
        prior_metadata = storage.read_metadata(prior_version)
        if prior_metadata:
            prior_sharpe = prior_metadata["metrics"].get("eval_sharpe")

    # Decide on promotion (first model auto-promotes)
    if prior_version is None:
        promoted = True
        logger.info("[PPO_LSTM] First model - auto-promoting")
    else:
        promoted = prior_sharpe is None or result.eval_sharpe > prior_sharpe
        logger.info(f"[PPO_LSTM] Promotion: {'YES' if promoted else 'NO'}")

    # Create metadata
    metadata = create_ppo_lstm_metadata(
        version=version,
        data_window_start=start_date.isoformat(),
        data_window_end=end_date.isoformat(),
        symbols=available_symbols,
        config=config,
        promoted=promoted,
        prior_version=prior_version,
        policy_loss=result.final_policy_loss,
        value_loss=result.final_value_loss,
        avg_episode_return=result.avg_episode_return,
        avg_episode_sharpe=result.avg_episode_sharpe,
        eval_sharpe=result.eval_sharpe,
        eval_cagr=result.eval_cagr,
        eval_max_drawdown=result.eval_max_drawdown,
    )

    # Write artifacts
    logger.info(f"[PPO_LSTM] Writing artifacts for version {version}...")
    storage.write_artifacts(
        version=version,
        model=result.model,
        scaler=result.scaler,
        config=config,
        symbol_order=available_symbols,
        metadata=metadata,
    )

    # Promote if appropriate
    if promoted:
        storage.promote_version(version)
        logger.info(f"[PPO_LSTM] Version {version} promoted to current")

    # Optionally push to HuggingFace Hub
    hf_repo = None
    hf_url = None
    storage_backend = get_storage_backend()
    hf_model_repo = get_hf_ppo_lstm_model_repo()

    if storage_backend == "hf" and hf_model_repo:
        try:
            from brain_api.storage.huggingface import HuggingFaceModelStorage

            hf_storage = HuggingFaceModelStorage(repo_id=hf_model_repo)
            hf_info = hf_storage.upload_model(
                version=version,
                model=result.model,
                feature_scaler=result.scaler,
                config=config,
                metadata=metadata,
                make_current=promoted,
            )
            hf_repo = hf_info.repo_id
            hf_url = f"https://huggingface.co/{hf_info.repo_id}/tree/{version}"
            logger.info(f"[PPO_LSTM] Model uploaded to HuggingFace: {hf_url}")
        except Exception as e:
            logger.error(f"[PPO_LSTM] Failed to upload model to HuggingFace: {e}")
            # Don't fail the training request if HF upload fails

    return PPOLSTMTrainResponse(
        version=version,
        data_window_start=start_date.isoformat(),
        data_window_end=end_date.isoformat(),
        metrics={
            "policy_loss": result.final_policy_loss,
            "value_loss": result.final_value_loss,
            "avg_episode_return": result.avg_episode_return,
            "avg_episode_sharpe": result.avg_episode_sharpe,
            "eval_sharpe": result.eval_sharpe,
            "eval_cagr": result.eval_cagr,
            "eval_max_drawdown": result.eval_max_drawdown,
        },
        promoted=promoted,
        prior_version=prior_version,
        symbols_used=available_symbols,
        hf_repo=hf_repo,
        hf_url=hf_url,
    )


# ============================================================================
# PPO + LSTM Fine-tuning Endpoint (weekly)
# ============================================================================


@router.post("/ppo_lstm/finetune", response_model=PPOLSTMTrainResponse)
def finetune_ppo_lstm_endpoint(
    storage: PPOLSTMLocalStorage = Depends(get_ppo_lstm_storage),
    symbols: list[str] = Depends(get_top15_symbols),
) -> PPOLSTMTrainResponse:
    """Fine-tune PPO + LSTM on recent 26-week data.

    This endpoint is called weekly (Sunday cron) to adapt the model to
    recent market conditions. It:
    1. Loads the current promoted model
    2. Fine-tunes on the last 26 weeks of data
    3. Uses lower learning rate and fewer timesteps
    4. Promotes if it beats the prior model

    Requires a prior trained model to exist.

    Returns:
        Training result including version, metrics, and promotion status.
    """
    import time
    from datetime import timedelta

    import numpy as np

    t_start = time.time()
    logger.info("[PPO_LSTM Finetune] Starting fine-tuning")

    # Load prior model (required for fine-tuning)
    prior_version = storage.read_current_version()
    if prior_version is None:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400,
            detail="No prior PPO_LSTM model to fine-tune. Train a full model first with POST /train/ppo_lstm/full"
        )

    logger.info(f"[PPO_LSTM Finetune] Loading prior model: {prior_version}")
    prior_artifacts = storage.load_current_artifacts()
    prior_config = prior_artifacts.config

    # Use 26-week lookback for fine-tuning
    finetune_config = PPOFinetuneConfig()
    from datetime import date
    end_date = date.today()
    start_date = end_date - timedelta(weeks=finetune_config.lookback_weeks + 4)  # Extra buffer

    logger.info(f"[PPO_LSTM Finetune] Data window: {start_date} to {end_date}")
    logger.info(f"[PPO_LSTM Finetune] Symbols: {symbols}")

    # Compute version for fine-tuned model
    version = ppo_lstm_compute_version(start_date, end_date, symbols, prior_config)
    version = f"{version}-ft"  # Mark as fine-tuned
    logger.info(f"[PPO_LSTM Finetune] Version: {version}")

    # Check if already exists (idempotent)
    if storage.version_exists(version):
        logger.info(f"[PPO_LSTM Finetune] Version {version} already exists (idempotent)")
        existing_metadata = storage.read_metadata(version)
        if existing_metadata:
            return PPOLSTMTrainResponse(
                version=version,
                data_window_start=existing_metadata["data_window"]["start"],
                data_window_end=existing_metadata["data_window"]["end"],
                metrics=existing_metadata["metrics"],
                promoted=existing_metadata["promoted"],
                prior_version=existing_metadata.get("prior_version"),
                symbols_used=existing_metadata["symbols"],
            )

    # Load recent price data
    logger.info("[PPO_LSTM Finetune] Loading price data...")
    t0 = time.time()
    prices_dict = load_prices_yfinance(symbols, start_date, end_date)
    t_prices = time.time() - t0
    logger.info(f"[PPO_LSTM Finetune] Loaded prices in {t_prices:.1f}s")

    if len(prices_dict) == 0:
        raise ValueError("No price data available")

    # Filter and align symbols
    available_symbols = [s for s in symbols if s in prices_dict]
    if len(available_symbols) < 5:
        raise ValueError(f"Need at least 5 symbols, got {len(available_symbols)}")

    # Resample to weekly
    weekly_prices = {}
    for symbol in available_symbols:
        df = prices_dict[symbol]
        if df is not None and len(df) > 0:
            weekly = df["close"].resample("W-FRI").last().dropna()
            weekly_prices[symbol] = weekly.values

    min_weeks = min(len(weekly_prices[s]) for s in available_symbols if s in weekly_prices)

    # Get weekly date index for walk-forward forecasts
    first_symbol = available_symbols[0]
    weekly_df = prices_dict[first_symbol]["close"].resample("W-FRI").last().dropna()
    weekly_dates = weekly_df.index[-min_weeks:]

    for symbol in available_symbols:
        if symbol in weekly_prices:
            weekly_prices[symbol] = weekly_prices[symbol][-min_weeks:]

    # Load REAL historical signals
    logger.info("[PPO_LSTM Finetune] Loading historical signals...")
    signals = build_rl_training_signals(
        prices_dict=prices_dict,
        symbols=available_symbols,
        start_date=start_date,
        end_date=end_date,
    )

    # Align signals to the common week count
    for symbol in available_symbols:
        if symbol in signals:
            for signal_name in signals[symbol]:
                signal_arr = signals[symbol][signal_name]
                if len(signal_arr) >= min_weeks:
                    signals[symbol][signal_name] = signal_arr[-min_weeks + 1:]
                else:
                    padded = np.zeros(min_weeks - 1)
                    padded[-len(signal_arr):] = signal_arr[:min_weeks - 1] if len(signal_arr) > 0 else 0
                    signals[symbol][signal_name] = padded
        else:
            signals[symbol] = {
                "news_sentiment": np.zeros(min_weeks - 1),
                "gross_margin": np.zeros(min_weeks - 1),
                "operating_margin": np.zeros(min_weeks - 1),
                "net_margin": np.zeros(min_weeks - 1),
                "current_ratio": np.zeros(min_weeks - 1),
                "debt_to_equity": np.zeros(min_weeks - 1),
                "fundamental_age": np.ones(min_weeks - 1),
            }

    # Generate walk-forward forecast features (use snapshots if available)
    use_snapshots = _snapshots_available("lstm")
    lstm_predictions = build_forecast_features(
        weekly_prices=weekly_prices,
        weekly_dates=weekly_dates,
        symbols=available_symbols,
        forecaster_type="lstm",
        use_model_snapshots=use_snapshots,
    )

    # Align forecast features
    for symbol in available_symbols:
        if symbol in lstm_predictions:
            pred_arr = lstm_predictions[symbol]
            if len(pred_arr) >= min_weeks - 1:
                lstm_predictions[symbol] = pred_arr[-(min_weeks - 1):]
            else:
                padded = np.zeros(min_weeks - 1)
                padded[-len(pred_arr):] = pred_arr
                lstm_predictions[symbol] = padded
        else:
            lstm_predictions[symbol] = np.zeros(min_weeks - 1)

    training_data = build_training_data(
        prices=weekly_prices,
        signals=signals,
        lstm_predictions=lstm_predictions,
        symbol_order=available_symbols,
    )

    logger.info(f"[PPO_LSTM Finetune] Training data: {training_data.n_weeks} weeks")

    # Fine-tune
    logger.info("[PPO_LSTM Finetune] Starting fine-tuning...")
    t0 = time.time()
    result = finetune_ppo_lstm(
        training_data=training_data,
        prior_model=prior_artifacts.model,
        prior_scaler=prior_artifacts.scaler,
        prior_config=prior_config,
        finetune_config=finetune_config,
    )
    t_train = time.time() - t0
    logger.info(f"[PPO_LSTM Finetune] Complete in {t_train:.1f}s")

    # Get prior sharpe for comparison
    prior_metadata = storage.read_metadata(prior_version)
    prior_sharpe = prior_metadata["metrics"].get("eval_sharpe") if prior_metadata else None

    # Decide on promotion (must beat prior)
    promoted = prior_sharpe is None or result.eval_sharpe > prior_sharpe
    logger.info(f"[PPO_LSTM Finetune] Prior sharpe: {prior_sharpe}, New sharpe: {result.eval_sharpe}")
    logger.info(f"[PPO_LSTM Finetune] Promotion: {'YES' if promoted else 'NO'}")

    # Create metadata
    metadata = create_ppo_lstm_metadata(
        version=version,
        data_window_start=start_date.isoformat(),
        data_window_end=end_date.isoformat(),
        symbols=available_symbols,
        config=prior_config,
        promoted=promoted,
        prior_version=prior_version,
        policy_loss=result.final_policy_loss,
        value_loss=result.final_value_loss,
        avg_episode_return=result.avg_episode_return,
        avg_episode_sharpe=result.avg_episode_sharpe,
        eval_sharpe=result.eval_sharpe,
        eval_cagr=result.eval_cagr,
        eval_max_drawdown=result.eval_max_drawdown,
    )

    # Write artifacts
    storage.write_artifacts(
        version=version,
        model=result.model,
        scaler=result.scaler,
        config=prior_config,
        symbol_order=available_symbols,
        metadata=metadata,
    )

    # Promote if better
    if promoted:
        storage.promote_version(version)
        logger.info(f"[PPO_LSTM Finetune] Version {version} promoted to current")

    # Optionally push to HuggingFace Hub
    hf_repo = None
    hf_url = None
    storage_backend = get_storage_backend()
    hf_model_repo = get_hf_ppo_lstm_model_repo()

    if storage_backend == "hf" and hf_model_repo:
        try:
            from brain_api.storage.huggingface import HuggingFaceModelStorage

            hf_storage = HuggingFaceModelStorage(repo_id=hf_model_repo)
            hf_info = hf_storage.upload_model(
                version=version,
                model=result.model,
                feature_scaler=result.scaler,
                config=prior_config,
                metadata=metadata,
                make_current=promoted,
            )
            hf_repo = hf_info.repo_id
            hf_url = f"https://huggingface.co/{hf_info.repo_id}/tree/{version}"
            logger.info(f"[PPO_LSTM Finetune] Model uploaded to HuggingFace: {hf_url}")
        except Exception as e:
            logger.error(f"[PPO_LSTM Finetune] Failed to upload model to HuggingFace: {e}")

    return PPOLSTMTrainResponse(
        version=version,
        data_window_start=start_date.isoformat(),
        data_window_end=end_date.isoformat(),
        metrics={
            "policy_loss": result.final_policy_loss,
            "value_loss": result.final_value_loss,
            "avg_episode_return": result.avg_episode_return,
            "avg_episode_sharpe": result.avg_episode_sharpe,
            "eval_sharpe": result.eval_sharpe,
            "eval_cagr": result.eval_cagr,
            "eval_max_drawdown": result.eval_max_drawdown,
        },
        promoted=promoted,
        prior_version=prior_version,
        symbols_used=available_symbols,
        hf_repo=hf_repo,
        hf_url=hf_url,
    )


# ============================================================================
# PPO + PatchTST Training Endpoint
# ============================================================================


class PPOPatchTSTTrainResponse(BaseModel):
    """Response model for PPO + PatchTST training endpoint."""

    version: str
    data_window_start: str  # YYYY-MM-DD
    data_window_end: str  # YYYY-MM-DD
    metrics: dict[str, Any]
    promoted: bool
    prior_version: str | None = None
    symbols_used: list[str]
    hf_repo: str | None = None  # HuggingFace repo if uploaded
    hf_url: str | None = None  # URL to model on HuggingFace


def get_ppo_patchtst_storage() -> PPOPatchTSTLocalStorage:
    """Get the PPO + PatchTST storage instance."""
    return PPOPatchTSTLocalStorage()


def get_ppo_patchtst_config() -> PPOPatchTSTConfig:
    """Get PPO + PatchTST configuration."""
    return DEFAULT_PPO_PATCHTST_CONFIG


@router.post("/ppo_patchtst/full", response_model=PPOPatchTSTTrainResponse)
def train_ppo_patchtst_endpoint(
    storage: PPOPatchTSTLocalStorage = Depends(get_ppo_patchtst_storage),
    symbols: list[str] = Depends(get_top15_symbols),
    config: PPOPatchTSTConfig = Depends(get_ppo_patchtst_config),
) -> PPOPatchTSTTrainResponse:
    """Train PPO portfolio allocator using PatchTST forecasts.

    This endpoint:
    1. Loads historical price data and signals
    2. Generates PatchTST forecast features (from pre-trained PatchTST)
    3. Trains PPO policy for portfolio allocation
    4. Evaluates on held-out data
    5. Promotes if first model or beats prior

    Returns:
        Training result including version, metrics, and promotion status.
    """
    import time

    import numpy as np

    # Resolve window from API config
    start_date, end_date = resolve_training_window()
    logger.info(f"[PPO_PatchTST] Starting training for {len(symbols)} symbols")
    logger.info(f"[PPO_PatchTST] Data window: {start_date} to {end_date}")
    logger.info(f"[PPO_PatchTST] Symbols: {symbols}")

    # Compute deterministic version
    version = ppo_patchtst_compute_version(start_date, end_date, symbols, config)
    logger.info(f"[PPO_PatchTST] Computed version: {version}")

    # Check if this version already exists (idempotent)
    if storage.version_exists(version):
        logger.info(f"[PPO_PatchTST] Version {version} already exists (idempotent)")
        existing_metadata = storage.read_metadata(version)
        if existing_metadata:
            return PPOPatchTSTTrainResponse(
                version=version,
                data_window_start=existing_metadata["data_window"]["start"],
                data_window_end=existing_metadata["data_window"]["end"],
                metrics=existing_metadata["metrics"],
                promoted=existing_metadata["promoted"],
                prior_version=existing_metadata.get("prior_version"),
                symbols_used=existing_metadata["symbols"],
            )

    # Load price data
    logger.info("[PPO_PatchTST] Loading price data...")
    t0 = time.time()
    prices_dict = load_prices_yfinance(symbols, start_date, end_date)
    t_prices = time.time() - t0
    logger.info(f"[PPO_PatchTST] Loaded prices for {len(prices_dict)}/{len(symbols)} symbols in {t_prices:.1f}s")

    if len(prices_dict) == 0:
        raise ValueError("No price data available for training")

    # Filter symbols to those with price data
    available_symbols = [s for s in symbols if s in prices_dict]
    if len(available_symbols) < 5:
        raise ValueError(f"Need at least 5 symbols with data, got {len(available_symbols)}")

    # Resample prices to weekly (Friday close)
    logger.info("[PPO_PatchTST] Resampling prices to weekly...")
    weekly_prices = {}
    for symbol in available_symbols:
        df = prices_dict[symbol]
        if df is not None and len(df) > 0:
            weekly = df["close"].resample("W-FRI").last().dropna()
            weekly_prices[symbol] = weekly.values

    # Determine minimum length across all symbols
    min_weeks = min(len(weekly_prices[s]) for s in available_symbols if s in weekly_prices)
    logger.info(f"[PPO_PatchTST] Using {min_weeks} weeks of data")

    # Get weekly date index for walk-forward forecasts
    first_symbol = available_symbols[0]
    weekly_df = prices_dict[first_symbol]["close"].resample("W-FRI").last().dropna()
    weekly_dates = weekly_df.index[-min_weeks:]

    # Align all price series
    for symbol in available_symbols:
        if symbol in weekly_prices:
            weekly_prices[symbol] = weekly_prices[symbol][-min_weeks:]

    # Load REAL historical signals (news sentiment, fundamentals)
    logger.info("[PPO_PatchTST] Loading historical signals (news, fundamentals)...")
    signals = build_rl_training_signals(
        prices_dict=prices_dict,
        symbols=available_symbols,
        start_date=start_date,
        end_date=end_date,
    )

    # Align signals to the common week count
    for symbol in available_symbols:
        if symbol in signals:
            for signal_name in signals[symbol]:
                signal_arr = signals[symbol][signal_name]
                if len(signal_arr) >= min_weeks:
                    signals[symbol][signal_name] = signal_arr[-min_weeks + 1:]
                else:
                    padded = np.zeros(min_weeks - 1)
                    padded[-len(signal_arr):] = signal_arr[:min_weeks - 1] if len(signal_arr) > 0 else 0
                    signals[symbol][signal_name] = padded
        else:
            signals[symbol] = {
                "news_sentiment": np.zeros(min_weeks - 1),
                "gross_margin": np.zeros(min_weeks - 1),
                "operating_margin": np.zeros(min_weeks - 1),
                "net_margin": np.zeros(min_weeks - 1),
                "current_ratio": np.zeros(min_weeks - 1),
                "debt_to_equity": np.zeros(min_weeks - 1),
                "fundamental_age": np.ones(min_weeks - 1),
            }

    # Generate walk-forward forecast features (use snapshots if available)
    use_snapshots = _snapshots_available("patchtst")
    logger.info(f"[PPO_PatchTST] Generating walk-forward forecast features (snapshots={use_snapshots})...")
    patchtst_predictions = build_forecast_features(
        weekly_prices=weekly_prices,
        weekly_dates=weekly_dates,
        symbols=available_symbols,
        forecaster_type="patchtst",
        use_model_snapshots=use_snapshots,
    )

    # Align forecast features to common week count
    for symbol in available_symbols:
        if symbol in patchtst_predictions:
            pred_arr = patchtst_predictions[symbol]
            if len(pred_arr) >= min_weeks - 1:
                patchtst_predictions[symbol] = pred_arr[-(min_weeks - 1):]
            else:
                padded = np.zeros(min_weeks - 1)
                padded[-len(pred_arr):] = pred_arr
                patchtst_predictions[symbol] = padded
        else:
            patchtst_predictions[symbol] = np.zeros(min_weeks - 1)

    # Build training data (reuse from ppo_lstm)
    training_data = build_training_data(
        prices=weekly_prices,
        signals=signals,
        lstm_predictions=patchtst_predictions,  # Same structure, different source
        symbol_order=available_symbols,
    )

    logger.info(f"[PPO_PatchTST] Training data: {training_data.n_weeks} weeks, {training_data.n_stocks} stocks")
    logger.info(f"[PPO_PatchTST] Signals loaded for {len([s for s in signals if 'news_sentiment' in signals[s]])} symbols")

    # Train PPO
    logger.info("[PPO_PatchTST] Starting PPO training...")
    t0 = time.time()
    result = train_ppo_patchtst(training_data, config)
    t_train = time.time() - t0
    logger.info(f"[PPO_PatchTST] Training complete in {t_train:.1f}s")
    logger.info(f"[PPO_PatchTST] Eval sharpe: {result.eval_sharpe:.4f}, CAGR: {result.eval_cagr*100:.2f}%")

    # Get prior version info
    prior_version = storage.read_current_version()
    prior_sharpe = None
    if prior_version:
        prior_metadata = storage.read_metadata(prior_version)
        if prior_metadata:
            prior_sharpe = prior_metadata["metrics"].get("eval_sharpe")

    # Decide on promotion (first model auto-promotes)
    if prior_version is None:
        promoted = True
        logger.info("[PPO_PatchTST] First model - auto-promoting")
    else:
        promoted = prior_sharpe is None or result.eval_sharpe > prior_sharpe
        logger.info(f"[PPO_PatchTST] Promotion: {'YES' if promoted else 'NO'}")

    # Create metadata
    metadata = create_ppo_patchtst_metadata(
        version=version,
        data_window_start=start_date.isoformat(),
        data_window_end=end_date.isoformat(),
        symbols=available_symbols,
        config=config,
        promoted=promoted,
        prior_version=prior_version,
        policy_loss=result.final_policy_loss,
        value_loss=result.final_value_loss,
        avg_episode_return=result.avg_episode_return,
        avg_episode_sharpe=result.avg_episode_sharpe,
        eval_sharpe=result.eval_sharpe,
        eval_cagr=result.eval_cagr,
        eval_max_drawdown=result.eval_max_drawdown,
    )

    # Write artifacts
    logger.info(f"[PPO_PatchTST] Writing artifacts for version {version}...")
    storage.write_artifacts(
        version=version,
        model=result.model,
        scaler=result.scaler,
        config=config,
        symbol_order=available_symbols,
        metadata=metadata,
    )

    # Promote if appropriate
    if promoted:
        storage.promote_version(version)
        logger.info(f"[PPO_PatchTST] Version {version} promoted to current")

    # Optionally push to HuggingFace Hub
    hf_repo = None
    hf_url = None
    storage_backend = get_storage_backend()
    hf_model_repo = get_hf_ppo_patchtst_model_repo()

    if storage_backend == "hf" and hf_model_repo:
        try:
            from brain_api.storage.huggingface import HuggingFaceModelStorage

            hf_storage = HuggingFaceModelStorage(repo_id=hf_model_repo)
            hf_info = hf_storage.upload_model(
                version=version,
                model=result.model,
                feature_scaler=result.scaler,
                config=config,
                metadata=metadata,
                make_current=promoted,
            )
            hf_repo = hf_info.repo_id
            hf_url = f"https://huggingface.co/{hf_info.repo_id}/tree/{version}"
            logger.info(f"[PPO_PatchTST] Model uploaded to HuggingFace: {hf_url}")
        except Exception as e:
            logger.error(f"[PPO_PatchTST] Failed to upload model to HuggingFace: {e}")

    return PPOPatchTSTTrainResponse(
        version=version,
        data_window_start=start_date.isoformat(),
        data_window_end=end_date.isoformat(),
        metrics={
            "policy_loss": result.final_policy_loss,
            "value_loss": result.final_value_loss,
            "avg_episode_return": result.avg_episode_return,
            "avg_episode_sharpe": result.avg_episode_sharpe,
            "eval_sharpe": result.eval_sharpe,
            "eval_cagr": result.eval_cagr,
            "eval_max_drawdown": result.eval_max_drawdown,
        },
        promoted=promoted,
        prior_version=prior_version,
        symbols_used=available_symbols,
        hf_repo=hf_repo,
        hf_url=hf_url,
    )


# ============================================================================
# PPO + PatchTST Fine-tuning Endpoint (weekly)
# ============================================================================


@router.post("/ppo_patchtst/finetune", response_model=PPOPatchTSTTrainResponse)
def finetune_ppo_patchtst_endpoint(
    storage: PPOPatchTSTLocalStorage = Depends(get_ppo_patchtst_storage),
    symbols: list[str] = Depends(get_top15_symbols),
) -> PPOPatchTSTTrainResponse:
    """Fine-tune PPO + PatchTST on recent 26-week data.

    This endpoint is called weekly (Sunday cron) to adapt the model to
    recent market conditions. It:
    1. Loads the current promoted model
    2. Fine-tunes on the last 26 weeks of data
    3. Uses lower learning rate and fewer timesteps
    4. Promotes if it beats the prior model

    Requires a prior trained model to exist.

    Returns:
        Training result including version, metrics, and promotion status.
    """
    import time
    from datetime import timedelta

    import numpy as np

    t_start = time.time()
    logger.info("[PPO_PatchTST Finetune] Starting fine-tuning")

    # Load prior model (required for fine-tuning)
    prior_version = storage.read_current_version()
    if prior_version is None:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400,
            detail="No prior PPO_PatchTST model to fine-tune. Train a full model first with POST /train/ppo_patchtst/full"
        )

    logger.info(f"[PPO_PatchTST Finetune] Loading prior model: {prior_version}")
    prior_artifacts = storage.load_current_artifacts()
    prior_config = prior_artifacts.config

    # Use 26-week lookback for fine-tuning
    finetune_config = PPOFinetuneConfig()
    from datetime import date
    end_date = date.today()
    start_date = end_date - timedelta(weeks=finetune_config.lookback_weeks + 4)  # Extra buffer

    logger.info(f"[PPO_PatchTST Finetune] Data window: {start_date} to {end_date}")
    logger.info(f"[PPO_PatchTST Finetune] Symbols: {symbols}")

    # Compute version for fine-tuned model
    version = ppo_patchtst_compute_version(start_date, end_date, symbols, prior_config)
    version = f"{version}-ft"  # Mark as fine-tuned
    logger.info(f"[PPO_PatchTST Finetune] Version: {version}")

    # Check if already exists (idempotent)
    if storage.version_exists(version):
        logger.info(f"[PPO_PatchTST Finetune] Version {version} already exists (idempotent)")
        existing_metadata = storage.read_metadata(version)
        if existing_metadata:
            return PPOPatchTSTTrainResponse(
                version=version,
                data_window_start=existing_metadata["data_window"]["start"],
                data_window_end=existing_metadata["data_window"]["end"],
                metrics=existing_metadata["metrics"],
                promoted=existing_metadata["promoted"],
                prior_version=existing_metadata.get("prior_version"),
                symbols_used=existing_metadata["symbols"],
            )

    # Load recent price data
    logger.info("[PPO_PatchTST Finetune] Loading price data...")
    t0 = time.time()
    prices_dict = load_prices_yfinance(symbols, start_date, end_date)
    t_prices = time.time() - t0
    logger.info(f"[PPO_PatchTST Finetune] Loaded prices in {t_prices:.1f}s")

    if len(prices_dict) == 0:
        raise ValueError("No price data available")

    # Filter and align symbols
    available_symbols = [s for s in symbols if s in prices_dict]
    if len(available_symbols) < 5:
        raise ValueError(f"Need at least 5 symbols, got {len(available_symbols)}")

    # Resample to weekly
    weekly_prices = {}
    for symbol in available_symbols:
        df = prices_dict[symbol]
        if df is not None and len(df) > 0:
            weekly = df["close"].resample("W-FRI").last().dropna()
            weekly_prices[symbol] = weekly.values

    min_weeks = min(len(weekly_prices[s]) for s in available_symbols if s in weekly_prices)

    # Get weekly date index for walk-forward forecasts
    first_symbol = available_symbols[0]
    weekly_df = prices_dict[first_symbol]["close"].resample("W-FRI").last().dropna()
    weekly_dates = weekly_df.index[-min_weeks:]

    for symbol in available_symbols:
        if symbol in weekly_prices:
            weekly_prices[symbol] = weekly_prices[symbol][-min_weeks:]

    # Load REAL historical signals
    logger.info("[PPO_PatchTST Finetune] Loading historical signals...")
    signals = build_rl_training_signals(
        prices_dict=prices_dict,
        symbols=available_symbols,
        start_date=start_date,
        end_date=end_date,
    )

    # Align signals to the common week count
    for symbol in available_symbols:
        if symbol in signals:
            for signal_name in signals[symbol]:
                signal_arr = signals[symbol][signal_name]
                if len(signal_arr) >= min_weeks:
                    signals[symbol][signal_name] = signal_arr[-min_weeks + 1:]
                else:
                    padded = np.zeros(min_weeks - 1)
                    padded[-len(signal_arr):] = signal_arr[:min_weeks - 1] if len(signal_arr) > 0 else 0
                    signals[symbol][signal_name] = padded
        else:
            signals[symbol] = {
                "news_sentiment": np.zeros(min_weeks - 1),
                "gross_margin": np.zeros(min_weeks - 1),
                "operating_margin": np.zeros(min_weeks - 1),
                "net_margin": np.zeros(min_weeks - 1),
                "current_ratio": np.zeros(min_weeks - 1),
                "debt_to_equity": np.zeros(min_weeks - 1),
                "fundamental_age": np.ones(min_weeks - 1),
            }

    # Generate walk-forward forecast features (use snapshots if available)
    use_snapshots = _snapshots_available("patchtst")
    patchtst_predictions = build_forecast_features(
        weekly_prices=weekly_prices,
        weekly_dates=weekly_dates,
        symbols=available_symbols,
        forecaster_type="patchtst",
        use_model_snapshots=use_snapshots,
    )

    # Align forecast features
    for symbol in available_symbols:
        if symbol in patchtst_predictions:
            pred_arr = patchtst_predictions[symbol]
            if len(pred_arr) >= min_weeks - 1:
                patchtst_predictions[symbol] = pred_arr[-(min_weeks - 1):]
            else:
                padded = np.zeros(min_weeks - 1)
                padded[-len(pred_arr):] = pred_arr
                patchtst_predictions[symbol] = padded
        else:
            patchtst_predictions[symbol] = np.zeros(min_weeks - 1)

    training_data = build_training_data(
        prices=weekly_prices,
        signals=signals,
        lstm_predictions=patchtst_predictions,  # Same structure
        symbol_order=available_symbols,
    )

    logger.info(f"[PPO_PatchTST Finetune] Training data: {training_data.n_weeks} weeks")

    # Fine-tune
    logger.info("[PPO_PatchTST Finetune] Starting fine-tuning...")
    t0 = time.time()
    result = finetune_ppo_patchtst(
        training_data=training_data,
        prior_model=prior_artifacts.model,
        prior_scaler=prior_artifacts.scaler,
        prior_config=prior_config,
        finetune_config=finetune_config,
    )
    t_train = time.time() - t0
    logger.info(f"[PPO_PatchTST Finetune] Complete in {t_train:.1f}s")

    # Get prior sharpe for comparison
    prior_metadata = storage.read_metadata(prior_version)
    prior_sharpe = prior_metadata["metrics"].get("eval_sharpe") if prior_metadata else None

    # Decide on promotion (must beat prior)
    promoted = prior_sharpe is None or result.eval_sharpe > prior_sharpe
    logger.info(f"[PPO_PatchTST Finetune] Prior sharpe: {prior_sharpe}, New sharpe: {result.eval_sharpe}")
    logger.info(f"[PPO_PatchTST Finetune] Promotion: {'YES' if promoted else 'NO'}")

    # Create metadata
    metadata = create_ppo_patchtst_metadata(
        version=version,
        data_window_start=start_date.isoformat(),
        data_window_end=end_date.isoformat(),
        symbols=available_symbols,
        config=prior_config,
        promoted=promoted,
        prior_version=prior_version,
        policy_loss=result.final_policy_loss,
        value_loss=result.final_value_loss,
        avg_episode_return=result.avg_episode_return,
        avg_episode_sharpe=result.avg_episode_sharpe,
        eval_sharpe=result.eval_sharpe,
        eval_cagr=result.eval_cagr,
        eval_max_drawdown=result.eval_max_drawdown,
    )

    # Write artifacts
    storage.write_artifacts(
        version=version,
        model=result.model,
        scaler=result.scaler,
        config=prior_config,
        symbol_order=available_symbols,
        metadata=metadata,
    )

    # Promote if better
    if promoted:
        storage.promote_version(version)
        logger.info(f"[PPO_PatchTST Finetune] Version {version} promoted to current")

    # Optionally push to HuggingFace Hub
    hf_repo = None
    hf_url = None
    storage_backend = get_storage_backend()
    hf_model_repo = get_hf_ppo_patchtst_model_repo()

    if storage_backend == "hf" and hf_model_repo:
        try:
            from brain_api.storage.huggingface import HuggingFaceModelStorage

            hf_storage = HuggingFaceModelStorage(repo_id=hf_model_repo)
            hf_info = hf_storage.upload_model(
                version=version,
                model=result.model,
                feature_scaler=result.scaler,
                config=prior_config,
                metadata=metadata,
                make_current=promoted,
            )
            hf_repo = hf_info.repo_id
            hf_url = f"https://huggingface.co/{hf_info.repo_id}/tree/{version}"
            logger.info(f"[PPO_PatchTST Finetune] Model uploaded to HuggingFace: {hf_url}")
        except Exception as e:
            logger.error(f"[PPO_PatchTST Finetune] Failed to upload model to HuggingFace: {e}")

    return PPOPatchTSTTrainResponse(
        version=version,
        data_window_start=start_date.isoformat(),
        data_window_end=end_date.isoformat(),
        metrics={
            "policy_loss": result.final_policy_loss,
            "value_loss": result.final_value_loss,
            "avg_episode_return": result.avg_episode_return,
            "avg_episode_sharpe": result.avg_episode_sharpe,
            "eval_sharpe": result.eval_sharpe,
            "eval_cagr": result.eval_cagr,
            "eval_max_drawdown": result.eval_max_drawdown,
        },
        promoted=promoted,
        prior_version=prior_version,
        symbols_used=available_symbols,
        hf_repo=hf_repo,
        hf_url=hf_url,
    )


# ============================================================================
# SAC + LSTM Training Endpoints
# ============================================================================


class SACLSTMTrainResponse(BaseModel):
    """Response model for SAC + LSTM training endpoint."""

    version: str
    data_window_start: str
    data_window_end: str
    metrics: dict[str, float]
    promoted: bool
    prior_version: str | None
    symbols_used: list[str]
    hf_repo: str | None = None
    hf_url: str | None = None


def get_sac_lstm_storage() -> SACLSTMLocalStorage:
    """Get the SAC + LSTM storage instance."""
    return SACLSTMLocalStorage()


def get_sac_lstm_config() -> SACLSTMConfig:
    """Get SAC + LSTM configuration."""
    return DEFAULT_SAC_LSTM_CONFIG


@router.post("/sac_lstm/full", response_model=SACLSTMTrainResponse)
def train_sac_lstm_endpoint(
    storage: SACLSTMLocalStorage = Depends(get_sac_lstm_storage),
    symbols: list[str] = Depends(get_top15_symbols),
    config: SACLSTMConfig = Depends(get_sac_lstm_config),
) -> SACLSTMTrainResponse:
    """Train SAC portfolio allocator using LSTM forecasts."""
    import time
    import numpy as np

    start_date, end_date = resolve_training_window()
    logger.info(f"[SAC_LSTM] Starting training for {len(symbols)} symbols")
    version = sac_lstm_compute_version(start_date, end_date, symbols, config)

    if storage.version_exists(version):
        existing_metadata = storage.read_metadata(version)
        if existing_metadata:
            return SACLSTMTrainResponse(
                version=version,
                data_window_start=existing_metadata["data_window"]["start"],
                data_window_end=existing_metadata["data_window"]["end"],
                metrics=existing_metadata["metrics"],
                promoted=existing_metadata["promoted"],
                prior_version=existing_metadata.get("prior_version"),
                symbols_used=existing_metadata["symbols"],
            )

    prices_dict = load_prices_yfinance(symbols, start_date, end_date)
    available_symbols = [s for s in symbols if s in prices_dict]

    weekly_prices = {}
    for symbol in available_symbols:
        df = prices_dict[symbol]
        if df is not None and len(df) > 0:
            weekly = df["close"].resample("W-FRI").last().dropna()
            weekly_prices[symbol] = weekly.values

    min_weeks = min(len(weekly_prices[s]) for s in available_symbols if s in weekly_prices)
    first_symbol = available_symbols[0]
    weekly_df = prices_dict[first_symbol]["close"].resample("W-FRI").last().dropna()
    weekly_dates = weekly_df.index[-min_weeks:]

    for symbol in available_symbols:
        if symbol in weekly_prices:
            weekly_prices[symbol] = weekly_prices[symbol][-min_weeks:]

    signals = build_rl_training_signals(prices_dict, available_symbols, start_date, end_date)
    for symbol in available_symbols:
        if symbol not in signals:
            signals[symbol] = {k: np.zeros(min_weeks - 1) for k in ["news_sentiment", "gross_margin", "operating_margin", "net_margin", "current_ratio", "debt_to_equity"]}
            signals[symbol]["fundamental_age"] = np.ones(min_weeks - 1)

    use_snapshots = _snapshots_available("lstm")
    lstm_predictions = build_forecast_features(weekly_prices, weekly_dates, available_symbols, "lstm", use_snapshots)
    for symbol in available_symbols:
        if symbol not in lstm_predictions:
            lstm_predictions[symbol] = np.zeros(min_weeks - 1)

    training_data = sac_build_training_data(weekly_prices, signals, lstm_predictions, available_symbols)
    result = train_sac_lstm(training_data, config)

    prior_version = storage.read_current_version()
    prior_cagr = None
    if prior_version:
        prior_metadata = storage.read_metadata(prior_version)
        if prior_metadata:
            prior_cagr = prior_metadata["metrics"].get("eval_cagr")

    promoted = prior_version is None or prior_cagr is None or result.eval_cagr > prior_cagr

    metadata = create_sac_lstm_metadata(
        version=version, data_window_start=start_date.isoformat(), data_window_end=end_date.isoformat(),
        symbols=available_symbols, config=config, promoted=promoted, prior_version=prior_version,
        actor_loss=result.final_actor_loss, critic_loss=result.final_critic_loss,
        avg_episode_return=result.avg_episode_return, avg_episode_sharpe=result.avg_episode_sharpe,
        eval_sharpe=result.eval_sharpe, eval_cagr=result.eval_cagr, eval_max_drawdown=result.eval_max_drawdown,
    )

    storage.write_artifacts(version, result.actor, result.critic, result.critic_target, result.log_alpha,
                            result.scaler, config, available_symbols, metadata)
    if promoted:
        storage.promote_version(version)

    return SACLSTMTrainResponse(
        version=version, data_window_start=start_date.isoformat(), data_window_end=end_date.isoformat(),
        metrics={"actor_loss": result.final_actor_loss, "critic_loss": result.final_critic_loss,
                 "avg_episode_return": result.avg_episode_return, "avg_episode_sharpe": result.avg_episode_sharpe,
                 "eval_sharpe": result.eval_sharpe, "eval_cagr": result.eval_cagr, "eval_max_drawdown": result.eval_max_drawdown},
        promoted=promoted, prior_version=prior_version, symbols_used=available_symbols,
    )


@router.post("/sac_lstm/finetune", response_model=SACLSTMTrainResponse)
def finetune_sac_lstm_endpoint(
    storage: SACLSTMLocalStorage = Depends(get_sac_lstm_storage),
    symbols: list[str] = Depends(get_top15_symbols),
) -> SACLSTMTrainResponse:
    """Fine-tune SAC + LSTM on recent data. Requires prior trained model."""
    import time
    from datetime import timedelta
    import numpy as np

    prior_version = storage.read_current_version()
    if prior_version is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="No prior SAC_LSTM model. Train with POST /train/sac_lstm/full first")

    prior_artifacts = storage.load_current_artifacts()
    prior_config = prior_artifacts.config

    finetune_config = SACFinetuneConfig()
    from datetime import date
    end_date = date.today()
    start_date = end_date - timedelta(weeks=finetune_config.lookback_weeks + 4)

    version = f"{sac_lstm_compute_version(start_date, end_date, symbols, prior_config)}-ft"

    if storage.version_exists(version):
        existing_metadata = storage.read_metadata(version)
        if existing_metadata:
            return SACLSTMTrainResponse(
                version=version, data_window_start=existing_metadata["data_window"]["start"],
                data_window_end=existing_metadata["data_window"]["end"], metrics=existing_metadata["metrics"],
                promoted=existing_metadata["promoted"], prior_version=existing_metadata.get("prior_version"),
                symbols_used=existing_metadata["symbols"],
            )

    prices_dict = load_prices_yfinance(symbols, start_date, end_date)
    available_symbols = [s for s in symbols if s in prices_dict]

    weekly_prices = {}
    for symbol in available_symbols:
        df = prices_dict[symbol]
        if df is not None and len(df) > 0:
            weekly = df["close"].resample("W-FRI").last().dropna()
            weekly_prices[symbol] = weekly.values

    min_weeks = min(len(weekly_prices[s]) for s in available_symbols if s in weekly_prices)
    weekly_df = prices_dict[available_symbols[0]]["close"].resample("W-FRI").last().dropna()
    weekly_dates = weekly_df.index[-min_weeks:]

    for symbol in available_symbols:
        weekly_prices[symbol] = weekly_prices[symbol][-min_weeks:]

    signals = build_rl_training_signals(prices_dict, available_symbols, start_date, end_date)
    for symbol in available_symbols:
        if symbol not in signals:
            signals[symbol] = {k: np.zeros(min_weeks - 1) for k in ["news_sentiment", "gross_margin", "operating_margin", "net_margin", "current_ratio", "debt_to_equity"]}
            signals[symbol]["fundamental_age"] = np.ones(min_weeks - 1)

    use_snapshots = _snapshots_available("lstm")
    lstm_predictions = build_forecast_features(weekly_prices, weekly_dates, available_symbols, "lstm", use_snapshots)
    for symbol in available_symbols:
        if symbol not in lstm_predictions:
            lstm_predictions[symbol] = np.zeros(min_weeks - 1)

    training_data = sac_build_training_data(weekly_prices, signals, lstm_predictions, available_symbols)
    result = finetune_sac_lstm(training_data, prior_artifacts.actor, prior_artifacts.critic,
                               prior_artifacts.critic_target, prior_artifacts.log_alpha,
                               prior_artifacts.scaler, prior_config, finetune_config)

    prior_metadata = storage.read_metadata(prior_version)
    prior_cagr = prior_metadata["metrics"].get("eval_cagr") if prior_metadata else None
    promoted = prior_cagr is None or result.eval_cagr > prior_cagr

    metadata = create_sac_lstm_metadata(
        version=version, data_window_start=start_date.isoformat(), data_window_end=end_date.isoformat(),
        symbols=available_symbols, config=prior_config, promoted=promoted, prior_version=prior_version,
        actor_loss=result.final_actor_loss, critic_loss=result.final_critic_loss,
        avg_episode_return=result.avg_episode_return, avg_episode_sharpe=result.avg_episode_sharpe,
        eval_sharpe=result.eval_sharpe, eval_cagr=result.eval_cagr, eval_max_drawdown=result.eval_max_drawdown,
    )

    storage.write_artifacts(version, result.actor, result.critic, result.critic_target, result.log_alpha,
                            result.scaler, prior_config, available_symbols, metadata)
    if promoted:
        storage.promote_version(version)

    return SACLSTMTrainResponse(
        version=version, data_window_start=start_date.isoformat(), data_window_end=end_date.isoformat(),
        metrics={"actor_loss": result.final_actor_loss, "critic_loss": result.final_critic_loss,
                 "avg_episode_return": result.avg_episode_return, "avg_episode_sharpe": result.avg_episode_sharpe,
                 "eval_sharpe": result.eval_sharpe, "eval_cagr": result.eval_cagr, "eval_max_drawdown": result.eval_max_drawdown},
        promoted=promoted, prior_version=prior_version, symbols_used=available_symbols,
    )


# ============================================================================
# SAC + PatchTST Training Endpoints
# ============================================================================


class SACPatchTSTTrainResponse(BaseModel):
    """Response model for SAC + PatchTST training endpoint."""

    version: str
    data_window_start: str
    data_window_end: str
    metrics: dict[str, float]
    promoted: bool
    prior_version: str | None
    symbols_used: list[str]
    hf_repo: str | None = None
    hf_url: str | None = None


def get_sac_patchtst_storage() -> SACPatchTSTLocalStorage:
    """Get the SAC + PatchTST storage instance."""
    return SACPatchTSTLocalStorage()


def get_sac_patchtst_config() -> SACPatchTSTConfig:
    """Get SAC + PatchTST configuration."""
    return DEFAULT_SAC_PATCHTST_CONFIG


@router.post("/sac_patchtst/full", response_model=SACPatchTSTTrainResponse)
def train_sac_patchtst_endpoint(
    storage: SACPatchTSTLocalStorage = Depends(get_sac_patchtst_storage),
    symbols: list[str] = Depends(get_top15_symbols),
    config: SACPatchTSTConfig = Depends(get_sac_patchtst_config),
) -> SACPatchTSTTrainResponse:
    """Train SAC portfolio allocator using PatchTST forecasts."""
    import time
    import numpy as np

    start_date, end_date = resolve_training_window()
    logger.info(f"[SAC_PatchTST] Starting training for {len(symbols)} symbols")
    version = sac_patchtst_compute_version(start_date, end_date, symbols, config)

    if storage.version_exists(version):
        existing_metadata = storage.read_metadata(version)
        if existing_metadata:
            return SACPatchTSTTrainResponse(
                version=version, data_window_start=existing_metadata["data_window"]["start"],
                data_window_end=existing_metadata["data_window"]["end"], metrics=existing_metadata["metrics"],
                promoted=existing_metadata["promoted"], prior_version=existing_metadata.get("prior_version"),
                symbols_used=existing_metadata["symbols"],
            )

    prices_dict = load_prices_yfinance(symbols, start_date, end_date)
    available_symbols = [s for s in symbols if s in prices_dict]

    weekly_prices = {}
    for symbol in available_symbols:
        df = prices_dict[symbol]
        if df is not None and len(df) > 0:
            weekly = df["close"].resample("W-FRI").last().dropna()
            weekly_prices[symbol] = weekly.values

    min_weeks = min(len(weekly_prices[s]) for s in available_symbols if s in weekly_prices)
    weekly_df = prices_dict[available_symbols[0]]["close"].resample("W-FRI").last().dropna()
    weekly_dates = weekly_df.index[-min_weeks:]

    for symbol in available_symbols:
        weekly_prices[symbol] = weekly_prices[symbol][-min_weeks:]

    signals = build_rl_training_signals(prices_dict, available_symbols, start_date, end_date)
    for symbol in available_symbols:
        if symbol not in signals:
            signals[symbol] = {k: np.zeros(min_weeks - 1) for k in ["news_sentiment", "gross_margin", "operating_margin", "net_margin", "current_ratio", "debt_to_equity"]}
            signals[symbol]["fundamental_age"] = np.ones(min_weeks - 1)

    use_snapshots = _snapshots_available("patchtst")
    patchtst_predictions = build_forecast_features(weekly_prices, weekly_dates, available_symbols, "patchtst", use_snapshots)
    for symbol in available_symbols:
        if symbol not in patchtst_predictions:
            patchtst_predictions[symbol] = np.zeros(min_weeks - 1)

    training_data = sac_build_training_data(weekly_prices, signals, patchtst_predictions, available_symbols)
    result = train_sac_patchtst(training_data, config)

    prior_version = storage.read_current_version()
    prior_cagr = None
    if prior_version:
        prior_metadata = storage.read_metadata(prior_version)
        if prior_metadata:
            prior_cagr = prior_metadata["metrics"].get("eval_cagr")

    promoted = prior_version is None or prior_cagr is None or result.eval_cagr > prior_cagr

    metadata = create_sac_patchtst_metadata(
        version=version, data_window_start=start_date.isoformat(), data_window_end=end_date.isoformat(),
        symbols=available_symbols, config=config, promoted=promoted, prior_version=prior_version,
        actor_loss=result.final_actor_loss, critic_loss=result.final_critic_loss,
        avg_episode_return=result.avg_episode_return, avg_episode_sharpe=result.avg_episode_sharpe,
        eval_sharpe=result.eval_sharpe, eval_cagr=result.eval_cagr, eval_max_drawdown=result.eval_max_drawdown,
    )

    storage.write_artifacts(version, result.actor, result.critic, result.critic_target, result.log_alpha,
                            result.scaler, config, available_symbols, metadata)
    if promoted:
        storage.promote_version(version)

    return SACPatchTSTTrainResponse(
        version=version, data_window_start=start_date.isoformat(), data_window_end=end_date.isoformat(),
        metrics={"actor_loss": result.final_actor_loss, "critic_loss": result.final_critic_loss,
                 "avg_episode_return": result.avg_episode_return, "avg_episode_sharpe": result.avg_episode_sharpe,
                 "eval_sharpe": result.eval_sharpe, "eval_cagr": result.eval_cagr, "eval_max_drawdown": result.eval_max_drawdown},
        promoted=promoted, prior_version=prior_version, symbols_used=available_symbols,
    )


@router.post("/sac_patchtst/finetune", response_model=SACPatchTSTTrainResponse)
def finetune_sac_patchtst_endpoint(
    storage: SACPatchTSTLocalStorage = Depends(get_sac_patchtst_storage),
    symbols: list[str] = Depends(get_top15_symbols),
) -> SACPatchTSTTrainResponse:
    """Fine-tune SAC + PatchTST on recent data. Requires prior trained model."""
    import time
    from datetime import timedelta
    import numpy as np

    prior_version = storage.read_current_version()
    if prior_version is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="No prior SAC_PatchTST model. Train with POST /train/sac_patchtst/full first")

    prior_artifacts = storage.load_current_artifacts()
    prior_config = prior_artifacts.config

    finetune_config = SACFinetuneConfig()
    from datetime import date
    end_date = date.today()
    start_date = end_date - timedelta(weeks=finetune_config.lookback_weeks + 4)

    version = f"{sac_patchtst_compute_version(start_date, end_date, symbols, prior_config)}-ft"

    if storage.version_exists(version):
        existing_metadata = storage.read_metadata(version)
        if existing_metadata:
            return SACPatchTSTTrainResponse(
                version=version, data_window_start=existing_metadata["data_window"]["start"],
                data_window_end=existing_metadata["data_window"]["end"], metrics=existing_metadata["metrics"],
                promoted=existing_metadata["promoted"], prior_version=existing_metadata.get("prior_version"),
                symbols_used=existing_metadata["symbols"],
            )

    prices_dict = load_prices_yfinance(symbols, start_date, end_date)
    available_symbols = [s for s in symbols if s in prices_dict]

    weekly_prices = {}
    for symbol in available_symbols:
        df = prices_dict[symbol]
        if df is not None and len(df) > 0:
            weekly = df["close"].resample("W-FRI").last().dropna()
            weekly_prices[symbol] = weekly.values

    min_weeks = min(len(weekly_prices[s]) for s in available_symbols if s in weekly_prices)
    weekly_df = prices_dict[available_symbols[0]]["close"].resample("W-FRI").last().dropna()
    weekly_dates = weekly_df.index[-min_weeks:]

    for symbol in available_symbols:
        weekly_prices[symbol] = weekly_prices[symbol][-min_weeks:]

    signals = build_rl_training_signals(prices_dict, available_symbols, start_date, end_date)
    for symbol in available_symbols:
        if symbol not in signals:
            signals[symbol] = {k: np.zeros(min_weeks - 1) for k in ["news_sentiment", "gross_margin", "operating_margin", "net_margin", "current_ratio", "debt_to_equity"]}
            signals[symbol]["fundamental_age"] = np.ones(min_weeks - 1)

    use_snapshots = _snapshots_available("patchtst")
    patchtst_predictions = build_forecast_features(weekly_prices, weekly_dates, available_symbols, "patchtst", use_snapshots)
    for symbol in available_symbols:
        if symbol not in patchtst_predictions:
            patchtst_predictions[symbol] = np.zeros(min_weeks - 1)

    training_data = sac_build_training_data(weekly_prices, signals, patchtst_predictions, available_symbols)
    result = finetune_sac_patchtst(training_data, prior_artifacts.actor, prior_artifacts.critic,
                                    prior_artifacts.critic_target, prior_artifacts.log_alpha,
                                    prior_artifacts.scaler, prior_config, finetune_config)

    prior_metadata = storage.read_metadata(prior_version)
    prior_cagr = prior_metadata["metrics"].get("eval_cagr") if prior_metadata else None
    promoted = prior_cagr is None or result.eval_cagr > prior_cagr

    metadata = create_sac_patchtst_metadata(
        version=version, data_window_start=start_date.isoformat(), data_window_end=end_date.isoformat(),
        symbols=available_symbols, config=prior_config, promoted=promoted, prior_version=prior_version,
        actor_loss=result.final_actor_loss, critic_loss=result.final_critic_loss,
        avg_episode_return=result.avg_episode_return, avg_episode_sharpe=result.avg_episode_sharpe,
        eval_sharpe=result.eval_sharpe, eval_cagr=result.eval_cagr, eval_max_drawdown=result.eval_max_drawdown,
    )

    storage.write_artifacts(version, result.actor, result.critic, result.critic_target, result.log_alpha,
                            result.scaler, prior_config, available_symbols, metadata)
    if promoted:
        storage.promote_version(version)

    return SACPatchTSTTrainResponse(
        version=version, data_window_start=start_date.isoformat(), data_window_end=end_date.isoformat(),
        metrics={"actor_loss": result.final_actor_loss, "critic_loss": result.final_critic_loss,
                 "avg_episode_return": result.avg_episode_return, "avg_episode_sharpe": result.avg_episode_sharpe,
                 "eval_sharpe": result.eval_sharpe, "eval_cagr": result.eval_cagr, "eval_max_drawdown": result.eval_max_drawdown},
        promoted=promoted, prior_version=prior_version, symbols_used=available_symbols,
    )
