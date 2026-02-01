"""PatchTST training endpoint."""

import logging
import time
from datetime import date

import pandas as pd
from fastapi import APIRouter, Depends, Query

from brain_api.core.config import (
    get_hf_patchtst_model_repo,
    get_storage_backend,
    resolve_training_window,
)
from brain_api.core.data_freshness import ensure_fresh_training_data
from brain_api.core.patchtst import (
    PatchTSTConfig,
    align_multivariate_data,
    load_historical_fundamentals,
    load_historical_news_sentiment,
)
from brain_api.core.patchtst import (
    build_dataset as patchtst_build_dataset,
)
from brain_api.core.patchtst import (
    compute_version as patchtst_compute_version,
)
from brain_api.core.patchtst import (
    evaluate_for_promotion as patchtst_evaluate_for_promotion,
)
from brain_api.core.patchtst import (
    load_prices_yfinance as patchtst_load_prices,
)
from brain_api.core.patchtst import (
    train_model_pytorch as patchtst_train_model,
)
from brain_api.storage.forecaster_snapshots import (
    SnapshotLocalStorage,
    create_snapshot_metadata,
)
from brain_api.storage.local import PatchTSTModelStorage, create_patchtst_metadata

from .dependencies import (
    PatchTSTDataAligner,
    PatchTSTDatasetBuilder,
    PatchTSTFundamentalsLoader,
    PatchTSTNewsLoader,
    PatchTSTPriceLoader,
    PatchTSTTrainer,
    get_patchtst_config,
    get_patchtst_data_aligner,
    get_patchtst_dataset_builder,
    get_patchtst_fundamentals_loader,
    get_patchtst_news_loader,
    get_patchtst_price_loader,
    get_patchtst_storage,
    get_patchtst_trainer,
    get_symbols,
)
from .helpers import get_prior_version_info
from .models import PatchTSTTrainResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/patchtst", response_model=PatchTSTTrainResponse)
def train_patchtst(
    skip_snapshot: bool = Query(
        False,
        description="Skip saving snapshot (by default saves snapshot for current + all historical years)",
    ),
    storage: PatchTSTModelStorage = Depends(get_patchtst_storage),
    symbols: list[str] = Depends(get_symbols),
    config: PatchTSTConfig = Depends(get_patchtst_config),
    price_loader: PatchTSTPriceLoader = Depends(get_patchtst_price_loader),
    news_loader: PatchTSTNewsLoader = Depends(get_patchtst_news_loader),
    fundamentals_loader: PatchTSTFundamentalsLoader = Depends(
        get_patchtst_fundamentals_loader
    ),
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
    # Resolve window from API config
    start_date, end_date = resolve_training_window()
    logger.info(f"[PatchTST] Starting training for {len(symbols)} symbols")
    logger.info(f"[PatchTST] Data window: {start_date} to {end_date}")
    logger.info(f"[PatchTST] Symbols: {symbols}")
    logger.info(
        f"[PatchTST] Config: {config.num_input_channels} channels, {config.epochs} epochs"
    )

    # Compute deterministic version
    version = patchtst_compute_version(start_date, end_date, symbols, config)
    logger.info(f"[PatchTST] Computed version: {version}")

    # Check if this version already exists (idempotent)
    if storage.version_exists(version):
        logger.info(
            f"[PatchTST] Version {version} already exists (idempotent), returning cached result"
        )
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

    # Ensure training data is fresh
    try:
        freshness_result = ensure_fresh_training_data(symbols, start_date, end_date)
        logger.info(
            f"[PatchTST] Data freshness: {freshness_result.sentiment_gaps_filled} sentiment gaps filled, "
            f"{len(freshness_result.fundamentals_refreshed)} fundamentals refreshed"
        )
        if freshness_result.fundamentals_failed:
            logger.warning(
                f"[PatchTST] Failed to refresh fundamentals: {freshness_result.fundamentals_failed}"
            )
    except Exception as e:
        logger.warning(f"[PatchTST] Data freshness check failed (continuing): {e}")

    # Load price data
    logger.info(f"[PatchTST] Loading price data for {len(symbols)} symbols...")
    t0 = time.time()
    prices = price_loader(symbols, start_date, end_date)
    t_prices = time.time() - t0
    logger.info(
        f"[PatchTST] Loaded prices for {len(prices)}/{len(symbols)} symbols in {t_prices:.1f}s"
    )

    if len(prices) == 0:
        logger.error("[PatchTST] No price data loaded - cannot train model")
        raise ValueError("No price data available for training")

    # Load news sentiment
    logger.info("[PatchTST] Loading historical news sentiment...")
    t0 = time.time()
    news_sentiment = news_loader(symbols, start_date, end_date)
    t_news = time.time() - t0
    logger.info(
        f"[PatchTST] Loaded news sentiment for {len(news_sentiment)}/{len(symbols)} symbols in {t_news:.1f}s"
    )

    # Load fundamentals
    logger.info("[PatchTST] Loading historical fundamentals...")
    t0 = time.time()
    fundamentals = fundamentals_loader(symbols, start_date, end_date)
    t_fund = time.time() - t0
    logger.info(
        f"[PatchTST] Loaded fundamentals for {len(fundamentals)}/{len(symbols)} symbols in {t_fund:.1f}s"
    )

    # Align all data into multi-channel features
    logger.info("[PatchTST] Aligning multivariate data...")
    t0 = time.time()
    aligned_features = data_aligner(prices, news_sentiment, fundamentals, config)
    t_align = time.time() - t0
    logger.info(
        f"[PatchTST] Aligned data for {len(aligned_features)}/{len(prices)} symbols in {t_align:.1f}s"
    )

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
    logger.info(
        f"[PatchTST] Dataset built in {t_dataset:.1f}s: {len(dataset.X)} samples"
    )

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
    logger.info(
        f"[PatchTST] Metrics: train_loss={result.train_loss:.6f}, val_loss={result.val_loss:.6f}, baseline={result.baseline_loss:.6f}"
    )

    # Get prior version info for promotion decision (checks local, then HF if needed)
    from brain_api.storage.huggingface import PatchTSTHuggingFaceModelStorage

    hf_model_repo = get_hf_patchtst_model_repo()
    prior_info = get_prior_version_info(
        local_storage=storage,
        hf_storage_class=PatchTSTHuggingFaceModelStorage,
        hf_model_repo=hf_model_repo,
    )
    prior_version = prior_info.version
    prior_val_loss = prior_info.val_loss

    if prior_version:
        logger.info(
            f"[PatchTST] Prior version: {prior_version}, val_loss: {prior_val_loss}"
        )
    else:
        logger.info("[PatchTST] No prior version exists (first model)")

    # Decide on promotion
    promoted = patchtst_evaluate_for_promotion(
        val_loss=result.val_loss,
        prior_val_loss=prior_val_loss,
    )
    logger.info(
        f"[PatchTST] Promotion decision: {'PROMOTED' if promoted else 'NOT promoted'}"
    )

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

    if storage_backend == "hf" and hf_model_repo:
        try:
            hf_storage = PatchTSTHuggingFaceModelStorage(repo_id=hf_model_repo)

            # Check if HF main branch has a version (might be empty even if local has one)
            hf_has_main = hf_storage.get_current_version() is not None

            # Promote to main if: passed promotion check OR HF main is empty (first upload)
            should_make_current = promoted or not hf_has_main
            logger.info(
                f"[PatchTST] HF upload: promoted={promoted}, hf_has_main={hf_has_main}, "
                f"make_current={should_make_current}"
            )

            hf_info = hf_storage.upload_model(
                version=version,
                model=result.model,
                feature_scaler=result.feature_scaler,
                config=config,
                metadata=metadata,
                make_current=should_make_current,
            )
            hf_repo = hf_info.repo_id
            hf_url = f"https://huggingface.co/{hf_info.repo_id}/tree/{version}"
            logger.info(f"[PatchTST] Model uploaded to HuggingFace: {hf_url}")
        except Exception as e:
            logger.error(f"[PatchTST] Failed to upload model to HuggingFace: {e}")
            # Don't fail the training request if HF upload fails

    # Save snapshots (unless skip_snapshot=True)
    if not skip_snapshot:
        snapshot_storage = SnapshotLocalStorage("patchtst")
        check_hf = storage_backend == "hf"

        # Save snapshot for current training window
        if not snapshot_storage.snapshot_exists_anywhere(end_date, check_hf=check_hf):
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
                    logger.info(
                        f"[PatchTST] Uploaded snapshot {end_date} to HuggingFace"
                    )
                except Exception as e:
                    logger.error(f"[PatchTST] Failed to upload snapshot to HF: {e}")

        # Also backfill all historical snapshots
        logger.info("[PatchTST] Backfilling historical snapshots...")
        _backfill_patchtst_snapshots(
            symbols, config, start_date, end_date, snapshot_storage, storage_backend
        )

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


def _filter_prices_by_cutoff(
    prices: dict[str, pd.DataFrame],
    cutoff_date: date,
) -> dict[str, pd.DataFrame]:
    """Filter price DataFrames to include only data up to cutoff_date.

    Args:
        prices: Dict mapping symbol -> DataFrame with DatetimeIndex
        cutoff_date: Include data up to and including this date

    Returns:
        Filtered dict with same structure, excluding symbols with no data after filtering
    """
    cutoff_ts = pd.Timestamp(cutoff_date)
    return {
        symbol: df[df.index <= cutoff_ts].copy()
        for symbol, df in prices.items()
        if len(df[df.index <= cutoff_ts]) > 0
    }


def _filter_signals_by_cutoff(
    signals: dict[str, pd.DataFrame],
    cutoff_date: date,
) -> dict[str, pd.DataFrame]:
    """Filter signal DataFrames to include only data up to cutoff_date.

    Works with both DatetimeIndex and regular index (will try to convert).

    Args:
        signals: Dict mapping symbol -> DataFrame
        cutoff_date: Include data up to and including this date

    Returns:
        Filtered dict with same structure, excluding symbols with no data after filtering
    """
    cutoff_ts = pd.Timestamp(cutoff_date)
    result = {}

    for symbol, df in signals.items():
        if df.empty:
            continue

        # Handle both DatetimeIndex and regular index
        if isinstance(df.index, pd.DatetimeIndex):
            filtered = df[df.index <= cutoff_ts]
        else:
            # Try to convert index to datetime for comparison
            try:
                idx = pd.to_datetime(df.index)
                mask = idx <= cutoff_ts
                filtered = df[mask]
            except (ValueError, TypeError):
                # If conversion fails, include all data
                filtered = df

        if len(filtered) > 0:
            result[symbol] = filtered.copy()

    return result


def _backfill_patchtst_snapshots(
    symbols: list[str],
    config: PatchTSTConfig,
    start_date: date,
    end_date: date,
    snapshot_storage: SnapshotLocalStorage,
    storage_backend: str = "local",
) -> None:
    """Backfill PatchTST snapshots for all historical years.

    For each year from (start_year + 4) to end_year-1, trains a snapshot
    on data up to Dec 31 of that year. Uses 4-year bootstrap period.

    Optimization: Loads all data (prices, news sentiment, fundamentals) ONCE
    for the full window and filters incrementally by year instead of
    re-downloading for each snapshot.

    Args:
        symbols: List of stock symbols
        config: PatchTST configuration
        start_date: Training data start date
        end_date: Training data end date
        snapshot_storage: Storage instance
        storage_backend: "local" or "hf" - if "hf", uploads to HuggingFace
    """
    start_year = start_date.year
    end_year = end_date.year
    bootstrap_years = 4
    check_hf = storage_backend == "hf"

    # Check if any snapshots need to be created (check HF too if in HF mode)
    snapshots_needed = []
    for year in range(start_year + bootstrap_years, end_year):
        cutoff_date = date(year, 12, 31)
        if not snapshot_storage.snapshot_exists_anywhere(
            cutoff_date, check_hf=check_hf
        ):
            snapshots_needed.append(cutoff_date)

    if not snapshots_needed:
        logger.info("[PatchTST Backfill] All snapshots already exist, nothing to do")
        return

    logger.info(
        f"[PatchTST Backfill] Need to create {len(snapshots_needed)} snapshots: {snapshots_needed}"
    )

    # Load ALL data ONCE for full window
    logger.info(
        "[PatchTST Backfill] Loading prices for full window (single download)..."
    )
    t0 = time.time()
    prices_full = patchtst_load_prices(symbols, start_date, end_date)
    t_prices = time.time() - t0
    logger.info(
        f"[PatchTST Backfill] Loaded prices for {len(prices_full)} symbols in {t_prices:.1f}s"
    )

    if len(prices_full) == 0:
        logger.warning(
            "[PatchTST Backfill] No price data loaded, cannot create snapshots"
        )
        return

    logger.info(
        "[PatchTST Backfill] Loading news sentiment for full window (single download)..."
    )
    t0 = time.time()
    news_full = load_historical_news_sentiment(symbols, start_date, end_date)
    t_news = time.time() - t0
    logger.info(
        f"[PatchTST Backfill] Loaded news sentiment for {len(news_full)} symbols in {t_news:.1f}s"
    )

    logger.info(
        "[PatchTST Backfill] Loading fundamentals for full window (single download)..."
    )
    t0 = time.time()
    fundamentals_full = load_historical_fundamentals(symbols, start_date, end_date)
    t_fund = time.time() - t0
    logger.info(
        f"[PatchTST Backfill] Loaded fundamentals for {len(fundamentals_full)} symbols in {t_fund:.1f}s"
    )

    # Train each snapshot using filtered data
    for cutoff_date in snapshots_needed:
        logger.info(f"[PatchTST Backfill] Training snapshot for cutoff {cutoff_date}")
        t0 = time.time()

        # Filter all data sources to cutoff (no re-download!)
        prices = _filter_prices_by_cutoff(prices_full, cutoff_date)
        if len(prices) == 0:
            logger.warning(
                f"[PatchTST Backfill] No price data for cutoff {cutoff_date}, skipping"
            )
            continue

        news_sentiment = _filter_signals_by_cutoff(news_full, cutoff_date)
        fundamentals = _filter_signals_by_cutoff(fundamentals_full, cutoff_date)

        aligned_features = align_multivariate_data(
            prices, news_sentiment, fundamentals, config
        )

        if len(aligned_features) == 0:
            logger.warning(
                f"[PatchTST Backfill] No aligned features for cutoff {cutoff_date}, skipping"
            )
            continue

        dataset = patchtst_build_dataset(aligned_features, prices, config)
        if len(dataset.X) == 0:
            logger.warning(
                f"[PatchTST Backfill] Empty dataset for cutoff {cutoff_date}, skipping"
            )
            continue

        result = patchtst_train_model(
            dataset.X, dataset.y, dataset.feature_scaler, config
        )

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
        logger.info(
            f"[PatchTST Backfill] Saved snapshot for {cutoff_date} in {time.time() - t0:.1f}s"
        )

        # Upload to HuggingFace if in HF mode
        if storage_backend == "hf":
            try:
                snapshot_storage.upload_snapshot_to_hf(cutoff_date)
                logger.info(
                    f"[PatchTST Backfill] Uploaded snapshot {cutoff_date} to HuggingFace"
                )
            except Exception as e:
                logger.error(
                    f"[PatchTST Backfill] Failed to upload snapshot to HF: {e}"
                )
