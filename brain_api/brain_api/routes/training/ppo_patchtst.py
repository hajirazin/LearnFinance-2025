"""PPO + PatchTST training endpoints."""

import logging
import time
from datetime import timedelta

import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from brain_api.core.config import (
    get_hf_ppo_patchtst_model_repo,
    get_storage_backend,
    resolve_cutoff_date,
    resolve_training_window,
)
from brain_api.core.lstm import load_prices_yfinance
from brain_api.core.portfolio_rl.data_loading import build_rl_training_signals
from brain_api.core.portfolio_rl.walkforward import build_forecast_features
from brain_api.core.ppo_lstm import PPOFinetuneConfig, build_training_data
from brain_api.core.ppo_patchtst import (
    PPOPatchTSTConfig,
    finetune_ppo_patchtst,
    train_ppo_patchtst,
)
from brain_api.core.ppo_patchtst import (
    compute_version as ppo_patchtst_compute_version,
)
from brain_api.storage.local import (
    PPOPatchTSTLocalStorage,
    create_ppo_patchtst_metadata,
)

from .dependencies import (
    get_ppo_patchtst_config,
    get_ppo_patchtst_storage,
    get_top15_symbols,
    snapshots_available,
)
from .helpers import get_prior_version_info
from .models import PPOPatchTSTTrainResponse

router = APIRouter()
logger = logging.getLogger(__name__)


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
    logger.info(
        f"[PPO_PatchTST] Loaded prices for {len(prices_dict)}/{len(symbols)} symbols in {t_prices:.1f}s"
    )

    if len(prices_dict) == 0:
        raise ValueError("No price data available for training")

    # Filter symbols to those with price data
    available_symbols = [s for s in symbols if s in prices_dict]
    if len(available_symbols) < 5:
        raise ValueError(
            f"Need at least 5 symbols with data, got {len(available_symbols)}"
        )

    # Resample prices to weekly (Friday close)
    logger.info("[PPO_PatchTST] Resampling prices to weekly...")
    weekly_prices = {}
    for symbol in available_symbols:
        df = prices_dict[symbol]
        if df is not None and len(df) > 0:
            weekly = df["close"].resample("W-FRI").last().dropna()
            weekly_prices[symbol] = weekly.values

    # Determine minimum length across all symbols
    min_weeks = min(
        len(weekly_prices[s]) for s in available_symbols if s in weekly_prices
    )
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
                    signals[symbol][signal_name] = signal_arr[-min_weeks + 1 :]
                else:
                    padded = np.zeros(min_weeks - 1)
                    padded[-len(signal_arr) :] = (
                        signal_arr[: min_weeks - 1] if len(signal_arr) > 0 else 0
                    )
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
    use_snapshots = snapshots_available("patchtst")
    logger.info(
        f"[PPO_PatchTST] Generating walk-forward forecast features (snapshots={use_snapshots})..."
    )
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
                patchtst_predictions[symbol] = pred_arr[-(min_weeks - 1) :]
            else:
                padded = np.zeros(min_weeks - 1)
                padded[-len(pred_arr) :] = pred_arr
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

    logger.info(
        f"[PPO_PatchTST] Training data: {training_data.n_weeks} weeks, {training_data.n_stocks} stocks"
    )
    logger.info(
        f"[PPO_PatchTST] Signals loaded for {len([s for s in signals if 'news_sentiment' in signals[s]])} symbols"
    )

    # Train PPO
    logger.info("[PPO_PatchTST] Starting PPO training...")
    t0 = time.time()
    result = train_ppo_patchtst(training_data, config)
    t_train = time.time() - t0
    logger.info(f"[PPO_PatchTST] Training complete in {t_train:.1f}s")
    logger.info(
        f"[PPO_PatchTST] Eval sharpe: {result.eval_sharpe:.4f}, CAGR: {result.eval_cagr * 100:.2f}%"
    )

    # Get prior version info (checks local, then HF if needed)
    from brain_api.storage.huggingface import HuggingFaceModelStorage

    hf_model_repo = get_hf_ppo_patchtst_model_repo()
    prior_info = get_prior_version_info(
        local_storage=storage,
        hf_storage_class=HuggingFaceModelStorage,
        hf_model_repo=hf_model_repo,
    )
    prior_version = prior_info.version
    prior_sharpe = (
        prior_info.metadata.get("metrics", {}).get("eval_sharpe")
        if prior_info.metadata
        else None
    )

    if prior_version:
        logger.info(
            f"[PPO_PatchTST] Prior version: {prior_version}, eval_sharpe: {prior_sharpe}"
        )
    else:
        logger.info("[PPO_PatchTST] No prior version exists (first model)")

    # Decide on promotion (first model auto-promotes)
    if prior_version is None:
        promoted = True
        logger.info("[PPO_PatchTST] First model - auto-promoting")
    else:
        promoted = prior_sharpe is None or result.eval_sharpe > prior_sharpe
        logger.info(
            f"[PPO_PatchTST] Metrics: sharpe={result.eval_sharpe:.4f}, cagr={result.eval_cagr:.4f}"
        )
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

    if storage_backend == "hf" and hf_model_repo:
        try:
            hf_storage = HuggingFaceModelStorage(repo_id=hf_model_repo)

            # Check if HF main branch has a version (might be empty even if local has one)
            hf_has_main = hf_storage.get_current_version() is not None

            # Promote to main if: passed promotion check OR HF main is empty (first upload)
            should_make_current = promoted or not hf_has_main
            logger.info(
                f"[PPO_PatchTST] HF upload: promoted={promoted}, hf_has_main={hf_has_main}, "
                f"make_current={should_make_current}"
            )

            hf_info = hf_storage.upload_model(
                version=version,
                model=result.model,
                feature_scaler=result.scaler,
                config=config,
                metadata=metadata,
                make_current=should_make_current,
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
    time.time()
    logger.info("[PPO_PatchTST Finetune] Starting fine-tuning")

    # Load prior model (required for fine-tuning)
    prior_version = storage.read_current_version()
    if prior_version is None:
        raise HTTPException(
            status_code=400,
            detail="No prior PPO_PatchTST model to fine-tune. Train a full model first with POST /train/ppo_patchtst/full",
        )

    logger.info(f"[PPO_PatchTST Finetune] Loading prior model: {prior_version}")
    prior_artifacts = storage.load_current_artifacts()
    prior_config = prior_artifacts.config

    # Use 26-week lookback for fine-tuning
    finetune_config = PPOFinetuneConfig()
    end_date = resolve_cutoff_date()  # Always a Friday
    start_date = end_date - timedelta(
        weeks=finetune_config.lookback_weeks + 4
    )  # Extra buffer for weekends/holidays

    logger.info(f"[PPO_PatchTST Finetune] Data window: {start_date} to {end_date}")
    logger.info(f"[PPO_PatchTST Finetune] Symbols: {symbols}")

    # Compute version for fine-tuned model
    version = ppo_patchtst_compute_version(start_date, end_date, symbols, prior_config)
    version = f"{version}-ft"  # Mark as fine-tuned
    logger.info(f"[PPO_PatchTST Finetune] Version: {version}")

    # Check if already exists (idempotent)
    if storage.version_exists(version):
        logger.info(
            f"[PPO_PatchTST Finetune] Version {version} already exists (idempotent)"
        )
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

    min_weeks = min(
        len(weekly_prices[s]) for s in available_symbols if s in weekly_prices
    )

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
                    signals[symbol][signal_name] = signal_arr[-min_weeks + 1 :]
                else:
                    padded = np.zeros(min_weeks - 1)
                    padded[-len(signal_arr) :] = (
                        signal_arr[: min_weeks - 1] if len(signal_arr) > 0 else 0
                    )
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
    use_snapshots = snapshots_available("patchtst")
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
                patchtst_predictions[symbol] = pred_arr[-(min_weeks - 1) :]
            else:
                padded = np.zeros(min_weeks - 1)
                padded[-len(pred_arr) :] = pred_arr
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
    prior_sharpe = (
        prior_metadata["metrics"].get("eval_sharpe") if prior_metadata else None
    )

    # Decide on promotion (must beat prior)
    promoted = prior_sharpe is None or result.eval_sharpe > prior_sharpe

    logger.info(
        f"[PPO_PatchTST Finetune] Metrics: sharpe={result.eval_sharpe:.4f}, cagr={result.eval_cagr:.4f}"
    )
    logger.info(
        f"[PPO_PatchTST Finetune] Prior sharpe: {prior_sharpe}, New sharpe: {result.eval_sharpe}"
    )
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
    hf_model_repo_ft = get_hf_ppo_patchtst_model_repo()

    if storage_backend == "hf" and hf_model_repo_ft:
        try:
            from brain_api.storage.huggingface import HuggingFaceModelStorage

            hf_storage = HuggingFaceModelStorage(repo_id=hf_model_repo_ft)

            # Check if HF main branch has a version (might be empty even if local has one)
            hf_has_main = hf_storage.get_current_version() is not None

            # Promote to main if: passed promotion check OR HF main is empty (first upload)
            should_make_current = promoted or not hf_has_main
            logger.info(
                f"[PPO_PatchTST Finetune] HF upload: promoted={promoted}, hf_has_main={hf_has_main}, "
                f"make_current={should_make_current}"
            )

            hf_info = hf_storage.upload_model(
                version=version,
                model=result.model,
                feature_scaler=result.scaler,
                config=prior_config,
                metadata=metadata,
                make_current=should_make_current,
            )
            hf_repo = hf_info.repo_id
            hf_url = f"https://huggingface.co/{hf_info.repo_id}/tree/{version}"
            logger.info(
                f"[PPO_PatchTST Finetune] Model uploaded to HuggingFace: {hf_url}"
            )
        except Exception as e:
            logger.error(
                f"[PPO_PatchTST Finetune] Failed to upload model to HuggingFace: {e}"
            )

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
