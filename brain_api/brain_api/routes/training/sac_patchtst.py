"""SAC + PatchTST training endpoints."""

import logging
from datetime import timedelta

import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from brain_api.core.config import (
    get_hf_sac_patchtst_model_repo,
    get_storage_backend,
    resolve_cutoff_date,
    resolve_training_window,
)
from brain_api.core.lstm import load_prices_yfinance
from brain_api.core.portfolio_rl.data_loading import build_rl_training_signals
from brain_api.core.portfolio_rl.sac_config import SACFinetuneConfig
from brain_api.core.portfolio_rl.walkforward import build_forecast_features
from brain_api.core.sac_lstm import build_training_data as sac_build_training_data
from brain_api.core.sac_patchtst import (
    SACPatchTSTConfig,
    finetune_sac_patchtst,
    train_sac_patchtst,
)
from brain_api.core.sac_patchtst import (
    compute_version as sac_patchtst_compute_version,
)
from brain_api.storage.local import (
    SACPatchTSTLocalStorage,
    create_sac_patchtst_metadata,
)

from .dependencies import (
    get_sac_patchtst_config,
    get_sac_patchtst_storage,
    get_top15_symbols,
    snapshots_available,
)
from .helpers import get_prior_version_info
from .models import SACPatchTSTTrainResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/sac_patchtst/full", response_model=SACPatchTSTTrainResponse)
def train_sac_patchtst_endpoint(
    storage: SACPatchTSTLocalStorage = Depends(get_sac_patchtst_storage),
    symbols: list[str] = Depends(get_top15_symbols),
    config: SACPatchTSTConfig = Depends(get_sac_patchtst_config),
) -> SACPatchTSTTrainResponse:
    """Train SAC portfolio allocator using PatchTST forecasts."""
    start_date, end_date = resolve_training_window()
    logger.info(f"[SAC_PatchTST] Starting training for {len(symbols)} symbols")
    version = sac_patchtst_compute_version(start_date, end_date, symbols, config)

    if storage.version_exists(version):
        existing_metadata = storage.read_metadata(version)
        if existing_metadata:
            return SACPatchTSTTrainResponse(
                version=version,
                data_window_start=existing_metadata["data_window"]["start"],
                data_window_end=existing_metadata["data_window"]["end"],
                metrics=existing_metadata["metrics"],
                promoted=existing_metadata["promoted"],
                prior_version=existing_metadata.get("prior_version"),
                symbols_used=existing_metadata["symbols"],
            )

    # Load price data
    prices_dict = load_prices_yfinance(symbols, start_date, end_date)

    if len(prices_dict) == 0:
        raise ValueError("No price data available for training")

    # Filter symbols to those with price data
    available_symbols = [s for s in symbols if s in prices_dict]
    if len(available_symbols) < 5:
        raise ValueError(
            f"Need at least 5 symbols with data, got {len(available_symbols)}"
        )

    # Resample prices to weekly (Friday close)
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

    # Get weekly date index for walk-forward forecasts
    weekly_df = (
        prices_dict[available_symbols[0]]["close"].resample("W-FRI").last().dropna()
    )
    weekly_dates = weekly_df.index[-min_weeks:]

    # Align all price series
    for symbol in available_symbols:
        weekly_prices[symbol] = weekly_prices[symbol][-min_weeks:]

    # Load historical signals (news sentiment, fundamentals)
    signals = build_rl_training_signals(
        prices_dict, available_symbols, start_date, end_date
    )

    # Align signals to the common week count
    for symbol in available_symbols:
        if symbol in signals:
            for signal_name in signals[symbol]:
                signal_arr = signals[symbol][signal_name]
                if len(signal_arr) >= min_weeks:
                    signals[symbol][signal_name] = signal_arr[-min_weeks + 1 :]
                else:
                    # Pad with zeros if not enough data
                    padded = np.zeros(min_weeks - 1)
                    padded[-len(signal_arr) :] = (
                        signal_arr[: min_weeks - 1] if len(signal_arr) > 0 else 0
                    )
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
    use_snapshots = snapshots_available("patchtst")
    patchtst_predictions = build_forecast_features(
        weekly_prices, weekly_dates, available_symbols, "patchtst", use_snapshots
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

    # Build training data
    training_data = sac_build_training_data(
        weekly_prices, signals, patchtst_predictions, available_symbols
    )
    result = train_sac_patchtst(training_data, config)

    # Get prior version info (checks local, then HF if needed)
    from brain_api.storage.huggingface import SACPatchTSTHuggingFaceModelStorage

    hf_model_repo = get_hf_sac_patchtst_model_repo()
    prior_info = get_prior_version_info(
        local_storage=storage,
        hf_storage_class=SACPatchTSTHuggingFaceModelStorage,
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
            f"[SAC_PatchTST] Prior version: {prior_version}, eval_sharpe: {prior_sharpe}"
        )
    else:
        logger.info("[SAC_PatchTST] No prior version exists (first model)")

    promoted = (
        prior_version is None
        or prior_sharpe is None
        or result.eval_sharpe > prior_sharpe
    )
    logger.info(
        f"[SAC_PatchTST] Metrics: sharpe={result.eval_sharpe:.4f}, cagr={result.eval_cagr:.4f}"
    )
    logger.info(f"[SAC_PatchTST] Promotion: {'YES' if promoted else 'NO'}")

    metadata = create_sac_patchtst_metadata(
        version=version,
        data_window_start=start_date.isoformat(),
        data_window_end=end_date.isoformat(),
        symbols=available_symbols,
        config=config,
        promoted=promoted,
        prior_version=prior_version,
        actor_loss=result.final_actor_loss,
        critic_loss=result.final_critic_loss,
        avg_episode_return=result.avg_episode_return,
        avg_episode_sharpe=result.avg_episode_sharpe,
        eval_sharpe=result.eval_sharpe,
        eval_cagr=result.eval_cagr,
        eval_max_drawdown=result.eval_max_drawdown,
    )

    storage.write_artifacts(
        version,
        result.actor,
        result.critic,
        result.critic_target,
        result.log_alpha,
        result.scaler,
        config,
        available_symbols,
        metadata,
    )
    if promoted:
        storage.promote_version(version)

    # Optionally push to HuggingFace Hub
    hf_repo = None
    hf_url = None
    storage_backend = get_storage_backend()

    if storage_backend == "hf" and hf_model_repo:
        try:
            hf_storage = SACPatchTSTHuggingFaceModelStorage(repo_id=hf_model_repo)

            # Check if HF main branch has a version (might be empty even if local has one)
            hf_has_main = hf_storage.get_current_version() is not None

            # Promote to main if: passed promotion check OR HF main is empty (first upload)
            should_make_current = promoted or not hf_has_main
            logger.info(
                f"[SAC_PatchTST] HF upload: promoted={promoted}, hf_has_main={hf_has_main}, "
                f"make_current={should_make_current}"
            )

            hf_info = hf_storage.upload_model(
                version=version,
                actor=result.actor,
                critic=result.critic,
                critic_target=result.critic_target,
                log_alpha=result.log_alpha,
                scaler=result.scaler,
                config=config,
                symbol_order=available_symbols,
                metadata=metadata,
                make_current=should_make_current,
            )
            hf_repo = hf_info.repo_id
            hf_url = f"https://huggingface.co/{hf_info.repo_id}/tree/{version}"
            logger.info(f"[SAC_PatchTST] Model uploaded to HuggingFace: {hf_url}")
        except Exception as e:
            logger.error(f"[SAC_PatchTST] Failed to upload model to HuggingFace: {e}")
            # Don't fail the training request if HF upload fails

    return SACPatchTSTTrainResponse(
        version=version,
        data_window_start=start_date.isoformat(),
        data_window_end=end_date.isoformat(),
        metrics={
            "actor_loss": result.final_actor_loss,
            "critic_loss": result.final_critic_loss,
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


@router.post("/sac_patchtst/finetune", response_model=SACPatchTSTTrainResponse)
def finetune_sac_patchtst_endpoint(
    storage: SACPatchTSTLocalStorage = Depends(get_sac_patchtst_storage),
    symbols: list[str] = Depends(get_top15_symbols),
) -> SACPatchTSTTrainResponse:
    """Fine-tune SAC + PatchTST on recent data. Requires prior trained model."""
    prior_version = storage.read_current_version()
    if prior_version is None:
        raise HTTPException(
            status_code=400,
            detail="No prior SAC_PatchTST model. Train with POST /train/sac_patchtst/full first",
        )

    prior_artifacts = storage.load_current_artifacts()
    prior_config = prior_artifacts.config

    finetune_config = SACFinetuneConfig()
    end_date = resolve_cutoff_date()  # Always a Friday
    start_date = end_date - timedelta(
        weeks=finetune_config.lookback_weeks + 4
    )  # Extra buffer for weekends/holidays

    version = f"{sac_patchtst_compute_version(start_date, end_date, symbols, prior_config)}-ft"

    if storage.version_exists(version):
        existing_metadata = storage.read_metadata(version)
        if existing_metadata:
            return SACPatchTSTTrainResponse(
                version=version,
                data_window_start=existing_metadata["data_window"]["start"],
                data_window_end=existing_metadata["data_window"]["end"],
                metrics=existing_metadata["metrics"],
                promoted=existing_metadata["promoted"],
                prior_version=existing_metadata.get("prior_version"),
                symbols_used=existing_metadata["symbols"],
            )

    # Load price data
    prices_dict = load_prices_yfinance(symbols, start_date, end_date)

    if len(prices_dict) == 0:
        raise ValueError("No price data available for fine-tuning")

    # Filter symbols to those with price data
    available_symbols = [s for s in symbols if s in prices_dict]
    if len(available_symbols) < 5:
        raise ValueError(
            f"Need at least 5 symbols with data, got {len(available_symbols)}"
        )

    # Resample prices to weekly (Friday close)
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

    # Get weekly date index for walk-forward forecasts
    weekly_df = (
        prices_dict[available_symbols[0]]["close"].resample("W-FRI").last().dropna()
    )
    weekly_dates = weekly_df.index[-min_weeks:]

    # Align all price series
    for symbol in available_symbols:
        weekly_prices[symbol] = weekly_prices[symbol][-min_weeks:]

    # Load historical signals (news sentiment, fundamentals)
    signals = build_rl_training_signals(
        prices_dict, available_symbols, start_date, end_date
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
        weekly_prices, weekly_dates, available_symbols, "patchtst", use_snapshots
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

    # Build training data
    training_data = sac_build_training_data(
        weekly_prices, signals, patchtst_predictions, available_symbols
    )
    result = finetune_sac_patchtst(
        training_data,
        prior_artifacts.actor,
        prior_artifacts.critic,
        prior_artifacts.critic_target,
        prior_artifacts.log_alpha,
        prior_artifacts.scaler,
        prior_config,
        finetune_config,
    )

    prior_metadata = storage.read_metadata(prior_version)
    prior_sharpe = (
        prior_metadata["metrics"].get("eval_sharpe") if prior_metadata else None
    )

    promoted = prior_sharpe is None or result.eval_sharpe > prior_sharpe

    logger.info(
        f"[SAC_PatchTST Finetune] Metrics: sharpe={result.eval_sharpe:.4f}, cagr={result.eval_cagr:.4f}"
    )
    logger.info(
        f"[SAC_PatchTST Finetune] Prior sharpe: {prior_sharpe}, New sharpe: {result.eval_sharpe}"
    )
    logger.info(f"[SAC_PatchTST Finetune] Promotion: {'YES' if promoted else 'NO'}")

    metadata = create_sac_patchtst_metadata(
        version=version,
        data_window_start=start_date.isoformat(),
        data_window_end=end_date.isoformat(),
        symbols=available_symbols,
        config=prior_config,
        promoted=promoted,
        prior_version=prior_version,
        actor_loss=result.final_actor_loss,
        critic_loss=result.final_critic_loss,
        avg_episode_return=result.avg_episode_return,
        avg_episode_sharpe=result.avg_episode_sharpe,
        eval_sharpe=result.eval_sharpe,
        eval_cagr=result.eval_cagr,
        eval_max_drawdown=result.eval_max_drawdown,
    )

    storage.write_artifacts(
        version,
        result.actor,
        result.critic,
        result.critic_target,
        result.log_alpha,
        result.scaler,
        prior_config,
        available_symbols,
        metadata,
    )
    if promoted:
        storage.promote_version(version)

    # Optionally push to HuggingFace Hub
    hf_repo = None
    hf_url = None
    storage_backend = get_storage_backend()
    hf_model_repo_ft = get_hf_sac_patchtst_model_repo()

    if storage_backend == "hf" and hf_model_repo_ft:
        try:
            from brain_api.storage.huggingface import SACPatchTSTHuggingFaceModelStorage

            hf_storage = SACPatchTSTHuggingFaceModelStorage(repo_id=hf_model_repo_ft)

            # Check if HF main branch has a version (might be empty even if local has one)
            hf_has_main = hf_storage.get_current_version() is not None

            # Promote to main if: passed promotion check OR HF main is empty (first upload)
            should_make_current = promoted or not hf_has_main
            logger.info(
                f"[SAC_PatchTST Finetune] HF upload: promoted={promoted}, hf_has_main={hf_has_main}, "
                f"make_current={should_make_current}"
            )

            hf_info = hf_storage.upload_model(
                version=version,
                actor=result.actor,
                critic=result.critic,
                critic_target=result.critic_target,
                log_alpha=result.log_alpha,
                scaler=result.scaler,
                config=prior_config,
                symbol_order=available_symbols,
                metadata=metadata,
                make_current=should_make_current,
            )
            hf_repo = hf_info.repo_id
            hf_url = f"https://huggingface.co/{hf_info.repo_id}/tree/{version}"
            logger.info(
                f"[SAC_PatchTST Finetune] Model uploaded to HuggingFace: {hf_url}"
            )
        except Exception as e:
            logger.error(
                f"[SAC_PatchTST Finetune] Failed to upload model to HuggingFace: {e}"
            )
            # Don't fail the training request if HF upload fails

    return SACPatchTSTTrainResponse(
        version=version,
        data_window_start=start_date.isoformat(),
        data_window_end=end_date.isoformat(),
        metrics={
            "actor_loss": result.final_actor_loss,
            "critic_loss": result.final_critic_loss,
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
