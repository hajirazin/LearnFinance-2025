"""SAC training endpoints with dual forecasts (LSTM + PatchTST)."""

import logging
from datetime import timedelta

import numpy as np
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import JSONResponse

from brain_api.core.config import (
    get_hf_sac_model_repo,
    get_storage_backend,
    resolve_cutoff_date,
    resolve_training_window,
)
from brain_api.core.lstm import load_prices_yfinance
from brain_api.core.portfolio_rl.data_loading import build_rl_training_signals
from brain_api.core.portfolio_rl.sac_config import SACFinetuneConfig
from brain_api.core.portfolio_rl.walkforward import (
    build_dual_forecast_features,
)
from brain_api.core.sac import (
    SACConfig,
    finetune_sac,
    train_sac,
)
from brain_api.core.sac import (
    build_training_data as sac_build_training_data,
)
from brain_api.core.sac import (
    compute_version as sac_compute_version,
)
from brain_api.core.training_utils import TrainingCancelledError
from brain_api.storage.sac import SACLocalStorage, create_sac_metadata

from .dependencies import (
    get_rl_training_symbols,
    get_sac_config,
    get_sac_storage,
)
from .helpers import get_prior_version_info
from .job_registry import (
    cancel_job,
    complete_job,
    fail_job,
    get_or_create_job,
    update_progress,
)
from .models import SACTrainResponse, TrainingJobResponse

router = APIRouter()
logger = logging.getLogger(__name__)

MIN_PROMOTION_CAGR = 0.12


@router.post("/sac/full", response_model=SACTrainResponse)
def train_sac_endpoint(
    background_tasks: BackgroundTasks,
    storage: SACLocalStorage = Depends(get_sac_storage),
    symbols: list[str] = Depends(get_rl_training_symbols),
    config: SACConfig = Depends(get_sac_config),
) -> SACTrainResponse | JSONResponse:
    """Train SAC portfolio allocator using dual forecasts.

    Returns 200 with cached result if version already exists (idempotent).
    Returns 202 with job_id if training is started in the background.
    Poll GET /train/status/{job_id} for progress and final result.
    """
    start_date, end_date = resolve_training_window()
    logger.info(f"[SAC] Starting training for {len(symbols)} symbols")
    version = sac_compute_version(start_date, end_date, symbols, config)

    if storage.version_exists(version):
        existing_metadata = storage.read_metadata(version)
        if existing_metadata:
            return SACTrainResponse(
                version=version,
                data_window_start=existing_metadata["data_window"]["start"],
                data_window_end=existing_metadata["data_window"]["end"],
                metrics=existing_metadata["metrics"],
                promoted=existing_metadata["promoted"],
                prior_version=existing_metadata.get("prior_version"),
                symbols_used=existing_metadata["symbols"],
            )

    job, is_new = get_or_create_job("sac", version)
    if not is_new:
        logger.info(f"[SAC] Job {job.job_id} already running, returning 202")
        return JSONResponse(
            status_code=202,
            content=TrainingJobResponse(
                job_id=job.job_id,
                status=job.status,
                message=f"SAC training already in progress for {version}",
            ).model_dump(),
        )

    background_tasks.add_task(
        _run_sac_full_training,
        job_id=job.job_id,
        symbols=symbols,
        config=config,
        storage=storage,
    )
    logger.info(f"[SAC] Background training started: {job.job_id}")

    return JSONResponse(
        status_code=202,
        content=TrainingJobResponse(
            job_id=job.job_id,
            status="pending",
            message=f"SAC training started for {version}",
        ).model_dump(),
    )


def _run_sac_full_training(
    *,
    job_id: str,
    symbols: list[str],
    config: SACConfig,
    storage: SACLocalStorage,
) -> None:
    """Background task that runs the full SAC training pipeline."""
    from brain_api.main import shutdown_event

    try:
        start_date, end_date = resolve_training_window()
        version = sac_compute_version(start_date, end_date, symbols, config)

        update_progress(job_id, {"phase": "loading_prices"})
        prices_dict = load_prices_yfinance(symbols, start_date, end_date)

        if len(prices_dict) == 0:
            raise ValueError("No price data available for training")

        available_symbols = [s for s in symbols if s in prices_dict]
        if len(available_symbols) < 5:
            raise ValueError(
                f"Need at least 5 symbols with data, got {len(available_symbols)}"
            )

        weekly_prices = {}
        for symbol in available_symbols:
            df = prices_dict[symbol]
            if df is not None and len(df) > 0:
                weekly = df["close"].resample("W-FRI").last().dropna()
                weekly_prices[symbol] = weekly.values

        min_weeks = min(
            len(weekly_prices[s]) for s in available_symbols if s in weekly_prices
        )

        first_symbol = available_symbols[0]
        weekly_df = prices_dict[first_symbol]["close"].resample("W-FRI").last().dropna()
        weekly_dates = weekly_df.index[-min_weeks:]

        for symbol in available_symbols:
            if symbol in weekly_prices:
                weekly_prices[symbol] = weekly_prices[symbol][-min_weeks:]

        update_progress(job_id, {"phase": "loading_signals"})
        signals = build_rl_training_signals(
            prices_dict, available_symbols, start_date, end_date
        )

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

        update_progress(job_id, {"phase": "walk_forward_forecasts"})
        lstm_predictions, patchtst_predictions = build_dual_forecast_features(
            weekly_prices=weekly_prices,
            weekly_dates=weekly_dates,
            symbols=available_symbols,
            shutdown_event=shutdown_event,
        )

        for symbol in available_symbols:
            if symbol in lstm_predictions:
                pred_arr = lstm_predictions[symbol]
                if len(pred_arr) >= min_weeks - 1:
                    lstm_predictions[symbol] = pred_arr[-(min_weeks - 1) :]
                else:
                    padded = np.zeros(min_weeks - 1)
                    padded[-len(pred_arr) :] = pred_arr
                    lstm_predictions[symbol] = padded
            else:
                lstm_predictions[symbol] = np.zeros(min_weeks - 1)

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

        update_progress(job_id, {"phase": "training"})
        training_data = sac_build_training_data(
            weekly_prices,
            signals,
            lstm_predictions,
            patchtst_predictions,
            available_symbols,
        )
        result = train_sac(training_data, config, shutdown_event=shutdown_event)

        logger.info(
            f"[SAC] Eval sharpe: {result.eval_sharpe:.4f}, CAGR: {result.eval_cagr * 100:.2f}%"
        )

        update_progress(job_id, {"phase": "promotion_check"})
        from brain_api.storage.sac import SACHuggingFaceModelStorage

        hf_model_repo = get_hf_sac_model_repo()
        prior_info = get_prior_version_info(
            local_storage=storage,
            hf_storage_class=SACHuggingFaceModelStorage,
            hf_model_repo=hf_model_repo,
        )
        prior_version = prior_info.version

        promoted = prior_version is None or result.eval_cagr > MIN_PROMOTION_CAGR
        logger.info(
            f"[SAC] Promotion: {'YES' if promoted else 'NO'} "
            f"(CAGR: {result.eval_cagr * 100:.2f}%, threshold: {MIN_PROMOTION_CAGR * 100:.0f}%)"
        )

        metadata = create_sac_metadata(
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

        update_progress(job_id, {"phase": "writing_artifacts"})
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

        hf_repo = None
        hf_url = None
        storage_backend = get_storage_backend()

        if storage_backend == "hf" and hf_model_repo:
            try:
                hf_storage = SACHuggingFaceModelStorage(repo_id=hf_model_repo)
                hf_has_main = hf_storage.get_current_version() is not None
                should_make_current = promoted or not hf_has_main

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
                logger.info(f"[SAC] Model uploaded to HuggingFace: {hf_url}")
            except Exception as e:
                logger.error(f"[SAC] Failed to upload model to HuggingFace: {e}")

        response = SACTrainResponse(
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
        complete_job(job_id, response.model_dump())
        logger.info(f"[SAC] Job {job_id} completed successfully")

    except TrainingCancelledError:
        cancel_job(job_id)
        logger.info(f"[SAC] Job {job_id} cancelled by shutdown")
    except Exception as e:
        fail_job(job_id, str(e))
        logger.error(f"[SAC] Job {job_id} failed: {e}")


@router.post("/sac/finetune", response_model=SACTrainResponse)
def finetune_sac_endpoint(
    background_tasks: BackgroundTasks,
    storage: SACLocalStorage = Depends(get_sac_storage),
) -> SACTrainResponse | JSONResponse:
    """Fine-tune SAC on recent data.

    Returns 200 with cached result if version already exists (idempotent).
    Returns 202 with job_id if fine-tuning is started in the background.
    Poll GET /train/status/{job_id} for progress and final result.
    """
    prior_version = storage.read_current_version()
    if prior_version is None:
        raise HTTPException(
            status_code=400,
            detail="No prior SAC model. Train with POST /train/sac/full first",
        )

    symbols = storage.load_symbol_order(prior_version)
    prior_config = storage.load_current_artifacts().config

    finetune_config = SACFinetuneConfig()
    end_date = resolve_cutoff_date()
    start_date = end_date - timedelta(weeks=finetune_config.lookback_weeks + 4)

    version = f"{sac_compute_version(start_date, end_date, symbols, prior_config)}-ft"

    if storage.version_exists(version):
        existing_metadata = storage.read_metadata(version)
        if existing_metadata:
            return SACTrainResponse(
                version=version,
                data_window_start=existing_metadata["data_window"]["start"],
                data_window_end=existing_metadata["data_window"]["end"],
                metrics=existing_metadata["metrics"],
                promoted=existing_metadata["promoted"],
                prior_version=existing_metadata.get("prior_version"),
                symbols_used=existing_metadata["symbols"],
            )

    job, is_new = get_or_create_job("sac_finetune", version)
    if not is_new:
        logger.info(f"[SAC Finetune] Job {job.job_id} already running, returning 202")
        return JSONResponse(
            status_code=202,
            content=TrainingJobResponse(
                job_id=job.job_id,
                status=job.status,
                message=f"SAC finetune already in progress for {version}",
            ).model_dump(),
        )

    background_tasks.add_task(
        _run_sac_finetune,
        job_id=job.job_id,
        storage=storage,
        prior_version=prior_version,
    )
    logger.info(f"[SAC Finetune] Background training started: {job.job_id}")

    return JSONResponse(
        status_code=202,
        content=TrainingJobResponse(
            job_id=job.job_id,
            status="pending",
            message=f"SAC finetune started for {version}",
        ).model_dump(),
    )


def _run_sac_finetune(
    *,
    job_id: str,
    storage: SACLocalStorage,
    prior_version: str,
) -> None:
    """Background task that runs SAC fine-tuning."""
    from brain_api.main import shutdown_event

    try:
        symbols = storage.load_symbol_order(prior_version)
        logger.info(
            f"[SAC Finetune] Using {len(symbols)} symbols from model {prior_version}"
        )

        prior_artifacts = storage.load_current_artifacts()
        prior_config = prior_artifacts.config

        finetune_config = SACFinetuneConfig()
        end_date = resolve_cutoff_date()
        start_date = end_date - timedelta(weeks=finetune_config.lookback_weeks + 4)

        version = (
            f"{sac_compute_version(start_date, end_date, symbols, prior_config)}-ft"
        )

        update_progress(job_id, {"phase": "loading_prices"})
        prices_dict = load_prices_yfinance(symbols, start_date, end_date)

        if len(prices_dict) == 0:
            raise ValueError("No price data available for fine-tuning")

        available_symbols = [s for s in symbols if s in prices_dict]
        if len(available_symbols) < 5:
            raise ValueError(
                f"Need at least 5 symbols with data, got {len(available_symbols)}"
            )

        weekly_prices = {}
        for symbol in available_symbols:
            df = prices_dict[symbol]
            if df is not None and len(df) > 0:
                weekly = df["close"].resample("W-FRI").last().dropna()
                weekly_prices[symbol] = weekly.values

        min_weeks = min(
            len(weekly_prices[s]) for s in available_symbols if s in weekly_prices
        )

        weekly_df = (
            prices_dict[available_symbols[0]]["close"].resample("W-FRI").last().dropna()
        )
        weekly_dates = weekly_df.index[-min_weeks:]

        for symbol in available_symbols:
            weekly_prices[symbol] = weekly_prices[symbol][-min_weeks:]

        update_progress(job_id, {"phase": "loading_signals"})
        signals = build_rl_training_signals(
            prices_dict, available_symbols, start_date, end_date
        )

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

        update_progress(job_id, {"phase": "walk_forward_forecasts"})
        lstm_predictions, patchtst_predictions = build_dual_forecast_features(
            weekly_prices=weekly_prices,
            weekly_dates=weekly_dates,
            symbols=available_symbols,
            shutdown_event=shutdown_event,
        )

        for symbol in available_symbols:
            if symbol in lstm_predictions:
                pred_arr = lstm_predictions[symbol]
                if len(pred_arr) >= min_weeks - 1:
                    lstm_predictions[symbol] = pred_arr[-(min_weeks - 1) :]
                else:
                    padded = np.zeros(min_weeks - 1)
                    padded[-len(pred_arr) :] = pred_arr
                    lstm_predictions[symbol] = padded
            else:
                lstm_predictions[symbol] = np.zeros(min_weeks - 1)

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

        update_progress(job_id, {"phase": "training"})
        training_data = sac_build_training_data(
            weekly_prices,
            signals,
            lstm_predictions,
            patchtst_predictions,
            available_symbols,
        )
        result = finetune_sac(
            training_data,
            prior_artifacts.actor,
            prior_artifacts.critic,
            prior_artifacts.critic_target,
            prior_artifacts.log_alpha,
            prior_artifacts.scaler,
            prior_config,
            finetune_config,
            shutdown_event=shutdown_event,
        )

        promoted = result.eval_cagr > MIN_PROMOTION_CAGR
        logger.info(
            f"[SAC Finetune] Promotion: {'YES' if promoted else 'NO'} "
            f"(CAGR: {result.eval_cagr * 100:.2f}%, threshold: {MIN_PROMOTION_CAGR * 100:.0f}%)"
        )

        metadata = create_sac_metadata(
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

        update_progress(job_id, {"phase": "writing_artifacts"})
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

        hf_repo = None
        hf_url = None
        storage_backend = get_storage_backend()
        hf_model_repo_ft = get_hf_sac_model_repo()

        if storage_backend == "hf" and hf_model_repo_ft:
            try:
                from brain_api.storage.sac import SACHuggingFaceModelStorage

                hf_storage = SACHuggingFaceModelStorage(repo_id=hf_model_repo_ft)
                hf_has_main = hf_storage.get_current_version() is not None
                should_make_current = promoted or not hf_has_main

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
                logger.info(f"[SAC Finetune] Model uploaded to HuggingFace: {hf_url}")
            except Exception as e:
                logger.error(
                    f"[SAC Finetune] Failed to upload model to HuggingFace: {e}"
                )

        response = SACTrainResponse(
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
        complete_job(job_id, response.model_dump())
        logger.info(f"[SAC Finetune] Job {job_id} completed successfully")

    except TrainingCancelledError:
        cancel_job(job_id)
        logger.info(f"[SAC Finetune] Job {job_id} cancelled by shutdown")
    except Exception as e:
        fail_job(job_id, str(e))
        logger.error(f"[SAC Finetune] Job {job_id} failed: {e}")
