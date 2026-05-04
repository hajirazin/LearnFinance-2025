"""``POST /train/sac/finetune`` endpoint and background fine-tune task."""

import logging
from datetime import timedelta

import numpy as np
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import JSONResponse

from brain_api.core.config import resolve_cutoff_date
from brain_api.core.lstm import load_prices_yfinance
from brain_api.core.model_buckets import ModelType, get_bucket
from brain_api.core.portfolio_rl.data_loading import build_rl_training_signals
from brain_api.core.portfolio_rl.sac_config import SACFinetuneConfig
from brain_api.core.portfolio_rl.walkforward import build_dual_forecast_features
from brain_api.core.sac import (
    build_training_data as sac_build_training_data,
)
from brain_api.core.sac import (
    compute_version as sac_compute_version,
)
from brain_api.core.sac import finetune_sac
from brain_api.core.sac.promotion import evaluate_sac_finetune_artifact_health
from brain_api.core.training_utils import TrainingCancelledError
from brain_api.storage.policy import (
    StoragePolicyError,
    build_common_train_response_kwargs,
    get_prior_metadata_for_bucket,
    try_load_existing_train_metadata,
)
from brain_api.storage.sac import (
    SACHalalFilteredModelStorage,
    create_sac_metadata,
)

from ..dependencies import get_sac_storage
from ..job_registry import (
    cancel_job,
    complete_job,
    fail_job,
    get_or_create_job,
    update_progress,
)
from ..models import SACTrainResponse, TrainingJobResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/sac/finetune", response_model=SACTrainResponse)
def finetune_sac_endpoint(
    background_tasks: BackgroundTasks,
    storage: SACHalalFilteredModelStorage = Depends(get_sac_storage),
) -> SACTrainResponse | JSONResponse:
    """Fine-tune SAC on recent data.

    Returns 200 with cached result if version already exists (idempotent).
    Returns 202 with job_id if fine-tuning is started in the background.
    Poll GET /train/status/{job_id} for progress and final result.

    Finetune is hard-pinned to the ``sac_halal_filtered`` bucket per
    AGENTS.md known limitation. The storage policy still applies: under
    ``hf_first`` the prior model can come from HF if local is empty.
    """
    finetune_bucket = get_bucket(ModelType.SAC, "halal_filtered")
    try:
        prior_metadata = get_prior_metadata_for_bucket(bucket=finetune_bucket)
    except StoragePolicyError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"[SAC Finetune] hf_first prior fetch failed for bucket "
                f"{finetune_bucket.bucket_name}: {exc}"
            ),
        ) from exc
    if prior_metadata is None:
        raise HTTPException(
            status_code=400,
            detail="No prior SAC model. Train with POST /train/sac/full first",
        )
    prior_version = prior_metadata.get("version")
    if prior_version is None:
        raise HTTPException(
            status_code=400,
            detail="No prior SAC model. Train with POST /train/sac/full first",
        )

    # Pull the prior artifacts via the policy helper so hf_first hosts
    # download from HF + cache locally before we ask SAC for the
    # symbol order / config (which only exist on disk).
    from brain_api.storage.policy import load_current_artifacts_for_bucket

    prior_artifacts_preview = load_current_artifacts_for_bucket(
        bucket=finetune_bucket,
        model_label=finetune_bucket.model_label,
    )
    symbols = list(prior_artifacts_preview.symbol_order)
    prior_config = prior_artifacts_preview.config

    finetune_config = SACFinetuneConfig()
    end_date = resolve_cutoff_date()
    start_date = end_date - timedelta(weeks=finetune_config.lookback_weeks + 4)

    version = f"{sac_compute_version(start_date, end_date, symbols, prior_config)}-ft"

    # HF-aware idempotency skip: under hf_first the helper consults
    # the bucket's HF repo for ``revision=version`` so a wiped local
    # cache does not silently retrain work that already exists on HF.
    # Finetune produces ``-ft``-suffixed versions which are still
    # full HF branches, so the same revision-pointer contract applies.
    existing_metadata = try_load_existing_train_metadata(
        bucket=finetune_bucket, version=version, local_storage=storage
    )
    if existing_metadata:
        return SACTrainResponse(
            **build_common_train_response_kwargs(version, existing_metadata),
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
    storage: SACHalalFilteredModelStorage,
    prior_version: str,
) -> None:
    """Background task that runs SAC fine-tuning.

    Finetune is hard-pinned to the ``sac_halal_filtered`` bucket per
    AGENTS.md known limitation. The prior artifacts come through the
    storage policy helper so ``hf_first`` hosts download from HF +
    cache locally before the rest of the pipeline reads them.
    """
    from brain_api.main import shutdown_event
    from brain_api.storage.policy import load_current_artifacts_for_bucket

    finetune_bucket = get_bucket(ModelType.SAC, "halal_filtered")

    try:
        prior_artifacts = load_current_artifacts_for_bucket(
            bucket=finetune_bucket,
            model_label=finetune_bucket.model_label,
        )
        symbols = list(prior_artifacts.symbol_order)
        logger.info(
            f"[SAC Finetune] Using {len(symbols)} symbols from model {prior_version}"
        )
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

        # Two-write ordering: write artifacts first so the
        # file-existence guardrails inside the health check observe
        # them, then re-write metadata.json with the populated
        # ``promoted`` and ``failure_reasons``.
        provisional_metadata = create_sac_metadata(
            version=version,
            data_window_start=start_date.isoformat(),
            data_window_end=end_date.isoformat(),
            symbols=available_symbols,
            config=prior_config,
            promoted=False,  # placeholder
            prior_version=prior_version,
            actor_loss=result.final_actor_loss,
            critic_loss=result.final_critic_loss,
            avg_episode_return=result.avg_episode_return,
            avg_episode_sharpe=result.avg_episode_sharpe,
            eval_sharpe=result.eval_sharpe,
            eval_cagr=result.eval_cagr,
            eval_max_drawdown=result.eval_max_drawdown,
            bucket_name="sac_halal_filtered",
            failure_reasons=[],  # placeholder
        )

        update_progress(job_id, {"phase": "writing_artifacts"})
        version_dir = storage.write_artifacts(
            version,
            result.actor,
            result.critic,
            result.critic_target,
            result.log_alpha,
            result.scaler,
            prior_config,
            available_symbols,
            provisional_metadata,
        )

        # Finetune symbol-order guardrail: SAC's actor/critic action
        # space is positional, so a finetune that drops a delisted
        # symbol or reorders the slate would silently misalign with
        # the prior model's weights. Reject any drift.
        health = evaluate_sac_finetune_artifact_health(
            actor_loss=result.final_actor_loss,
            critic_loss=result.final_critic_loss,
            eval_cagr=result.eval_cagr,
            eval_sharpe=result.eval_sharpe,
            eval_max_drawdown=result.eval_max_drawdown,
            prior_symbol_order=list(prior_artifacts.symbol_order),
            actual_symbol_order=available_symbols,
            artifact_dir=version_dir,
        )
        promoted = health.is_healthy
        logger.info(
            f"[SAC Finetune] Promotion: {'YES' if promoted else 'NO'} "
            f"(CAGR: {result.eval_cagr * 100:.2f}%)"
            + ("" if promoted else f" (failures: {health.failure_reasons})")
        )

        # Final metadata write with the real promoted + failure_reasons.
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
            # Finetune is halal_filtered-only by design (see AGENTS.md
            # known limitation). The default bucket_name covers it.
            bucket_name="sac_halal_filtered",
            failure_reasons=health.failure_reasons,
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

        hf_repo = None
        hf_url = None
        from brain_api.core.config import get_hf_sac_halal_filtered_model_repo

        hf_model_repo_ft = get_hf_sac_halal_filtered_model_repo()

        # Writes ignore the read policy: upload whenever the bucket
        # has an HF repo configured. Closes audit Bug 6 for finetune.
        if hf_model_repo_ft:
            try:
                from brain_api.storage.sac import SACHuggingFaceModelStorage

                hf_storage = SACHuggingFaceModelStorage(
                    repo_id=hf_model_repo_ft, local_cache=storage
                )
                # make_current = promoted (no cold-start fallback). An
                # unhealthy finetune leaves HF main empty and forces
                # the operator to investigate -- per AGENTS.md rule #1.
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
                    make_current=promoted,
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
            failure_reasons=health.failure_reasons,
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
