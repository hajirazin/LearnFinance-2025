"""``POST /train/sac/full`` endpoint and background training task."""

import logging

import numpy as np
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse

from brain_api.core.config import resolve_training_window
from brain_api.core.lstm import load_prices_yfinance
from brain_api.core.model_buckets import (
    BucketConfig,
    ModelType,
    UnknownBucketError,
    get_bucket,
)
from brain_api.core.portfolio_rl.data_loading import build_rl_training_signals
from brain_api.core.portfolio_rl.walkforward import build_dual_forecast_features
from brain_api.core.sac import (
    DEFAULT_SAC_CONFIG,
    SACConfig,
    make_sac_config_for_n_stocks,
    train_sac,
)
from brain_api.core.sac import (
    build_training_data as sac_build_training_data,
)
from brain_api.core.sac import (
    compute_version as sac_compute_version,
)
from brain_api.core.sac.promotion import evaluate_sac_artifact_health
from brain_api.core.training_utils import TrainingCancelledError
from brain_api.storage.policy import (
    StoragePolicyError,
    build_common_train_response_kwargs,
    get_prior_metadata_for_bucket,
    try_load_existing_train_metadata,
)
from brain_api.storage.sac import (
    SACHalalFilteredModelStorage,
    SACHalalModelStorage,
    create_sac_metadata,
)

from ..job_registry import (
    cancel_job,
    complete_job,
    fail_job,
    get_or_create_job,
    update_progress,
)
from ..models import SACTrainResponse, TrainingJobResponse
from ._shared import SACTrainRequest, sac_us_allowed_universes

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/sac/full", response_model=SACTrainResponse)
def train_sac_endpoint(
    background_tasks: BackgroundTasks,
    request: SACTrainRequest = SACTrainRequest(),
) -> SACTrainResponse | JSONResponse:
    """Train SAC portfolio allocator using dual forecasts.

    Resolves the universe-specific config + storage via the bucket
    registry, so two parallel A/B workflows
    (``halal_filtered`` and ``halal``) can hit this endpoint with
    different ``universe`` values and write to independent
    ``current`` pointers without sharing process-wide state.

    Returns 200 with cached result if version already exists (idempotent).
    Returns 202 with job_id if training is started in the background.
    Poll GET /train/status/{job_id} for progress and final result.
    """
    allowed = sac_us_allowed_universes()
    if request.universe not in allowed:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown universe {request.universe!r} for /train/sac/full. "
                f"Valid options: {sorted(allowed)}."
            ),
        )
    try:
        bucket = get_bucket(ModelType.SAC, request.universe)
    except UnknownBucketError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    symbols = bucket.symbols_resolver()
    if bucket.symbol_validator is not None:
        try:
            bucket.symbol_validator(symbols)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    # Build a per-bucket SAC config: ``n_stocks`` and
    # ``target_entropy`` are rewritten to match the resolved slate.
    # For halal_filtered (validator pins to 15) this returns a config
    # byte-equivalent to ``DEFAULT_SAC_CONFIG`` so existing version
    # hashes / current artifacts are unaffected. For halal (variable
    # size) this resizes the SAC actor/critic dim before training.
    config = make_sac_config_for_n_stocks(DEFAULT_SAC_CONFIG, len(symbols))

    storage = bucket.local_storage_class()

    start_date, end_date = resolve_training_window()
    logger.info(
        f"[SAC] Starting training for {len(symbols)} symbols "
        f"(bucket={bucket.bucket_name}, n_stocks={config.n_stocks}, "
        f"target_entropy={config.target_entropy})"
    )
    version = sac_compute_version(start_date, end_date, symbols, config)

    # HF-aware idempotency skip: under hf_first the helper consults
    # the bucket's HF repo for ``revision=version`` so a wiped local
    # cache does not silently retrain work that already exists on HF.
    existing_metadata = try_load_existing_train_metadata(
        bucket=bucket, version=version, local_storage=storage
    )
    if existing_metadata:
        return SACTrainResponse(
            **build_common_train_response_kwargs(version, existing_metadata),
            symbols_used=existing_metadata["symbols"],
        )

    job, is_new = get_or_create_job(f"sac_{request.universe}", version)
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
        bucket=bucket,
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
    storage: SACHalalFilteredModelStorage | SACHalalModelStorage,
    bucket: BucketConfig,
) -> None:
    """Background task that runs the full SAC training pipeline.

    ``bucket`` carries the bucket-specific HF repo getter and storage
    class. ``bucket.bucket_name`` is threaded through to
    ``create_sac_metadata`` so each parallel A/B bucket's metadata
    identifies its own bucket on disk and on HF (vital for telling
    the two ``current`` artifacts apart when they share the SAC
    HuggingFace storage class).
    """
    bucket_name = bucket.bucket_name
    hf_repo_getter = bucket.hf_repo_getter
    hf_storage_class = bucket.hf_storage_class

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
        hf_model_repo = hf_repo_getter()
        # prior_version is kept purely for audit lineage on metadata.
        # The promotion decision below is the new artifact's own
        # guardrails (CAGR floor + finite metrics + symbol-count match
        # + artifact files present); prior metrics are NEVER consulted.
        try:
            prior_metadata = get_prior_metadata_for_bucket(bucket=bucket)
        except StoragePolicyError as exc:
            logger.warning(
                f"[SAC] hf_first prior fetch failed for bucket "
                f"{bucket.bucket_name}: {exc}; treating as inaugural"
            )
            prior_metadata = None
        prior_version: str | None = (
            prior_metadata.get("version") if prior_metadata is not None else None
        )

        # Two-write ordering: write artifacts first so the
        # file-existence guardrails inside the health check can see
        # them, then re-write metadata.json with the populated
        # ``promoted`` and ``failure_reasons``.
        provisional_metadata = create_sac_metadata(
            version=version,
            data_window_start=start_date.isoformat(),
            data_window_end=end_date.isoformat(),
            symbols=available_symbols,
            config=config,
            promoted=False,  # placeholder
            prior_version=prior_version,
            actor_loss=result.final_actor_loss,
            critic_loss=result.final_critic_loss,
            avg_episode_return=result.avg_episode_return,
            avg_episode_sharpe=result.avg_episode_sharpe,
            eval_sharpe=result.eval_sharpe,
            eval_cagr=result.eval_cagr,
            eval_max_drawdown=result.eval_max_drawdown,
            bucket_name=bucket_name,
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
            config,
            available_symbols,
            provisional_metadata,
        )

        # Bucket symbol resolver is the source of truth for the
        # action-space dimension; the resolver was already run at the
        # endpoint entry to drive ``len(symbols)``, so re-running is
        # cheap and avoids a stale cache window between request and
        # health check.
        expected_symbol_count = len(bucket.symbols_resolver())
        health = evaluate_sac_artifact_health(
            actor_loss=result.final_actor_loss,
            critic_loss=result.final_critic_loss,
            eval_cagr=result.eval_cagr,
            eval_sharpe=result.eval_sharpe,
            eval_max_drawdown=result.eval_max_drawdown,
            expected_symbol_count=expected_symbol_count,
            actual_symbol_count=len(available_symbols),
            artifact_dir=version_dir,
        )
        promoted = health.is_healthy
        logger.info(
            f"[SAC] Promotion: {'YES' if promoted else 'NO'} "
            f"(CAGR: {result.eval_cagr * 100:.2f}%)"
            + ("" if promoted else f" (failures: {health.failure_reasons})")
        )

        # Final metadata write with the real promoted + failure_reasons.
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
            bucket_name=bucket_name,
            failure_reasons=health.failure_reasons,
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

        hf_repo = None
        hf_url = None

        # Writes ignore the read policy: upload whenever the bucket
        # has an HF repo configured. Closes audit Bug 6.
        if hf_model_repo:
            try:
                hf_storage = hf_storage_class(
                    repo_id=hf_model_repo, local_cache=storage
                )
                # make_current = promoted (no cold-start fallback). An
                # unhealthy inaugural leaves HF main empty and forces
                # the operator to investigate -- per AGENTS.md rule #1.
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
                    make_current=promoted,
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
            failure_reasons=health.failure_reasons,
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
