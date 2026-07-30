"""``POST /train/sac/full`` endpoint and background training task."""

import logging
from dataclasses import replace

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
    SAC_EXPERIMENT_SEEDS,
    SACConfig,
    SACTrainingExperiment,
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
from brain_api.core.sac.trade_clock import (
    build_sac_weekly_trade_clock,
    extract_session_open_prices,
)
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
from .preflight import assess_sac_training_readiness

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

    # Symbol-equality short-circuit: if the bucket's current promoted
    # model was trained on the exact same symbol set, skip retraining
    # and return the current model's metadata. Sits before the
    # version-equality short-circuit (lines below) because it is the
    # broader gate -- it fires even when the data window has moved
    # forward, as long as the slate is unchanged. Per AGENTS.md rule
    # #1, an HF outage under hf_first surfaces as 503 rather than a
    # silent retrain so the operator notices HF is down.
    try:
        prior_metadata = get_prior_metadata_for_bucket(bucket=bucket)
    except StoragePolicyError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"SAC full: prior metadata fetch failed for bucket "
                f"{bucket.bucket_name!r}: {exc}. Refusing to retrain "
                f"silently; check HF reachability or STORAGE_BACKEND."
            ),
        ) from exc

    if (
        not request.force
        and prior_metadata is not None
        and set(prior_metadata.get("symbols", [])) == set(symbols)
    ):
        logger.info(
            f"[SAC] Symbol-equality short-circuit: bucket={bucket.bucket_name} "
            f"current version={prior_metadata['version']} symbols match resolved "
            f"slate ({len(symbols)} stocks); returning current metadata "
            f"(set force=True to bypass)."
        )
        return SACTrainResponse(
            **build_common_train_response_kwargs(
                prior_metadata["version"], prior_metadata
            ),
            symbols_used=prior_metadata["symbols"],
        )

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
    if not request.force and existing_metadata:
        return SACTrainResponse(
            **build_common_train_response_kwargs(version, existing_metadata),
            symbols_used=existing_metadata["symbols"],
        )

    # The endpoint's idempotency gates above intentionally remain cheap. Once
    # a real training run is required, enforce the same strict input contract
    # exposed by /sac/preflight before creating a durable background job.
    # ``force=True`` here means "perform the complete assessment"; it does not
    # alter the caller's retraining choice, which was already handled above.
    readiness = assess_sac_training_readiness(request.universe, force=True)
    if not readiness.ready:
        raise HTTPException(
            status_code=503 if readiness.errors else 409,
            detail={
                "message": "SAC training inputs are not ready",
                "universe": readiness.universe,
                "symbols": list(readiness.symbols),
                "missing": [issue.to_dict() for issue in readiness.missing],
                "errors": [issue.to_dict() for issue in readiness.errors],
            },
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
        missing_price_symbols = sorted(set(symbols) - set(available_symbols))
        if missing_price_symbols:
            raise ValueError(
                "SAC training requires returns for the exact halal slate; "
                f"missing price histories: {missing_price_symbols}"
            )

        trade_clock = build_sac_weekly_trade_clock(start_date, end_date)
        weekly_prices = {}
        for symbol in available_symbols:
            df = prices_dict[symbol]
            if df is not None and len(df) > 0:
                weekly_prices[symbol] = extract_session_open_prices(
                    df,
                    trade_clock.rebalance_sessions,
                    symbol=symbol,
                )

        missing_weekly_symbols = sorted(set(available_symbols) - set(weekly_prices))
        if missing_weekly_symbols:
            raise ValueError(
                "Missing weekly XNYS open-price series for SAC symbols: "
                f"{missing_weekly_symbols}"
            )
        min_weeks = len(trade_clock.rebalance_sessions)
        weekly_dates = trade_clock.transition_actor_cutoffs

        update_progress(job_id, {"phase": "loading_signals"})
        signals = build_rl_training_signals(
            prices_dict,
            available_symbols,
            start_date,
            end_date,
            weekly_cutoffs=trade_clock.transition_actor_cutoffs,
        )

        for symbol in available_symbols:
            if symbol not in signals:
                raise ValueError(f"Missing SAC training signals for {symbol}")
            for signal_name, signal_arr in signals[symbol].items():
                if len(signal_arr) < min_weeks - 1:
                    raise ValueError(
                        f"SAC training signal {signal_name} for {symbol} has "
                        f"{len(signal_arr)} weeks; need {min_weeks - 1}"
                    )
                signals[symbol][signal_name] = signal_arr[-(min_weeks - 1) :]

        update_progress(job_id, {"phase": "walk_forward_forecasts"})
        lstm_predictions, patchtst_predictions = build_dual_forecast_features(
            weekly_prices=weekly_prices,
            weekly_dates=weekly_dates,
            symbols=available_symbols,
            shutdown_event=shutdown_event,
            target_dates=trade_clock.transition_start_sessions,
        )

        for symbol in available_symbols:
            for forecast_name, predictions in (
                ("LSTM", lstm_predictions),
                ("PatchTST", patchtst_predictions),
            ):
                if symbol not in predictions:
                    raise ValueError(
                        f"Missing {forecast_name} training forecasts for {symbol}"
                    )
                pred_arr = predictions[symbol]
                if len(pred_arr) < min_weeks - 1:
                    raise ValueError(
                        f"{forecast_name} forecasts for {symbol} have "
                        f"{len(pred_arr)} weeks; need {min_weeks - 1}"
                    )
                predictions[symbol] = pred_arr[-(min_weeks - 1) :]

        update_progress(job_id, {"phase": "training"})
        training_data = sac_build_training_data(
            weekly_prices,
            signals,
            lstm_predictions,
            patchtst_predictions,
            available_symbols,
        )
        experiment = SACTrainingExperiment.run(
            config=config,
            train_candidate=lambda seed_config: train_sac(
                training_data,
                seed_config,
                shutdown_event=shutdown_event,
            ),
            cagr_of=lambda candidate_result: candidate_result.eval_cagr,
        )
        selected_candidate = experiment.selected
        result = selected_candidate.result
        selected_config = replace(config, seed=selected_candidate.seed)

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

        update_progress(job_id, {"phase": "writing_candidates"})
        candidate_paths = {}
        for candidate in experiment.candidates:
            candidate_result = candidate.result
            candidate_config = replace(config, seed=candidate.seed)
            candidate_metadata = create_sac_metadata(
                version=version,
                data_window_start=start_date.isoformat(),
                data_window_end=end_date.isoformat(),
                symbols=available_symbols,
                config=candidate_config,
                promoted=False,
                prior_version=prior_version,
                actor_loss=candidate_result.final_actor_loss,
                critic_loss=candidate_result.final_critic_loss,
                avg_episode_return=candidate_result.avg_episode_return,
                avg_episode_sharpe=candidate_result.avg_episode_sharpe,
                eval_sharpe=candidate_result.eval_sharpe,
                eval_cagr=candidate_result.eval_cagr,
                eval_max_drawdown=candidate_result.eval_max_drawdown,
                bucket_name=bucket_name,
                failure_reasons=[],
                state_schema_version=2,
                training_seed=candidate.seed,
                experiment_seeds=list(SAC_EXPERIMENT_SEEDS),
            )
            candidate_paths[candidate.seed] = storage.write_candidate_artifacts(
                version,
                candidate.seed,
                candidate_result.actor,
                candidate_result.critic,
                candidate_result.critic_target,
                candidate_result.log_alpha,
                candidate_result.scaler,
                candidate_config,
                available_symbols,
                candidate_metadata,
            )
        selected_candidate_dir = candidate_paths[selected_candidate.seed]

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
            artifact_dir=selected_candidate_dir,
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
            config=selected_config,
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
            state_schema_version=2,
            training_seed=selected_candidate.seed,
            experiment_seeds=list(SAC_EXPERIMENT_SEEDS),
        )
        if promoted:
            storage.promote_candidate(version, selected_candidate.seed)
            storage.write_artifacts(
                version,
                result.actor,
                result.critic,
                result.critic_target,
                result.log_alpha,
                result.scaler,
                selected_config,
                available_symbols,
                metadata,
            )
            storage.promote_version(version)
        else:
            storage.write_candidate_metadata(
                version,
                selected_candidate.seed,
                metadata,
            )

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
                    config=selected_config,
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
