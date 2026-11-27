"""US and India PatchTST training application workflow; no HTTP dependencies."""

import logging
import threading
from datetime import date

from brain_api.core.model_buckets import BucketConfig
from brain_api.core.patchtst.config import PatchTSTConfig
from brain_api.core.patchtst.progress import (
    ProgressCallback,
    check_cancelled,
    report_progress,
)
from brain_api.core.patchtst.training_evidence import can_reuse_weekly_fit
from brain_api.core.patchtst.version import compute_version as patchtst_compute_version
from brain_api.core.patchtst.weekly_data import load_weekly_training_inputs
from brain_api.core.patchtst.weekly_training import fit_weekly_patchtst
from brain_api.core.training_utils import evaluate_forecaster_artifact_health
from brain_api.news.store import NewsStore
from brain_api.news.sync import sync_news
from brain_api.services.patchtst_snapshots import backfill_weekly_patchtst_snapshots
from brain_api.storage.forecaster_snapshots import SnapshotLocalStorage
from brain_api.storage.metadata import create_training_metadata
from brain_api.storage.policy import (
    StoragePolicyError,
    build_common_train_response_kwargs,
    get_prior_metadata_for_bucket,
    try_load_existing_train_metadata,
)

logger = logging.getLogger(__name__)


def train_patchtst(
    *,
    symbols,
    storage,
    bucket: BucketConfig,
    skip_snapshot: bool,
    config: PatchTSTConfig,
    price_loader,
    training_window: tuple[date, date],
    trainer=fit_weekly_patchtst,
    log_prefix="[PatchTST]",
    shutdown_event: threading.Event | None = None,
    on_progress: ProgressCallback | None = None,
) -> dict:
    """Fit, persist, prepare snapshots, publish, and promote one weekly bucket."""
    check_cancelled(shutdown_event)
    if not config.weekly_regression:
        raise ValueError("PatchTST training requires the weekly regression recipe")
    hf_storage_class = bucket.hf_storage_class
    hf_model_repo_getter = bucket.hf_repo_getter
    start_date, end_date = training_window
    logger.info(f"{log_prefix} Starting training for {len(symbols)} symbols")
    logger.info(f"{log_prefix} Data window: {start_date} to {end_date}")
    logger.info(f"{log_prefix} Symbols: {symbols}")
    logger.info(
        f"{log_prefix} Config: {config.num_input_channels} channels, {config.epochs} epochs"
    )
    version = patchtst_compute_version(start_date, end_date, symbols, config)
    logger.info(f"{log_prefix} Computed version: {version}")
    existing_metadata = try_load_existing_train_metadata(
        bucket=bucket, version=version, local_storage=storage
    )
    if existing_metadata and can_reuse_weekly_fit(
        existing_metadata, config, len(symbols)
    ):
        logger.info(
            f"{log_prefix} Version {version} already exists (idempotent), returning cached result"
        )
        return {
            **build_common_train_response_kwargs(version, existing_metadata),
            "num_input_channels": config.num_input_channels,
            "signals_used": config.feature_names,
        }
    evidence = {}
    store = NewsStore(storage.base_path) if config.market == "us" else None
    if store is not None:
        evidence["news_commit"] = sync_news(store)
    report_progress(on_progress, phase="building_weekly_dataset")
    inputs = load_weekly_training_inputs(
        symbols,
        start_date,
        end_date,
        config,
        price_loader,
        store=store,
        on_progress=on_progress,
        shutdown_event=shutdown_event,
    )
    dataset = inputs.labeled(
        start_date, end_date, on_progress=on_progress, shutdown_event=shutdown_event
    )
    del inputs
    report_progress(on_progress, phase="training_weekly_regression")
    result = trainer(
        dataset, config, shutdown_event=shutdown_event, on_progress=on_progress
    )
    evidence.update(result.evidence)
    del dataset
    hf_model_repo = hf_model_repo_getter()
    try:
        prior_metadata = get_prior_metadata_for_bucket(bucket=bucket)
    except StoragePolicyError as exc:
        logger.warning(
            f"{log_prefix} hf_first prior fetch failed for bucket {bucket.bucket_name}: {exc}; treating as inaugural"
        )
        prior_metadata = None
    prior_version: str | None = (
        prior_metadata.get("version") if prior_metadata is not None else None
    )
    provisional_metadata = create_training_metadata(
        model_type=bucket.bucket_name,
        version=version,
        data_window_start=start_date.isoformat(),
        data_window_end=end_date.isoformat(),
        symbols=symbols,
        config_dict=config.to_dict(),
        train_loss=result.train_loss,
        val_loss=result.val_loss,
        baseline_loss=result.baseline_loss,
        best_epoch=result.best_epoch,
        stopped_epoch=result.stopped_epoch,
        promoted=False,
        prior_version=prior_version,
        failure_reasons=[],
        val_rank_ic=result.val_rank_ic,
    )
    logger.info(f"{log_prefix} Writing artifacts for version {version}...")
    version_dir = storage.write_artifacts(
        version=version,
        model=result.model,
        feature_scaler=result.feature_scaler,
        config=config,
        metadata=provisional_metadata,
    )
    logger.info(f"{log_prefix} Artifacts written successfully")
    health = evaluate_forecaster_artifact_health(
        train_loss=result.train_loss,
        val_loss=result.val_loss,
        baseline_loss=result.baseline_loss,
        artifact_dir=version_dir,
    )
    promoted = health.is_healthy
    logger.info(
        f"{log_prefix} Promotion decision: {('PROMOTED' if promoted else 'NOT promoted')}"
        + ("" if promoted else f" (failures: {health.failure_reasons})")
    )
    metadata = create_training_metadata(
        model_type=bucket.bucket_name,
        version=version,
        data_window_start=start_date.isoformat(),
        data_window_end=end_date.isoformat(),
        symbols=symbols,
        config_dict=config.to_dict(),
        train_loss=result.train_loss,
        val_loss=result.val_loss,
        baseline_loss=result.baseline_loss,
        best_epoch=result.best_epoch,
        stopped_epoch=result.stopped_epoch,
        promoted=promoted,
        prior_version=prior_version,
        failure_reasons=health.failure_reasons,
        val_rank_ic=result.val_rank_ic,
    )
    metadata["publication_pending"] = bool(hf_model_repo)
    metadata["snapshots_pending"] = bool(promoted and (not skip_snapshot))
    if evidence:
        metadata["training_evidence"] = evidence
        metadata["metrics"]["weekly_rmse_pp"] = evidence["weekly_rmse_pp"]
    storage.write_artifacts(
        version=version,
        model=result.model,
        feature_scaler=result.feature_scaler,
        config=config,
        metadata=metadata,
    )
    if promoted and (not skip_snapshot):
        report_progress(
            on_progress,
            phase="training_annual_snapshots",
            main_fit={
                "best_epoch": result.best_epoch,
                "stopped_epoch": result.stopped_epoch,
                "weekly_rmse_pp": evidence["weekly_rmse_pp"],
                "latest_label_end": evidence["latest_label_end"],
            },
        )
        snapshot_storage = SnapshotLocalStorage(
            bucket.bucket_name, base_path=storage.base_path
        )

        def snapshot_progress(details):
            report_progress(
                on_progress,
                main_fit={
                    "best_epoch": result.best_epoch,
                    "stopped_epoch": result.stopped_epoch,
                    "weekly_rmse_pp": evidence["weekly_rmse_pp"],
                    "latest_label_end": evidence["latest_label_end"],
                },
                **details,
            )

        supported_first = backfill_weekly_patchtst_snapshots(
            snapshot_storage,
            symbols,
            config,
            start_date,
            end_date,
            price_loader,
            log_prefix=log_prefix,
            shutdown_event=shutdown_event,
            on_progress=snapshot_progress,
            trainer=trainer,
        )
        if supported_first:
            evidence["first_snapshot_cutoff"] = supported_first.isoformat()
            metadata["training_evidence"] = evidence
            metadata["snapshots_pending"] = False
            storage.write_artifacts(
                version, result.model, result.feature_scaler, config, metadata
            )
    hf_repo = None
    hf_url = None
    check_cancelled(shutdown_event)
    if hf_model_repo:
        try:
            hf_storage = hf_storage_class(repo_id=hf_model_repo, local_cache=storage)
            logger.info(
                f"{log_prefix} HF upload: promoted={promoted} -> make_current={promoted}"
            )
            metadata["publication_pending"] = False
            metadata["hf_repo"] = hf_model_repo
            hf_info = hf_storage.upload_model(
                version=version,
                model=result.model,
                feature_scaler=result.feature_scaler,
                config=config,
                metadata=metadata,
                make_current=promoted,
            )
            hf_repo = hf_info.repo_id
            hf_url = f"https://huggingface.co/{hf_info.repo_id}/tree/{version}"
            storage.write_artifacts(
                version, result.model, result.feature_scaler, config, metadata
            )
            logger.info(f"{log_prefix} Model uploaded to HuggingFace: {hf_url}")
        except Exception as e:
            logger.error(f"{log_prefix} Failed to upload model to HuggingFace: {e}")
            metadata["publication_pending"] = True
            storage.write_artifacts(
                version, result.model, result.feature_scaler, config, metadata
            )
            raise
    # Publication may already have moved HF current. Finish local promotion
    # even when shutdown arrives during that commit so both pointers agree.
    if promoted:
        storage.promote_version(version)
    report_progress(
        on_progress,
        phase="completed",
        best_epoch=result.best_epoch,
        stopped_epoch=result.stopped_epoch,
    )
    return {
        "version": version,
        "data_window_start": start_date.isoformat(),
        "data_window_end": end_date.isoformat(),
        "metrics": {
            "train_loss": result.train_loss,
            "val_loss": result.val_loss,
            "baseline_loss": result.baseline_loss,
            "best_epoch": result.best_epoch,
            "stopped_epoch": result.stopped_epoch,
            "val_rank_ic": result.val_rank_ic,
            **({"weekly_rmse_pp": evidence["weekly_rmse_pp"]} if evidence else {}),
        },
        "promoted": promoted,
        "prior_version": prior_version,
        "failure_reasons": health.failure_reasons,
        "hf_repo": hf_repo,
        "hf_url": hf_url,
        "num_input_channels": config.num_input_channels,
        "signals_used": config.feature_names,
    }


def recover_patchtst(
    *,
    symbols,
    config,
    price_loader,
    bucket,
    training_window,
    version,
    existing_metadata,
    skip_snapshot=False,
    shutdown_event=None,
    on_progress=None,
    log_prefix="[PatchTST Recovery]",
) -> dict:
    """Complete snapshots/publication from a saved fit without fitting it again."""
    from brain_api.core.patchtst.training_evidence import validate_label_evidence

    check_cancelled(shutdown_event)
    storage = bucket.local_storage_class()
    metadata = dict(existing_metadata)
    validate_label_evidence(
        metadata.get("training_evidence", {}),
        config.market,
        date.fromisoformat(metadata["data_window"]["end"]),
        len(symbols),
    )
    if not skip_snapshot:
        first = backfill_weekly_patchtst_snapshots(
            SnapshotLocalStorage(bucket.bucket_name, base_path=storage.base_path),
            symbols,
            config,
            *training_window,
            price_loader,
            log_prefix=log_prefix,
            shutdown_event=shutdown_event,
            on_progress=on_progress,
        )
        metadata["training_evidence"] = {
            **metadata["training_evidence"],
            "first_snapshot_cutoff": first.isoformat(),
        }
        metadata["snapshots_pending"] = False
    check_cancelled(shutdown_event)
    repo = bucket.hf_repo_getter()
    needs_upload = bool(
        repo
        and (metadata.get("publication_pending") or metadata.get("hf_repo") != repo)
    )
    if needs_upload:
        artifacts = storage.load_version_artifacts(version)
        report_progress(on_progress, phase="publishing_saved_checkpoint")
        metadata.update(publication_pending=False, hf_repo=repo)
        try:
            bucket.hf_storage_class(repo_id=repo, local_cache=storage).upload_model(
                version=version,
                model=artifacts.model,
                feature_scaler=artifacts.feature_scaler,
                config=artifacts.config,
                metadata=metadata,
                make_current=metadata["promoted"],
            )
        except Exception:
            metadata["publication_pending"] = True
            storage.write_artifacts(
                version,
                artifacts.model,
                artifacts.feature_scaler,
                artifacts.config,
                metadata,
            )
            raise
    # A successful remote commit must be finalized locally before stopping.
    if storage.version_exists(version):
        artifacts = storage.load_version_artifacts(version)
        storage.write_artifacts(
            version,
            artifacts.model,
            artifacts.feature_scaler,
            artifacts.config,
            metadata,
        )
        if metadata["promoted"] and not metadata.get("snapshots_pending"):
            storage.promote_version(version)
    report_progress(on_progress, phase="completed")
    return {
        **build_common_train_response_kwargs(version, metadata),
        "num_input_channels": config.num_input_channels,
        "signals_used": config.feature_names,
        "hf_repo": repo if needs_upload else None,
        "hf_url": f"https://huggingface.co/{repo}/tree/{version}"
        if needs_upload
        else None,
    }
