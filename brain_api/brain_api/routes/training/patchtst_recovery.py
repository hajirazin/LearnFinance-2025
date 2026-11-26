"""Finish snapshot preparation and publication using saved PatchTST weights."""

import logging
from datetime import date

from brain_api.core.patchtst.training_evidence import validate_label_evidence
from brain_api.storage.forecaster_snapshots import SnapshotLocalStorage
from brain_api.storage.policy import build_common_train_response_kwargs

from .job_registry import complete_job, fail_job, update_progress
from .models import PatchTSTTrainResponse

logger = logging.getLogger(__name__)


def _run_patchtst_snapshots_only(
    *,
    job_id,
    symbols,
    config,
    price_loader,
    bucket,
    train_window,
    version,
    existing_metadata,
    skip_snapshot=False,
    log_prefix="[PatchTST Recovery]",
):
    from . import patchtst

    storage = bucket.local_storage_class()
    metadata = dict(existing_metadata)
    try:
        if config.weekly_regression:
            validate_label_evidence(
                metadata.get("training_evidence", {}),
                config.market,
                date.fromisoformat(metadata["data_window"]["end"]),
                len(symbols),
            )
        if not skip_snapshot:
            update_progress(job_id, {"phase": "snapshots_only_backfill"})
            snapshots = SnapshotLocalStorage(
                bucket.bucket_name,
                base_path=storage.base_path if config.weekly_regression else None,
            )
            first = patchtst._run_patchtst_snapshot_phase(
                train_window=train_window,
                symbols=symbols,
                config=config,
                snapshot_storage=snapshots,
                price_loader=price_loader,
                log_prefix=log_prefix,
            )
            if first:
                metadata["training_evidence"] = {
                    **metadata["training_evidence"],
                    "first_snapshot_cutoff": first.isoformat(),
                }
            metadata["snapshots_pending"] = False

        repo = bucket.hf_repo_getter()
        needs_upload = (
            config.weekly_regression
            and repo
            and (metadata.get("publication_pending") or metadata.get("hf_repo") != repo)
        )
        if needs_upload:
            update_progress(job_id, {"phase": "publishing_saved_checkpoint"})
            artifacts = storage.load_version_artifacts(version)
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
        if config.weekly_regression and storage.version_exists(version):
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
        response = PatchTSTTrainResponse(
            **build_common_train_response_kwargs(version, metadata),
            num_input_channels=config.num_input_channels,
            signals_used=config.feature_names
            if config.weekly_regression
            else ["ohlcv"],
            hf_repo=repo if needs_upload else None,
            hf_url=f"https://huggingface.co/{repo}/tree/{version}"
            if needs_upload
            else None,
        )
        complete_job(job_id, response.model_dump())
    except Exception as exc:
        fail_job(job_id, str(exc))
        logger.exception("%s Recovery failed", log_prefix)
