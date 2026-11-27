"""HTTP background adapter for resuming a saved PatchTST fit."""

import logging

from brain_api.core.training_utils import TrainingCancelledError
from brain_api.services.patchtst_training import recover_patchtst

from .job_registry import cancel_job, complete_job, fail_job, update_progress
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
    from brain_api.main import shutdown_event

    try:
        result = recover_patchtst(
            symbols=symbols,
            config=config,
            price_loader=price_loader,
            bucket=bucket,
            training_window=train_window,
            version=version,
            existing_metadata=existing_metadata,
            skip_snapshot=skip_snapshot,
            log_prefix=log_prefix,
            shutdown_event=shutdown_event,
            on_progress=lambda details: update_progress(job_id, details),
        )
        complete_job(job_id, PatchTSTTrainResponse(**result).model_dump())
        logger.info("%s Recovery completed job_id=%s", log_prefix, job_id)
    except TrainingCancelledError:
        cancel_job(job_id)
        logger.info("%s Recovery cancelled job_id=%s", log_prefix, job_id)
    except Exception as exc:
        fail_job(job_id, str(exc))
        logger.exception("%s Recovery failed job_id=%s", log_prefix, job_id)
