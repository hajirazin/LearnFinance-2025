"""DELETE hashed forecaster snapshots for LSTM, US PatchTST, and India PatchTST."""

import logging

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from brain_api.core.model_buckets import (
    ModelType,
    UnknownBucketError,
    get_bucket,
)
from brain_api.routes.training.lstm import _LSTM_US_ALLOWED_UNIVERSES
from brain_api.routes.training.patchtst import _PATCHTST_US_ALLOWED_UNIVERSES
from brain_api.routes.training.patchtst_india import _PATCHTST_INDIA_ALLOWED_UNIVERSES
from brain_api.storage.forecaster_snapshots import SnapshotLocalStorage
from brain_api.storage.forecaster_snapshots.delete_snapshots import (
    DeletedForecasterSnapshot,
    DeleteForecasterSnapshotsResult,
    ForecasterSnapshotDeleteError,
    ForecasterSnapshotDeleteFailure,
    HuggingFaceSnapshotRepoNotConfiguredError,
    SnapshotStorageTarget,
    delete_all_forecaster_snapshots,
)

router = APIRouter()
logger = logging.getLogger(__name__)

_STORAGE_DESCRIPTION = (
    "Where to delete hashed forecaster snapshots: local, hf, or both. Required."
)


class DeletedSnapshotItem(BaseModel):
    """One hashed snapshot removed by this call."""

    cutoff_date: str
    snapshot_digest: str
    deleted_local: bool
    deleted_hf: bool


class DeleteSnapshotsResponse(BaseModel):
    """Result of deleting every hashed snapshot for one forecaster bucket."""

    universe: str
    forecaster_bucket: str
    storage: SnapshotStorageTarget
    deleted: list[DeletedSnapshotItem]


def _deleted_item(item: DeletedForecasterSnapshot) -> DeletedSnapshotItem:
    return DeletedSnapshotItem(
        cutoff_date=item.cutoff_date.isoformat(),
        snapshot_digest=item.snapshot_digest,
        deleted_local=item.deleted_local,
        deleted_hf=item.deleted_hf,
    )


def _failure_payload(failure: ForecasterSnapshotDeleteFailure) -> dict[str, str]:
    return {
        "cutoff_date": failure.cutoff_date.isoformat(),
        "snapshot_digest": failure.snapshot_digest,
        "backend": failure.backend,
        "error": failure.error,
    }


def _delete_bucket_snapshots(
    model_type: ModelType,
    allowed_universes: frozenset[str],
    universe: str,
    storage: SnapshotStorageTarget,
) -> DeleteSnapshotsResponse:
    """Resolve ``universe`` to one bucket and delete its hashed snapshots."""
    if universe not in allowed_universes:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown universe {universe!r} for this snapshot delete route. "
                f"Valid options: {sorted(allowed_universes)}."
            ),
        )
    try:
        bucket = get_bucket(model_type, universe)
    except UnknownBucketError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    snapshot_storage = SnapshotLocalStorage(bucket.bucket_name)
    try:
        result = delete_all_forecaster_snapshots(snapshot_storage, storage)
    except HuggingFaceSnapshotRepoNotConfiguredError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ForecasterSnapshotDeleteError as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Failed to delete one or more forecaster snapshots",
                "deleted": [_deleted_item(item).model_dump() for item in exc.deleted],
                "failures": [_failure_payload(failure) for failure in exc.failures],
            },
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    logger.info(
        "Deleted %s hashed snapshots for %s universe=%s storage=%s",
        len(result.deleted),
        result.forecaster_bucket,
        universe,
        storage.value,
    )
    return _to_response(universe, result)


def _to_response(
    universe: str, result: DeleteForecasterSnapshotsResult
) -> DeleteSnapshotsResponse:
    return DeleteSnapshotsResponse(
        universe=universe,
        forecaster_bucket=result.forecaster_bucket,
        storage=result.storage,
        deleted=[_deleted_item(item) for item in result.deleted],
    )


@router.delete("/lstm/snapshots", response_model=DeleteSnapshotsResponse)
def delete_lstm_snapshots(
    universe: str = Query(
        ...,
        description=(
            "LSTM universe. Required. Must be one of "
            f"{sorted(_LSTM_US_ALLOWED_UNIVERSES)}."
        ),
    ),
    storage: SnapshotStorageTarget = Query(..., description=_STORAGE_DESCRIPTION),
) -> DeleteSnapshotsResponse:
    """Delete every hashed LSTM snapshot for ``universe`` on ``storage``."""
    return _delete_bucket_snapshots(
        ModelType.LSTM,
        _LSTM_US_ALLOWED_UNIVERSES,
        universe,
        storage,
    )


@router.delete("/patchtst/snapshots", response_model=DeleteSnapshotsResponse)
def delete_patchtst_us_snapshots(
    universe: str = Query(
        ...,
        description=(
            "US PatchTST universe. Required. Must be one of "
            f"{sorted(_PATCHTST_US_ALLOWED_UNIVERSES)}."
        ),
    ),
    storage: SnapshotStorageTarget = Query(..., description=_STORAGE_DESCRIPTION),
) -> DeleteSnapshotsResponse:
    """Delete every hashed US PatchTST snapshot for ``universe`` on ``storage``."""
    return _delete_bucket_snapshots(
        ModelType.PATCHTST,
        _PATCHTST_US_ALLOWED_UNIVERSES,
        universe,
        storage,
    )


@router.delete("/patchtst/india/snapshots", response_model=DeleteSnapshotsResponse)
def delete_patchtst_india_snapshots(
    universe: str = Query(
        ...,
        description=(
            "India PatchTST universe. Required. Must be one of "
            f"{sorted(_PATCHTST_INDIA_ALLOWED_UNIVERSES)}."
        ),
    ),
    storage: SnapshotStorageTarget = Query(..., description=_STORAGE_DESCRIPTION),
) -> DeleteSnapshotsResponse:
    """Delete every hashed India PatchTST snapshot for ``universe`` on ``storage``."""
    return _delete_bucket_snapshots(
        ModelType.PATCHTST,
        _PATCHTST_INDIA_ALLOWED_UNIVERSES,
        universe,
        storage,
    )
