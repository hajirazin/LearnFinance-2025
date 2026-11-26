"""The active US weekly forecaster defines the supported SAC training window."""

from datetime import date, timedelta

from brain_api.core.config import resolve_training_window
from brain_api.core.patchtst.config import (
    US_WEEKLY_CONFIG,
    PatchTSTConfig,
    patchtst_recipe,
)
from brain_api.storage.policy import StoragePolicyError, get_prior_metadata_for_bucket

SAC_PATCHTST_RECIPE = patchtst_recipe(US_WEEKLY_CONFIG)


def resolve_sac_training_window() -> tuple[date, date]:
    from brain_api.core.model_buckets import ModelType, get_bucket

    start, end = resolve_training_window()
    bucket = get_bucket(ModelType.PATCHTST, "halal_new")
    metadata = get_prior_metadata_for_bucket(bucket=bucket) or {}
    raw = metadata.get("config")
    if not raw or patchtst_recipe(PatchTSTConfig(**raw)) != SAC_PATCHTST_RECIPE:
        raise StoragePolicyError("SAC requires the active US weekly PatchTST regressor")
    earliest = metadata.get("training_evidence", {}).get("first_snapshot_cutoff")
    if not earliest:
        raise StoragePolicyError(
            "US PatchTST has no published weekly snapshot inventory"
        )
    return max(start, date.fromisoformat(earliest) + timedelta(days=1)), end
