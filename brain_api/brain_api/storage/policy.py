"""Storage policy for model artifacts: ``local_first`` vs ``hf_first``.

This module is the single source of truth for *where reads come from*
across every ``(model, universe)`` bucket (LSTM x halal_new, PatchTST
x halal_new, PatchTST x nifty_shariah_500, SAC x halal_filtered, SAC
x halal, plus the forecaster snapshot subsystem).

Writes are NOT policy-gated. Training routes always write to local
storage and unconditionally upload to HuggingFace when the bucket's
HF repo env is configured. The policy only decides where reads come
from:

* ``local_first`` (default): try local; on miss, fetch from HF,
  persist locally + atomically promote, return.
* ``hf_first``: cheap ``metadata.json`` fetch from HF ``main``; if
  local already has that exact version, short-circuit to local;
  otherwise download from HF, persist + promote, return.

Cold-start (HF ``main`` missing) is explicit:

* Inference (``load_current_artifacts_for_bucket``): raises
  :class:`fastapi.HTTPException` 503 with an actionable message.
* Training prior-version (``get_prior_metadata_for_bucket``): returns
  ``None`` so the caller treats the new training run as the inaugural
  promotion.

Per AGENTS.md "AI assistant behavioral rules" #1, every failure
surfaces -- no silent fallback masks a real bug.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date
from enum import Enum
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import (
    EntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

if TYPE_CHECKING:
    from brain_api.core.model_buckets import BucketConfig
    from brain_api.storage.forecaster_snapshots.local import SnapshotLocalStorage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Policy enum + resolver
# ---------------------------------------------------------------------------


ENV_STORAGE_BACKEND = "STORAGE_BACKEND"


class StoragePolicy(str, Enum):
    """Read-priority policy for model artifacts and snapshots.

    Members are intentionally string-valued so the env var maps 1:1
    onto the enum (``STORAGE_BACKEND=hf_first`` -> ``HF_FIRST``).
    """

    LOCAL_FIRST = "local_first"
    HF_FIRST = "hf_first"


class StoragePolicyError(RuntimeError):
    """Raised when the storage policy cannot satisfy a non-inference
    request (training prior-version reads, snapshot ensure).

    Inference call sites prefer ``HTTPException`` (503) directly so the
    response body is shaped for the API client. Training call sites
    catch this and decide whether to abort the run or fall through to
    inaugural-promote semantics.
    """


def get_storage_policy() -> StoragePolicy:
    """Resolve the storage policy from the ``STORAGE_BACKEND`` env var.

    Returns:
        ``StoragePolicy.LOCAL_FIRST`` when the env var is unset or
        empty; the corresponding enum member when it is one of the
        valid values.

    Raises:
        ValueError: if the env var is set to anything other than the
            two valid enum values. Legacy values (``local`` / ``hf``)
            also raise -- the migration is intentional and must be
            done by the operator before boot succeeds.
    """
    raw = os.environ.get(ENV_STORAGE_BACKEND, "").strip()
    if not raw:
        return StoragePolicy.LOCAL_FIRST
    try:
        return StoragePolicy(raw)
    except ValueError as e:
        valid = sorted(p.value for p in StoragePolicy)
        raise ValueError(
            f"Invalid {ENV_STORAGE_BACKEND}={raw!r}. "
            f"Valid values: {valid} (or empty for default {StoragePolicy.LOCAL_FIRST.value!r}). "
            f"Legacy values 'local'/'hf' are no longer supported -- "
            f"migrate to 'local_first' / 'hf_first'."
        ) from e


# ---------------------------------------------------------------------------
# Internal helpers (HF storage instantiation + metadata fetching)
# ---------------------------------------------------------------------------


def _instantiate_hf_storage(
    bucket: BucketConfig, local_storage: Any
) -> tuple[Any | None, str | None]:
    """Build the bucket's HF storage instance with explicit local_cache.

    Returns ``(hf_storage, hf_repo)`` when the bucket's HF repo env is
    set; ``(None, None)`` otherwise. Always passes
    ``local_cache=local_storage`` so SAC bucket isolation (Bug 4 in the
    audit) cannot regress -- the HF subclass writes downloaded
    artifacts into the matching local bucket directory.
    """
    hf_repo = bucket.hf_repo_getter()
    if not hf_repo:
        return None, None
    hf_storage = bucket.hf_storage_class(
        repo_id=hf_repo,
        local_cache=local_storage,
    )
    return hf_storage, hf_repo


def _fetch_hf_main_metadata(
    *, repo_id: str, token: str | None, model_label: str
) -> dict[str, Any] | None:
    """Fetch ``metadata.json`` from the HF repo's ``main`` revision.

    Returns:
        Parsed metadata dict if HF ``main`` exists; ``None`` if the
        repo, revision, or file is missing (cold-start).

    Raises:
        StoragePolicyError: when HF is unreachable for any other
            reason (network, auth, rate limit). This is the loud
            failure path so a misconfigured token or DNS doesn't
            silently look like a cold-start.
    """
    try:
        metadata_path = hf_hub_download(
            repo_id=repo_id,
            filename="metadata.json",
            repo_type="model",
            token=token,
        )
    except (RepositoryNotFoundError, RevisionNotFoundError, EntryNotFoundError):
        return None
    except Exception as exc:
        raise StoragePolicyError(
            f"HF unreachable for {model_label} ({repo_id}): {exc}"
        ) from exc
    with open(metadata_path) as f:
        return json.load(f)


def hf_versions_match(
    local_version: str | None, hf_metadata: dict[str, Any] | None
) -> bool:
    """Cheap equality used by the ``hf_first`` short-circuit.

    Returns ``True`` only when both sides are present and the version
    strings match exactly. Any ``None`` on either side returns ``False``
    so the caller proceeds with a fresh HF download.
    """
    if local_version is None or hf_metadata is None:
        return False
    return local_version == hf_metadata.get("version")


# ---------------------------------------------------------------------------
# Inference: load current artifacts (the helper every model route calls)
# ---------------------------------------------------------------------------


def load_current_artifacts_for_bucket(
    *,
    bucket: BucketConfig,
    model_label: str,
    policy: StoragePolicy | None = None,
) -> Any:
    """Load the current artifacts for a bucket per the active policy.

    This is the centralized read helper; LSTM, PatchTST (US + India),
    PatchTST score-batch, and SAC inference all route through it. It
    instantiates the bucket's local + HF storage classes with the
    matching ``local_cache`` so two SAC buckets (``halal_filtered`` /
    ``halal``) cannot cache into each other's directory.

    Args:
        bucket: ``BucketConfig`` for the ``(model, universe)`` pair.
        model_label: Human-readable label used in error messages
            (e.g. ``"LSTM halal_new"``, ``"SAC halal"``).
        policy: Override; when ``None``, reads ``STORAGE_BACKEND``.

    Returns:
        The model-specific artifacts object (``LSTMArtifacts``,
        ``PatchTSTArtifacts``, or ``SACArtifacts``) ready for
        inference.

    Raises:
        HTTPException 503: if the artifacts cannot be loaded under the
            active policy. The detail string explains which side
            failed (local empty + no HF repo, HF unreachable, HF
            ``main`` missing, etc.).
    """
    if policy is None:
        policy = get_storage_policy()
    local_storage = bucket.local_storage_class()

    if policy is StoragePolicy.LOCAL_FIRST:
        return _load_local_first(local_storage, bucket, model_label)
    return _load_hf_first(local_storage, bucket, model_label)


def _load_local_first(
    local_storage: Any, bucket: BucketConfig, model_label: str
) -> Any:
    """Try local; on miss, fall back to HF."""
    try:
        return local_storage.load_current_artifacts()
    except (ValueError, FileNotFoundError) as local_err:
        local_msg = str(local_err)
        logger.info(
            f"[storage-policy:local_first] {model_label}: local empty ({local_msg}); "
            f"trying HuggingFace."
        )

    hf_storage, hf_repo = _instantiate_hf_storage(bucket, local_storage)
    if hf_storage is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"No {model_label} model available: local is empty and no "
                f"HF repo is configured for bucket {bucket.bucket_name!r}. "
                f"Train the model first or set the bucket's HF repo env var."
            ),
        )
    try:
        return hf_storage.download_model(use_cache=True)
    except Exception as hf_err:
        raise HTTPException(
            status_code=503,
            detail=(
                f"No {model_label} model available: local empty and HF "
                f"({hf_repo}) failed: {hf_err}"
            ),
        ) from None


def _load_hf_first(local_storage: Any, bucket: BucketConfig, model_label: str) -> Any:
    """Check HF main metadata; reuse local only if it matches that exact version."""
    hf_storage, hf_repo = _instantiate_hf_storage(bucket, local_storage)
    if hf_storage is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"hf_first policy requires an HF repo for {model_label} "
                f"(bucket {bucket.bucket_name!r}). Set the bucket's HF "
                f"repo env var or switch STORAGE_BACKEND to local_first."
            ),
        )

    try:
        hf_metadata = _fetch_hf_main_metadata(
            repo_id=hf_repo,
            token=hf_storage.token,
            model_label=model_label,
        )
    except StoragePolicyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None

    if hf_metadata is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"hf_first: HF main is missing for {model_label} "
                f"(bucket {bucket.bucket_name!r}, repo {hf_repo}). "
                f"Cold-start: train locally first so the upload populates "
                f"HF main, or switch STORAGE_BACKEND to local_first."
            ),
        )

    hf_version = hf_metadata.get("version")
    local_version = local_storage.read_current_version()
    if hf_versions_match(local_version, hf_metadata):
        logger.info(
            f"[storage-policy:hf_first] {model_label}: local matches HF "
            f"main version {hf_version}; using local."
        )
        return local_storage.load_current_artifacts()

    logger.info(
        f"[storage-policy:hf_first] {model_label}: local={local_version!r} "
        f"!= HF={hf_version!r}; downloading from HF."
    )
    try:
        return hf_storage.download_model(version=hf_version, use_cache=True)
    except Exception as hf_err:
        raise HTTPException(
            status_code=503,
            detail=(
                f"hf_first: download failed for {model_label} version "
                f"{hf_version!r} from {hf_repo}: {hf_err}"
            ),
        ) from None


# ---------------------------------------------------------------------------
# Training: prior version metadata (for promotion comparison)
# ---------------------------------------------------------------------------


def get_prior_metadata_for_bucket(
    *,
    bucket: BucketConfig,
    policy: StoragePolicy | None = None,
) -> dict[str, Any] | None:
    """Resolve the prior-promoted metadata dict for a training run.

    Promotion logic compares the new model's val_loss / CAGR against
    the prior version's. The prior comes from local first under
    ``local_first`` and from HF first under ``hf_first``.

    Cold-start (no prior anywhere) returns ``None`` so the caller
    promotes the new version as the inaugural one.

    Args:
        bucket: ``BucketConfig`` for the ``(model, universe)`` pair.
        policy: Override; when ``None``, reads ``STORAGE_BACKEND``.

    Returns:
        Metadata dict from local or HF, or ``None`` for cold-start.

    Raises:
        StoragePolicyError: when ``hf_first`` is active but HF is
            unreachable. The training pipeline catches this and
            decides whether to abort or fall back to local. Inference
            call sites should not use this helper -- they want
            ``load_current_artifacts_for_bucket`` instead.
    """
    if policy is None:
        policy = get_storage_policy()
    local_storage = bucket.local_storage_class()

    def _local_metadata() -> dict[str, Any] | None:
        version = local_storage.read_current_version()
        if version is None:
            return None
        return local_storage.read_metadata(version)

    if policy is StoragePolicy.LOCAL_FIRST:
        meta = _local_metadata()
        if meta is not None:
            return meta
        hf_storage, hf_repo = _instantiate_hf_storage(bucket, local_storage)
        if hf_storage is None:
            return None
        try:
            return _fetch_hf_main_metadata(
                repo_id=hf_repo,
                token=hf_storage.token,
                model_label=bucket.bucket_name,
            )
        except StoragePolicyError as exc:
            logger.warning(
                f"[storage-policy:local_first] {bucket.bucket_name}: "
                f"HF metadata fetch failed (treating as no prior): {exc}"
            )
            return None

    # HF_FIRST
    hf_storage, hf_repo = _instantiate_hf_storage(bucket, local_storage)
    if hf_storage is None:
        raise StoragePolicyError(
            f"hf_first policy requires HF repo for bucket "
            f"{bucket.bucket_name!r}; got none. Set the HF env var or "
            f"switch STORAGE_BACKEND to local_first."
        )
    hf_metadata = _fetch_hf_main_metadata(
        repo_id=hf_repo,
        token=hf_storage.token,
        model_label=bucket.bucket_name,
    )
    if hf_metadata is not None:
        return hf_metadata
    # HF main missing under hf_first: inaugural promotion.
    logger.info(
        f"[storage-policy:hf_first] {bucket.bucket_name}: HF main missing; "
        f"treating new version as inaugural promotion."
    )
    return None


# ---------------------------------------------------------------------------
# Snapshots: ensure-available helper (used by SAC walk-forward training)
# ---------------------------------------------------------------------------


def ensure_snapshot_for_bucket(
    *,
    snapshot_storage: SnapshotLocalStorage,
    cutoff_date: date,
    policy: StoragePolicy | None = None,
) -> bool:
    """Ensure a forecaster snapshot is available locally for inference.

    Snapshots are content-addressed by ``cutoff_date`` (year-end) and
    do not have version drift in the way main-branch artifacts do, so
    both policies behave the same on the happy path: prefer local,
    fall back to HF download. The policy selector exists for two
    reasons:

    1. Consistency: every storage read in the codebase flows through
       this module so behavior is auditable from one place.
    2. Configuration check: ``hf_first`` requires the bucket's HF repo
       env to be set; we surface that as a loud failure so an
       ephemeral host doesn't silently work in local-only mode.

    Args:
        snapshot_storage: ``SnapshotLocalStorage`` for the forecaster
            bucket (``lstm_halal_new``, ``patchtst_halal_new``,
            ``patchtst_nifty_shariah_500``).
        cutoff_date: Year-end cutoff for the snapshot to ensure.
        policy: Override; when ``None``, reads ``STORAGE_BACKEND``.

    Returns:
        ``True`` if the snapshot is now available locally (after
        potential download), ``False`` otherwise.

    Raises:
        StoragePolicyError: when ``hf_first`` is active but the
            forecaster bucket has no HF repo configured.
    """
    if policy is None:
        policy = get_storage_policy()

    if snapshot_storage.snapshot_exists(cutoff_date):
        return True

    hf_repo = snapshot_storage._get_hf_repo()
    if policy is StoragePolicy.HF_FIRST and not hf_repo:
        raise StoragePolicyError(
            f"hf_first policy requires HF repo for snapshot bucket "
            f"{snapshot_storage.forecaster_type!r}; got none."
        )
    if not hf_repo:
        return False

    return snapshot_storage.download_snapshot_from_hf(cutoff_date)


__all__ = [
    "ENV_STORAGE_BACKEND",
    "StoragePolicy",
    "StoragePolicyError",
    "ensure_snapshot_for_bucket",
    "get_prior_metadata_for_bucket",
    "get_storage_policy",
    "hf_versions_match",
    "load_current_artifacts_for_bucket",
]
