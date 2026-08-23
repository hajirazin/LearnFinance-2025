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
from enum import StrEnum
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


class StoragePolicy(StrEnum):
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
    cold_start_status_code: int = 503,
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
        cold_start_status_code: Status code to use when no model
            exists anywhere (genuine cold-start). Defaults to ``503``
            (Service Unavailable) so inference routes match the
            AGENTS.md "Cold start ... surfaces as a 503 for inference"
            contract. ``/models/active-symbols`` opts into ``400``
            because its legacy contract is ``400 + "Train one first."``.
            Only the two genuine cold-start branches honour this knob:
            ``local_first`` "local empty + no HF repo configured" and
            ``hf_first`` "HF main is missing". Transient/config 503s
            (HF unreachable, ``hf_first`` without a repo, HF download
            failed) stay as ``503`` regardless because they are NOT
            cold-start (a model could still exist; the failure is
            recoverable).

    Returns:
        The model-specific artifacts object (``LSTMArtifacts``,
        ``PatchTSTArtifacts``, or ``SACArtifacts``) ready for
        inference.

    Raises:
        HTTPException ``cold_start_status_code``: on genuine cold-start
            (no model anywhere).
        HTTPException 503: on transient or config failures (HF
            unreachable, ``hf_first`` without a repo, HF download
            failed).
    """
    if policy is None:
        policy = get_storage_policy()
    local_storage = bucket.local_storage_class()

    if policy is StoragePolicy.LOCAL_FIRST:
        return _load_local_first(
            local_storage, bucket, model_label, cold_start_status_code
        )
    return _load_hf_first(local_storage, bucket, model_label, cold_start_status_code)


def _load_local_first(
    local_storage: Any,
    bucket: BucketConfig,
    model_label: str,
    cold_start_status_code: int,
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
        # Cold-start: no model anywhere (local empty + no HF repo
        # configured for this bucket). Honour the caller-supplied
        # status code so /models/active-symbols can preserve its
        # legacy 400 contract while inference routes stay on 503.
        raise HTTPException(
            status_code=cold_start_status_code,
            detail=(
                f"No {model_label} model available: local is empty and no "
                f"HF repo is configured for bucket {bucket.bucket_name!r}. "
                f"Train the model first or set the bucket's HF repo env var."
            ),
        )
    try:
        return hf_storage.download_model(use_cache=True)
    except Exception as hf_err:
        # Transient: a model could still exist on HF; this is a
        # recoverable failure (network, auth, etc.). Always 503
        # regardless of cold_start_status_code.
        raise HTTPException(
            status_code=503,
            detail=(
                f"No {model_label} model available: local empty and HF "
                f"({hf_repo}) failed: {hf_err}"
            ),
        ) from None


def _load_hf_first(
    local_storage: Any,
    bucket: BucketConfig,
    model_label: str,
    cold_start_status_code: int,
) -> Any:
    """Check HF main metadata; reuse local only if it matches that exact version."""
    hf_storage, hf_repo = _instantiate_hf_storage(bucket, local_storage)
    if hf_storage is None:
        # Config error: hf_first policy was selected but the bucket
        # has no HF repo configured. NOT cold-start (a local model
        # could still exist; the policy just refuses to use it).
        # Always 503 regardless of cold_start_status_code.
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
        # Transient: HF metadata fetch failed (network/auth). Always 503.
        raise HTTPException(status_code=503, detail=str(exc)) from None

    if hf_metadata is None:
        # Cold-start: HF main is missing -> no model has ever been
        # promoted to this bucket. Honour the caller-supplied status
        # code so /models/active-symbols can preserve its legacy 400.
        raise HTTPException(
            status_code=cold_start_status_code,
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
# Training: HF-aware "version already exists" idempotency skip
# ---------------------------------------------------------------------------


def try_load_existing_train_metadata(
    *,
    bucket: BucketConfig,
    version: str,
    local_storage: Any,
    policy: StoragePolicy | None = None,
) -> dict[str, Any] | None:
    """Return metadata for ``version`` if it exists locally OR on HF.

    Used by every ``/train/*`` endpoint to short-circuit retraining
    when the deterministic version (``v{end_date}-{hash}``) has
    already been produced. Mirrors the ``hf_first`` contract used by
    ``/inference/*`` read paths so a wiped local cache (Pi cold
    start, Mac reset) does not silently retrain work that already
    exists on HF.

    * Local hit (always checked first): ``read_metadata`` from disk.
    * HF hit (``hf_first`` only): single ``hf_hub_download`` of
      ``metadata.json`` at ``revision=<version>``. No artifact
      download -- inference will populate local on demand via the
      existing read path. Each version is its own HF branch (see
      :class:`brain_api.storage.base_huggingface.BaseHuggingFaceModelStorage.list_versions`),
      so ``revision=version`` is the right pointer.
    * Miss: return ``None`` so the caller proceeds with training.

    Under ``local_first`` the HF check is skipped (current behaviour
    preserved). Per AGENTS.md rule #1, every failure path returns
    ``None`` rather than raising -- training is the recovery action,
    not a hard failure.

    Args:
        bucket: ``BucketConfig`` for the ``(model, universe)`` pair.
        version: Deterministic version string (``v{end_date}-{hash}``).
        local_storage: The bucket's local storage instance. Passed in
            (rather than re-instantiated) so the caller can keep the
            same instance it already used for downstream writes.
        policy: Override; when ``None``, reads ``STORAGE_BACKEND``.

    Returns:
        Metadata dict from local or HF, or ``None`` if neither has
        the version (or HF is unreachable / has no repo configured).
    """
    if local_storage.version_exists(version):
        return local_storage.read_metadata(version)

    if policy is None:
        policy = get_storage_policy()
    if policy is not StoragePolicy.HF_FIRST:
        return None

    hf_storage, hf_repo = _instantiate_hf_storage(bucket, local_storage)
    if hf_storage is None:
        # No HF repo configured for this bucket; nothing to consult.
        # Returning None lets the caller train rather than 500.
        return None

    try:
        path = hf_hub_download(
            repo_id=hf_repo,
            filename="metadata.json",
            repo_type="model",
            revision=version,
            token=hf_storage.token,
        )
    except (RevisionNotFoundError, EntryNotFoundError, RepositoryNotFoundError):
        # The branch / file genuinely doesn't exist -> not a hit.
        return None
    except Exception as exc:
        # Transient error (network, auth, rate limit). Log and treat
        # as miss so the caller can retrain rather than wedge on a
        # transient HF outage. We deliberately don't raise: this is
        # an idempotency optimization, not a correctness gate.
        logger.warning(
            f"[storage-policy:try_load_existing_train_metadata] "
            f"{bucket.bucket_name}: HF metadata fetch for {version!r} "
            f"failed transiently (treating as miss): {exc}"
        )
        return None
    with open(path) as f:
        metadata: dict[str, Any] = json.load(f)
    logger.info(
        f"[storage-policy:try_load_existing_train_metadata] "
        f"{bucket.bucket_name}: HF hit for {version!r}; skipping retrain."
    )
    return metadata


def build_common_train_response_kwargs(
    version: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build the 7 kwargs every ``*TrainResponse`` class shares.

    Extracted from the line-for-line identical idempotency-skip
    blocks across LSTM / PatchTST x2 / PatchTST India / SAC full.
    Per AGENTS.md "Code reuse" -- this is provably
    identical metadata-to-response mapping (no algorithm-specific
    math), and per AGENTS.md rule #2 each call site keeps its
    model-specific extras (``num_input_channels`` + ``signals_used``
    for PatchTST; ``symbols_used`` for SAC) so the differences in
    response shape are still visible at the call site.
    """
    return {
        "version": version,
        "data_window_start": metadata["data_window"]["start"],
        "data_window_end": metadata["data_window"]["end"],
        "metrics": metadata["metrics"],
        "promoted": metadata["promoted"],
        "prior_version": metadata.get("prior_version"),
        # Backward-compat: pre-guardrail metadata files have no
        # ``failure_reasons`` key. Treat missing as empty list so
        # old artifacts continue to deserialize.
        "failure_reasons": metadata.get("failure_reasons", []),
    }


# ---------------------------------------------------------------------------
# Snapshots: ensure-available helper for generic forecaster walk-forward loading
# ---------------------------------------------------------------------------


def ensure_snapshot_for_bucket(
    *,
    snapshot_storage: SnapshotLocalStorage,
    cutoff_date: date,
    policy: StoragePolicy | None = None,
) -> bool:
    """Ensure the expected hashed forecaster snapshot is available locally.

    Dec-31 walk-forward snapshots use resolver symbols + default forecast
    config and the extended backfill window start (see
    :mod:`brain_api.core.forecaster_snapshot_identity`). Folder / HF branch names
    are ``snapshot-{{cutoff}}-{{digest}}``.
    """

    from brain_api.core.forecaster_snapshot_identity import (
        expected_dec31_walkforward_snapshot_hash,
        lstm_walkforward_expectation_bundle,
        patchtst_walkforward_expectation_bundle,
    )

    bucket_type = snapshot_storage.forecaster_type
    if bucket_type == "lstm_halal_new":
        identity_bucket, wf_symbols, wf_cfg = lstm_walkforward_expectation_bundle()
    elif bucket_type == "patchtst_halal_new":
        identity_bucket, wf_symbols, wf_cfg = patchtst_walkforward_expectation_bundle()
    else:
        raise StoragePolicyError(
            f"Walk-forward snapshot ensure is wired only for lstm_halal_new and "
            f"patchtst_halal_new; got {bucket_type!r}"
        )

    snapshot_digest = expected_dec31_walkforward_snapshot_hash(
        forecaster_bucket=identity_bucket,
        cutoff_date=cutoff_date,
        resolver_symbols=wf_symbols,
        config_dict=wf_cfg,
    )

    return snapshot_storage.ensure_snapshot_available(
        cutoff_date, snapshot_digest, policy=policy
    )


__all__ = [
    "ENV_STORAGE_BACKEND",
    "StoragePolicy",
    "StoragePolicyError",
    "build_common_train_response_kwargs",
    "ensure_snapshot_for_bucket",
    "get_prior_metadata_for_bucket",
    "get_storage_policy",
    "hf_versions_match",
    "load_current_artifacts_for_bucket",
    "try_load_existing_train_metadata",
]
