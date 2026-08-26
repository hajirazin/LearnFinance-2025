"""Model metadata endpoints."""

import logging

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from brain_api.core.model_buckets import (
    ModelType,
    UnknownBucketError,
    get_bucket,
    list_universes_for,
)
from brain_api.storage.policy import load_current_artifacts_for_bucket

router = APIRouter()
logger = logging.getLogger(__name__)


class ActiveSymbolsResponse(BaseModel):
    """Response for the active symbols endpoint."""

    symbols: list[str]
    source_model: str
    model_version: str
    training_cutoff_date: str
    sac_schema_version: int


@router.get("/active-symbols", response_model=ActiveSymbolsResponse)
def get_active_symbols(
    universe: str = Query(
        ...,
        description=(
            "SAC bucket universe ('halal_filtered' or 'halal'). Mandatory: "
            "the two A/B SAC buckets have independent symbol orders, so the "
            "caller must declare which slate to read. Per AGENTS.md rule #1 "
            "(no silent fallbacks) there is no default."
        ),
    ),
) -> ActiveSymbolsResponse:
    """Get symbols from the currently promoted SAC allocator model.

    Used by the inference pipeline to determine which symbols to
    fetch signals for. Resolves the SAC bucket via
    ``get_bucket(ModelType.SAC, universe)`` so callers select between
    ``sac_halal_filtered`` (sticky-15 from PatchTST) and ``sac_halal``
    (legacy yfinance halal universe; variable size).

    Routes through :func:`load_current_artifacts_for_bucket` so that
    under ``STORAGE_BACKEND=hf_first`` the symbol slate is recovered
    from HF when local is empty (e.g. on a freshly-deployed Pi). This
    matches the contract used by ``/inference/sac`` -- the workflow's
    Phase-0 symbol read and Phase-2 inference now share a single read
    path rather than disagreeing on hf_first cold-start behaviour.

    Raises:
        HTTPException 422: if ``universe`` is not a registered SAC
            bucket (e.g. typo'd ``halal_new``).
        HTTPException 400: on genuine cold-start (no model anywhere).
            The legacy contract -- "Train one first." -- is preserved
            via the ``cold_start_status_code=400`` knob; transient
            failures (HF unreachable, etc.) still surface as 503.
    """
    try:
        bucket = get_bucket(ModelType.SAC, universe)
    except UnknownBucketError as exc:
        allowed = sorted(list_universes_for(ModelType.SAC))
        raise HTTPException(
            status_code=422,
            detail=f"Unknown universe '{universe}' for SAC. Allowed: {allowed}",
        ) from exc

    artifacts = load_current_artifacts_for_bucket(
        bucket=bucket,
        model_label=bucket.model_label,
        cold_start_status_code=400,
    )

    logger.info(
        f"[Models] Active symbols from {bucket.bucket_name} "
        f"({artifacts.version}): {len(artifacts.symbol_order)} symbols"
    )

    return ActiveSymbolsResponse(
        symbols=artifacts.symbol_order,
        source_model=bucket.bucket_name,
        model_version=artifacts.version,
        training_cutoff_date=artifacts.v3_auxiliary.training_cutoff_date,
        sac_schema_version=artifacts.metadata["sac_schema_version"],
    )


class PPODiscoveryActiveResponse(BaseModel):
    """Promoted ppo_discovery artifact identity."""

    universe: str
    model_type: str
    model_version: str
    snapshot_sha256: str | None
    news_required: bool


@router.get("/ppo-discovery/active", response_model=PPODiscoveryActiveResponse)
def get_ppo_discovery_active(
    universe: str = Query(..., description="Must be halal_new"),
) -> PPODiscoveryActiveResponse:
    try:
        bucket = get_bucket(ModelType.PPO_DISCOVERY, universe)
    except UnknownBucketError as exc:
        allowed = sorted(list_universes_for(ModelType.PPO_DISCOVERY))
        raise HTTPException(
            status_code=422,
            detail=f"Unknown universe '{universe}' for ppo_discovery. Allowed: {allowed}",
        ) from exc
    artifacts = load_current_artifacts_for_bucket(
        bucket=bucket,
        model_label=bucket.model_label,
        cold_start_status_code=503,
    )
    return PPODiscoveryActiveResponse(
        universe=universe,
        model_type="ppo_discovery",
        model_version=artifacts.version,
        snapshot_sha256=(artifacts.universe_manifest or {}).get("snapshot_sha256"),
        news_required=bool(artifacts.metadata.get("news_required")),
    )
