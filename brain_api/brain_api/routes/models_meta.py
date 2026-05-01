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

router = APIRouter()
logger = logging.getLogger(__name__)


class ActiveSymbolsResponse(BaseModel):
    """Response for the active symbols endpoint."""

    symbols: list[str]
    source_model: str
    model_version: str


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

    Raises:
        HTTPException 422: if ``universe`` is not a registered SAC
            bucket (e.g. typo'd ``halal_new``).
        HTTPException 400: if no SAC model is promoted yet for the
            requested bucket.
    """
    try:
        bucket = get_bucket(ModelType.SAC, universe)
    except UnknownBucketError as exc:
        allowed = sorted(list_universes_for(ModelType.SAC))
        raise HTTPException(
            status_code=422,
            detail=f"Unknown universe '{universe}' for SAC. Allowed: {allowed}",
        ) from exc

    storage = bucket.local_storage_class()
    version = storage.read_current_version()
    if not version:
        raise HTTPException(
            400,
            f"No promoted SAC model in bucket '{bucket.bucket_name}'. Train one first.",
        )

    symbols = storage.load_symbol_order(version)
    logger.info(
        f"[Models] Active symbols from {bucket.bucket_name} "
        f"({version}): {len(symbols)} symbols"
    )

    return ActiveSymbolsResponse(
        symbols=symbols,
        source_model=bucket.bucket_name,
        model_version=version,
    )
