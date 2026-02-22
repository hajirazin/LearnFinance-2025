"""Model metadata endpoints."""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from brain_api.storage.sac import SACLocalStorage

router = APIRouter()
logger = logging.getLogger(__name__)


class ActiveSymbolsResponse(BaseModel):
    """Response for the active symbols endpoint."""

    symbols: list[str]
    source_model: str
    model_version: str


@router.get("/active-symbols", response_model=ActiveSymbolsResponse)
def get_active_symbols() -> ActiveSymbolsResponse:
    """Get symbols from the currently promoted SAC allocator model.

    Used by the inference pipeline to determine which symbols to
    fetch signals for. SAC is the reference because all allocators
    (PPO, SAC, HRP) operate on the same symbol set.
    """
    storage = SACLocalStorage()
    version = storage.read_current_version()
    if not version:
        raise HTTPException(400, "No promoted SAC model. Train one first.")

    symbols = storage.load_symbol_order(version)
    logger.info(f"[Models] Active symbols from SAC {version}: {len(symbols)} symbols")

    return ActiveSymbolsResponse(
        symbols=symbols,
        source_model="sac",
        model_version=version,
    )
