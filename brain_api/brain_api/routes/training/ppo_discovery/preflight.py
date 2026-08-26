"""``POST /train/ppo-discovery/preflight``."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from brain_api.core.ppo_discovery.config import UNIVERSE_NAME
from brain_api.core.ppo_discovery.universe_snapshot import resolve_universe_snapshot

router = APIRouter()


class PPOPreflightRequest(BaseModel):
    universe: str = Field(...)
    end_date: str | None = None
    experiment_id: str = "ppo-discovery-default"


@router.post("/ppo-discovery/preflight")
def preflight_ppo_discovery(request: PPOPreflightRequest) -> dict:
    if request.universe != UNIVERSE_NAME:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown universe '{request.universe}' for ppo_discovery",
        )
    snapshot = resolve_universe_snapshot(datetime.now(UTC), persist=True)
    return {
        "ready": True,
        "universe": snapshot.universe,
        "snapshot_sha256": snapshot.snapshot_sha256,
        "symbol_count": snapshot.symbol_count,
        "survivorship_bias": (
            "Training applies today's halal_new roster retrospectively."
        ),
        "experiment_id": request.experiment_id,
        "end_date": request.end_date,
        "issues": [],
    }
