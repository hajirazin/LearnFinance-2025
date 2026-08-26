"""``POST /train/ppo-discovery/preflight``."""

from __future__ import annotations

from datetime import UTC, date, datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from brain_api.core.ppo_discovery.config import UNIVERSE_NAME
from brain_api.core.ppo_discovery.price_readiness import assess_price_readiness
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
    if request.end_date:
        end = date.fromisoformat(request.end_date)
    else:
        end = datetime.now(UTC).date()
    readiness = assess_price_readiness(snapshot.sorted_symbols, end_date=end)
    return {
        "ready": readiness["ready"],
        "universe": snapshot.universe,
        "snapshot_sha256": snapshot.snapshot_sha256,
        "sorted_symbols": list(snapshot.sorted_symbols),
        "symbol_count": snapshot.symbol_count,
        "survivorship_bias": (
            "Training applies today's halal_new roster retrospectively."
        ),
        "experiment_id": request.experiment_id,
        "end_date": request.end_date or end.isoformat(),
        "issues": readiness["issues"],
        "session_hashes": readiness["session_hashes"],
        "session_counts": readiness["session_counts"],
        "eligible_symbol_count": readiness["eligible_symbol_count"],
    }
