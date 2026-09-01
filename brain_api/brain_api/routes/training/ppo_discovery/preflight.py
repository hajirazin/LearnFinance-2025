"""``POST /train/ppo-discovery/preflight``."""

from __future__ import annotations

from datetime import UTC, date, datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from brain_api.core.ppo_discovery.config import UNIVERSE_NAME
from brain_api.core.ppo_discovery.price_readiness import assess_price_readiness
from brain_api.core.ppo_discovery.universe_snapshot import resolve_universe_snapshot
from brain_api.core.ppo_discovery.weeks import (
    actor_cutoff_datetimes,
    weekly_trade_clock,
)

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
    try:
        price_start = date(end.year - 7, 1, 1)
        clock = weekly_trade_clock(price_start, end)
        transition_cutoffs = actor_cutoff_datetimes(clock)[:-1]
        index_end = transition_cutoffs[-1].date()
    except (IndexError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"PPO preflight needs at least two weekly rebalance sessions: {exc}",
        ) from exc
    readiness = assess_price_readiness(
        snapshot.sorted_symbols,
        start_date=price_start,
        end_date=end,
        index_end_date=index_end,
    )
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
        "exclusions": readiness["exclusions"],
        "session_hashes": readiness["session_hashes"],
        "session_counts": readiness["session_counts"],
        "eligible_symbol_count": readiness["eligible_symbol_count"],
        "vix_provenance": readiness["vix_provenance"],
        "market_history_end_date": readiness["index_end_date"],
    }
