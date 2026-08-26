"""``POST /etl/ppo-discovery/news-history`` — one complete partition per week."""

from __future__ import annotations

from datetime import UTC, date, datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from brain_api.core.ppo_discovery.config import UNIVERSE_NAME
from brain_api.core.ppo_discovery.news_evidence import NewsEvidenceError
from brain_api.core.ppo_discovery.news_history import materialize_weekly_news_history
from brain_api.core.ppo_discovery.universe_snapshot import resolve_universe_snapshot

router = APIRouter()


class PPONewsHistoryRequest(BaseModel):
    start_date: str
    end_date: str
    universe: str = Field(default=UNIVERSE_NAME)
    force: bool = False


@router.post("/ppo-discovery/news-history")
def etl_ppo_discovery_news_history(request: PPONewsHistoryRequest) -> dict:
    """Freeze ``halal_new`` then materialize one news partition per weekly cutoff."""
    if request.universe != UNIVERSE_NAME:
        raise HTTPException(
            status_code=422,
            detail=f"ppo_discovery ETL universe must be {UNIVERSE_NAME!r}",
        )
    try:
        snapshot = resolve_universe_snapshot(datetime.now(UTC))
        start = date.fromisoformat(request.start_date[:10])
        end = date.fromisoformat(request.end_date[:10])
        result = materialize_weekly_news_history(
            snapshot.sorted_symbols,
            start,
            end,
            force=request.force,
        )
    except NewsEvidenceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "universe": snapshot.universe,
        "snapshot_sha256": snapshot.snapshot_sha256,
        "symbol_count": snapshot.symbol_count,
        **result,
    }
