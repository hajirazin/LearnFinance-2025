"""``POST /etl/ppo-discovery/news-history`` — PPO weekly news parquet only."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from brain_api.core.ppo_discovery.config import UNIVERSE_NAME
from brain_api.core.ppo_discovery.news_evidence import (
    NewsEvidenceError,
    materialize_news_evidence,
    news_window_for_cutoff,
    persist_weekly_news_features,
)
from brain_api.core.ppo_discovery.universe_snapshot import resolve_universe_snapshot

router = APIRouter()


class PPONewsHistoryRequest(BaseModel):
    start_date: str
    end_date: str
    universe: str = Field(default=UNIVERSE_NAME)
    force: bool = False


@router.post("/ppo-discovery/news-history")
def etl_ppo_discovery_news_history(request: PPONewsHistoryRequest) -> dict:
    """Freeze ``halal_new`` then materialize complete weekly news partitions."""
    if request.universe != UNIVERSE_NAME:
        raise HTTPException(
            status_code=422,
            detail=f"ppo_discovery ETL universe must be {UNIVERSE_NAME!r}",
        )
    try:
        snapshot = resolve_universe_snapshot(datetime.now(UTC))
        cutoff = datetime.fromisoformat(request.end_date)
        if cutoff.tzinfo is None:
            cutoff = cutoff.replace(tzinfo=UTC)
        previous = datetime.fromisoformat(request.start_date)
        if previous.tzinfo is None:
            previous = previous.replace(tzinfo=UTC)
        features = materialize_news_evidence(
            snapshot.sorted_symbols,
            cutoff,
            previous_cutoff=previous,
        )
        window = news_window_for_cutoff(cutoff, previous)
        path = persist_weekly_news_features(cutoff, features, window=window)
    except NewsEvidenceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "universe": snapshot.universe,
        "snapshot_sha256": snapshot.snapshot_sha256,
        "symbol_count": snapshot.symbol_count,
        "completed": len(features),
        "incomplete": 0,
        "article_counts": {
            symbol: row.article_count for symbol, row in features.items()
        },
        "parquet_path": str(path),
        "force": request.force,
    }
