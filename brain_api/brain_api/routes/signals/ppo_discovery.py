"""``POST /signals/ppo-discovery/state``."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from brain_api.core.model_buckets import ModelType, get_bucket
from brain_api.core.ppo_discovery.config import UNIVERSE_NAME
from brain_api.core.ppo_discovery.news_evidence import (
    NewsEvidenceError,
    materialize_news_evidence,
)
from brain_api.core.ppo_discovery.schemas import PPODiscoveryError
from brain_api.core.ppo_discovery.state_builder import (
    StateBuildRequest,
    build_ppo_discovery_state,
)
from brain_api.core.ppo_discovery.universe_snapshot import resolve_universe_snapshot
from brain_api.core.prices import load_prices_yfinance
from brain_api.storage.policy import load_current_artifacts_for_bucket

router = APIRouter()


class PPOStateRequest(BaseModel):
    as_of: str
    run_id: str
    attempt: int = Field(ge=1)
    current_weights: dict[str, float]
    universe: str = UNIVERSE_NAME


@router.post("/ppo-discovery/state")
def build_state(request: PPOStateRequest) -> dict[str, Any]:
    if request.universe != UNIVERSE_NAME:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown universe '{request.universe}' for ppo_discovery",
        )
    try:
        as_of = datetime.fromisoformat(request.as_of)
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=UTC)
        snapshot = resolve_universe_snapshot(as_of)
        news = materialize_news_evidence(snapshot.sorted_symbols, as_of)
        prices = load_prices_yfinance(
            list(snapshot.sorted_symbols),
            (as_of - timedelta(days=450)).date(),
            as_of.date(),
        )
        spy_map = load_prices_yfinance(
            ["SPY"], (as_of - timedelta(days=450)).date(), as_of.date()
        )
        spy = spy_map.get("SPY")
        if spy is None or spy.empty:
            raise PPODiscoveryError("SPY history missing")
        bucket = get_bucket(ModelType.PPO_DISCOVERY, UNIVERSE_NAME)
        artifacts = load_current_artifacts_for_bucket(
            bucket=bucket, model_label=bucket.model_label
        )
        scalers = artifacts.feature_scalers
        hmm = artifacts.regime_hmm or {}
        if "p_calm" not in hmm or "p_stress" not in hmm:
            raise PPODiscoveryError("regime_hmm artifact missing p_calm/p_stress")
        p_calm = float(hmm["p_calm"])
        p_stress = float(hmm["p_stress"])
        ohlcv = {symbol: frame for symbol, frame in prices.items() if frame is not None}
        state = build_ppo_discovery_state(
            StateBuildRequest(
                as_of=as_of,
                universe_snapshot=snapshot,
                ohlcv_by_symbol=ohlcv,
                news_by_symbol=news,
                current_weights=request.current_weights,
                p_calm=p_calm,
                p_stress=p_stress,
                spy_closes=spy["close"].to_numpy(),
                feature_scalers=scalers,
            )
        )
    except NewsEvidenceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PPODiscoveryError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    payload = state.to_dict()
    payload["run_id"] = request.run_id
    payload["attempt"] = request.attempt
    return payload
