"""``POST /signals/ppo-discovery/state``."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from brain_api.core.model_buckets import ModelType, get_bucket
from brain_api.core.ppo_discovery.config import UNIVERSE_NAME
from brain_api.core.ppo_discovery.news_adapter import (
    build_ppo_news_features,
    features_to_schema,
)
from brain_api.core.ppo_discovery.regime import (
    live_regime_probabilities,
    spy_vix_rows_after_cutoff,
)
from brain_api.core.ppo_discovery.schemas import PPODiscoveryError, SymbolNewsFeatures
from brain_api.core.ppo_discovery.state_builder import (
    StateBuildRequest,
    build_ppo_discovery_state,
)
from brain_api.core.ppo_discovery.universe_snapshot import resolve_universe_snapshot
from brain_api.core.prices import load_prices_yfinance
from brain_api.core.sac.market_sessions import completed_xnys_session_dates
from brain_api.core.sac.regime_hmm import RegimeHMMArtifact
from brain_api.core.vix_fallback import VixFallbackError, apply_cboe_vix_fallback
from brain_api.core.weekly_decision import (
    MondayCutoffError,
    monday_window_bounds,
    require_monday_decision_cutoff,
)
from brain_api.news.errors import NewsError
from brain_api.news.models import NewsWindow
from brain_api.news.service import NewsService, raise_http_status
from brain_api.news.store import NewsStore
from brain_api.storage.base import DEFAULT_DATA_PATH
from brain_api.storage.policy import load_current_artifacts_for_bucket

router = APIRouter()

ENCODER_CALENDAR_LOOKBACK_DAYS = 450


def get_news_service() -> NewsService:
    return NewsService(NewsStore(DEFAULT_DATA_PATH))


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
        cutoff = require_monday_decision_cutoff(as_of)
        snapshot = resolve_universe_snapshot(as_of)
        start_exclusive, end_inclusive = monday_window_bounds(cutoff.date())
        window = NewsWindow(
            start_exclusive=start_exclusive, end_inclusive=end_inclusive
        )
        _coverage, events = get_news_service().materialize(
            list(snapshot.sorted_symbols), window
        )
        events_by_symbol: dict[str, list] = {
            symbol: [] for symbol in snapshot.sorted_symbols
        }
        for event in events:
            if event.symbol in events_by_symbol:
                events_by_symbol[event.symbol].append(event)
        adapter = build_ppo_news_features(events_by_symbol, cutoff=cutoff)
        news: dict[str, SymbolNewsFeatures] = {
            symbol: features_to_schema(
                symbol, adapter[symbol], events_by_symbol[symbol], cutoff=cutoff
            )
            for symbol in snapshot.sorted_symbols
        }
        bucket = get_bucket(ModelType.PPO_DISCOVERY, UNIVERSE_NAME)
        artifacts = load_current_artifacts_for_bucket(
            bucket=bucket, model_label=bucket.model_label
        )
        scalers = artifacts.feature_scalers
        hmm_payload = artifacts.regime_hmm or {}
        try:
            hmm_artifact = RegimeHMMArtifact.from_dict(hmm_payload)
        except ValueError as exc:
            raise PPODiscoveryError(
                f"regime_hmm artifact cannot continue causally: {exc}"
            ) from exc
        decision_date = as_of.date()
        encoder_start = (as_of - timedelta(days=ENCODER_CALENDAR_LOOKBACK_DAYS)).date()
        ohlcv = load_prices_yfinance(
            list(snapshot.sorted_symbols),
            encoder_start,
            decision_date,
        )
        spy_vix = load_prices_yfinance(
            ["SPY", "^VIX"],
            hmm_artifact.training_cutoff_date,
            decision_date,
        )
        required_vix_dates = completed_xnys_session_dates(
            hmm_artifact.training_cutoff_date + timedelta(days=1), decision_date
        )
        vix_result = apply_cboe_vix_fallback(spy_vix, required_dates=required_vix_dates)
        spy_vix = vix_result.prices
        spy = spy_vix.get("SPY")
        if spy is None or spy.empty:
            raise PPODiscoveryError("SPY history missing")
        if "^VIX" not in spy_vix:
            raise PPODiscoveryError("^VIX history missing")
        rows = spy_vix_rows_after_cutoff(
            spy_vix,
            cutoff=hmm_artifact.training_cutoff_date,
            decision_date=decision_date,
        )
        p_calm, p_stress = live_regime_probabilities(
            hmm_payload, spy_vix_rows=rows, decision_date=decision_date
        )
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
                market_history_provenance={"vix_fallback": vix_result.audit.to_dict()},
            )
        )
    except MondayCutoffError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except NewsError as exc:
        raise HTTPException(
            status_code=raise_http_status(exc), detail=str(exc)
        ) from exc
    except PPODiscoveryError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except VixFallbackError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return state.to_dict()
