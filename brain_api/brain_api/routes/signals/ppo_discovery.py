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
from brain_api.core.ppo_discovery.regime import (
    live_regime_probabilities,
    spy_vix_rows_after_cutoff,
)
from brain_api.core.ppo_discovery.schemas import PPODiscoveryError
from brain_api.core.ppo_discovery.state_builder import (
    StateBuildRequest,
    build_ppo_discovery_state,
)
from brain_api.core.ppo_discovery.universe_snapshot import resolve_universe_snapshot
from brain_api.core.prices import load_prices_yfinance
from brain_api.core.sac.regime_hmm import RegimeHMMArtifact
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
        price_start = (as_of - timedelta(days=450)).date()
        decision_date = as_of.date()
        prices = load_prices_yfinance(
            [*list(snapshot.sorted_symbols), "SPY", "^VIX"],
            price_start,
            decision_date,
        )
        spy = prices.get("SPY")
        if spy is None or spy.empty:
            raise PPODiscoveryError("SPY history missing")
        if "^VIX" not in prices:
            raise PPODiscoveryError("^VIX history missing")
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
        rows = spy_vix_rows_after_cutoff(
            prices,
            cutoff=hmm_artifact.training_cutoff_date,
            decision_date=decision_date,
        )
        p_calm, p_stress = live_regime_probabilities(
            hmm_payload, spy_vix_rows=rows, decision_date=decision_date
        )
        ohlcv = {
            symbol: frame
            for symbol, frame in prices.items()
            if frame is not None and symbol not in {"SPY", "^VIX"}
        }
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
