"""Signal route handlers for SAC price and market-history evidence."""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from brain_api.core.lstm import load_prices_yfinance
from brain_api.core.sac.market_sessions import completed_xnys_session_dates
from brain_api.core.vix_fallback import VixFallbackError, apply_cboe_vix_fallback
from brain_api.routes.signals.models import (
    ClosesRequest,
    ClosesResponse,
    MarketHistoryRequest,
    MarketHistoryResponse,
    MarketHistoryRow,
)

router = APIRouter()


def get_data_base_path() -> Path:
    return Path("data")


@router.post("/prices", response_model=ClosesResponse)
def get_closes(request: ClosesRequest) -> ClosesResponse:
    """Adjusted closes and finite positive as-of prices for SAC v3.

    Short histories are returned intact so Brain's eligibility gate can mask
    the asset. Provider-wide failure raises; no signal is silently zero-filled.
    """
    as_of = date.fromisoformat(request.as_of_date)
    calendar_days = int(request.lookback_bars * 365 / 252) + 30
    start = as_of - timedelta(days=calendar_days)
    prices = load_prices_yfinance(request.symbols, start, as_of + timedelta(days=1))
    if not prices:
        raise HTTPException(status_code=503, detail="Adjusted-close provider failed")

    adjusted_closes: dict[str, list[float]] = {}
    for symbol in request.symbols:
        price_df = prices.get(symbol)
        if price_df is None or price_df.empty:
            adjusted_closes[symbol] = []
            continue
        series = price_df["close"]
        index = series.index
        if index.tz is not None:
            index = index.tz_localize(None)
        series = series.set_axis(index)
        series = series[series.index.normalize() <= pd.Timestamp(as_of)]
        tail = series.tail(request.lookback_bars)
        if bool(np.all(np.isfinite(tail))) and bool(np.all(tail > 0)):
            adjusted_closes[symbol] = [float(v) for v in tail]
        else:
            adjusted_closes[symbol] = []
    return ClosesResponse(
        as_of_date=request.as_of_date,
        adjusted_closes=adjusted_closes,
        provenance={
            "provider": "yfinance",
            "price_basis": "adjusted",
            "requested_symbols": request.symbols,
            "lookback_bars": request.lookback_bars,
        },
    )


@router.post("/market-history", response_model=MarketHistoryResponse)
def get_market_history(request: MarketHistoryRequest) -> MarketHistoryResponse:
    """Return gap-free, session-aligned raw SPY/VIX history for SAC v3."""
    start = request.start_date
    as_of = request.as_of_date
    if start > as_of:
        raise HTTPException(status_code=422, detail="start_date must be <= as_of_date")

    expected_dates = completed_xnys_session_dates(start, as_of)
    provenance = {
        "provider": "yfinance",
        "spy_price_basis": "adjusted",
        "vix_price_basis": "close",
        "calendar": "XNYS",
        "completed_sessions_only": True,
    }
    if not expected_dates:
        return MarketHistoryResponse(
            start_date=start,
            as_of_date=as_of,
            rows=[],
            provenance=provenance,
        )

    prices = load_prices_yfinance(["SPY", "^VIX"], start, as_of)
    try:
        vix_result = apply_cboe_vix_fallback(prices, required_dates=expected_dates)
    except VixFallbackError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    prices = vix_result.prices
    provenance["vix_fallback"] = vix_result.audit.to_dict()
    if set(prices) != {"SPY", "^VIX"}:
        raise HTTPException(
            status_code=503,
            detail="SPY/VIX market-history provider returned an incomplete response",
        )

    def _by_date(symbol: str) -> dict[date, float]:
        series = prices[symbol]["close"]
        index = (
            series.index.tz_localize(None)
            if series.index.tz is not None
            else series.index
        )
        return {
            timestamp.date(): float(value)
            for timestamp, value in series.set_axis(index).items()
            if np.isfinite(value) and value > 0
        }

    spy = _by_date("SPY")
    vix = _by_date("^VIX")
    missing = [
        session.isoformat()
        for session in expected_dates
        if session not in spy or session not in vix
    ]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Gap in aligned SPY/VIX market history: {missing}",
        )
    rows = [
        MarketHistoryRow(
            date=session.isoformat(),
            spy_adjusted_close=spy[session],
            vix_close=vix[session],
        )
        for session in expected_dates
    ]
    return MarketHistoryResponse(
        start_date=start,
        as_of_date=as_of,
        rows=rows,
        provenance=provenance,
    )
