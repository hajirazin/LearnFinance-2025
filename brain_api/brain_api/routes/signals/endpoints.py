"""Signal route handlers."""

from datetime import date, timedelta
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from brain_api.core.lstm import load_prices_yfinance
from brain_api.core.news_sentiment import (
    NewsFetcher,
    SentimentScorer,
    process_news_sentiment,
)
from brain_api.core.sac.market_sessions import completed_xnys_session_dates
from brain_api.routes.signals.dependencies import (
    get_data_base_path,
    get_news_fetcher,
    get_sentiment_parquet_path,
    get_sentiment_scorer,
)
from brain_api.routes.signals.helpers import (
    load_historical_sentiment,
    result_to_response,
)
from brain_api.routes.signals.models import (
    ClosesRequest,
    ClosesResponse,
    HistoricalNewsSentimentRequest,
    HistoricalNewsSentimentResponse,
    MarketHistoryRequest,
    MarketHistoryResponse,
    MarketHistoryRow,
    NewsSignalRequest,
    NewsSignalResponse,
)

router = APIRouter()


# ============================================================================
# News sentiment endpoints
# ============================================================================


@router.post("/news", response_model=NewsSignalResponse)
def get_news_sentiment(
    request: NewsSignalRequest,
    fetcher: Annotated[NewsFetcher, Depends(get_news_fetcher)],
    scorer: Annotated[SentimentScorer, Depends(get_sentiment_scorer)],
    base_path: Annotated[Path, Depends(get_data_base_path)],
) -> NewsSignalResponse:
    """Get news sentiment scores for the given symbols.

    This endpoint:
    1. Fetches news articles from yfinance for each symbol
    2. Scores each article using FinBERT (financial sentiment model)
    3. Computes a recency-weighted aggregate score per symbol
    4. Persists raw articles and features for audit/training
    5. Returns top K articles per symbol with scores

    If data already exists for the same run_id+attempt, returns cached results
    (idempotent for retries).
    """
    # Parse as-of date
    if request.as_of_date:
        as_of = date.fromisoformat(request.as_of_date)
    else:
        as_of = date.today()

    # Derive run_id if not provided
    run_id = request.run_id
    if run_id is None:
        run_id = f"paper:{as_of.isoformat()}"

    # Ensure return_top_k doesn't exceed max_articles_per_symbol
    return_top_k = min(request.return_top_k, request.max_articles_per_symbol)

    # Process news sentiment (with caching)
    result = process_news_sentiment(
        symbols=request.symbols,
        fetcher=fetcher,
        scorer=scorer,
        as_of_date=as_of,
        max_articles_per_symbol=request.max_articles_per_symbol,
        run_id=run_id,
        attempt=request.attempt,
        base_path=base_path,
    )

    return result_to_response(result, return_top_k)


@router.post("/news/historical", response_model=HistoricalNewsSentimentResponse)
def get_historical_news_sentiment(
    request: HistoricalNewsSentimentRequest,
    parquet_path: Annotated[Path, Depends(get_sentiment_parquet_path)],
) -> HistoricalNewsSentimentResponse:
    """Get HISTORICAL news sentiment for training (date range).

    Returns sentiment scores for all requested (date, symbol) combinations.
    Missing data is filled with neutral sentiment (score=0.0).

    Data source: Pre-computed daily_sentiment.parquet (from news_sentiment_etl)
    """
    data = load_historical_sentiment(
        parquet_path=parquet_path,
        symbols=request.symbols,
        start_date=request.start_date,
        end_date=request.end_date,
    )

    return HistoricalNewsSentimentResponse(
        start_date=request.start_date,
        end_date=request.end_date,
        data=data,
    )


@router.post("/prices", response_model=ClosesResponse)
def get_closes(request: ClosesRequest) -> ClosesResponse:
    """Adjusted closes and finite positive as-of prices for SAC v3.

    Short histories are returned intact so Brain's eligibility gate can mask
    the asset. Provider-wide failure raises; no signal is silently zero-filled.
    """
    as_of = date.fromisoformat(request.as_of_date)
    # Buffer for weekends/holidays; ~365/252 calendar days per trading day.
    calendar_days = int(request.lookback_bars * 365 / 252) + 30
    start = as_of - timedelta(days=calendar_days)
    prices = load_prices_yfinance(request.symbols, start, as_of + timedelta(days=1))
    if not prices:
        raise HTTPException(status_code=503, detail="Adjusted-close provider failed")

    adjusted_closes: dict[str, list[float]] = {}
    execution_prices: dict[str, float] = {}
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
        latest = float(series.iloc[-1])
        if np.isfinite(latest) and latest > 0:
            execution_prices[symbol] = latest

    return ClosesResponse(
        as_of_date=request.as_of_date,
        adjusted_closes=adjusted_closes,
        execution_prices=execution_prices,
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
