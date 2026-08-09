"""Signal route handlers."""

from datetime import date, timedelta
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from brain_api.core.lstm import load_prices_yfinance
from brain_api.core.news_sentiment import (
    NewsFetcher,
    SentimentScorer,
    process_news_sentiment,
)
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
    """Raw daily closes for SAC momentum signals (momentum_1w/4w/12_1).

    Fail-loud: any requested symbol with fewer than ``lookback_bars``
    trading-day closes on/before ``as_of_date`` is a 422, never a
    silent zero-fill or truncated series.
    """
    as_of = date.fromisoformat(request.as_of_date)
    # Buffer for weekends/holidays; ~365/252 calendar days per trading day.
    calendar_days = int(request.lookback_bars * 365 / 252) + 30
    start = as_of - timedelta(days=calendar_days)
    prices = load_prices_yfinance(request.symbols, start, as_of)

    closes: dict[str, list[float]] = {}
    for symbol in request.symbols:
        price_df = prices.get(symbol)
        if price_df is None or price_df.empty:
            raise HTTPException(
                status_code=422, detail=f"Missing price history for {symbol}"
            )
        series = price_df["close"]
        index = series.index
        if index.tz is not None:
            index = index.tz_localize(None)
        series = series.set_axis(index)
        series = series[series.index.normalize() <= pd.Timestamp(as_of)]
        if len(series) < request.lookback_bars:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Only {len(series)} closes for {symbol}; need >= "
                    f"{request.lookback_bars} for momentum_12_1"
                ),
            )
        closes[symbol] = [float(v) for v in series.tail(request.lookback_bars)]

    return ClosesResponse(as_of_date=request.as_of_date, closes=closes)
