"""Signal route handlers."""

from datetime import date
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends

from brain_api.core.data_freshness import refresh_stale_fundamentals
from brain_api.core.fundamentals import load_historical_fundamentals_from_cache
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
    get_yfinance_ratios,
    load_historical_sentiment,
    result_to_response,
)
from brain_api.routes.signals.models import (
    ApiStatusResponse,
    CurrentRatiosResponse,
    FundamentalsRequest,
    FundamentalsResponse,
    HistoricalFundamentalsRequest,
    HistoricalFundamentalsResponse,
    HistoricalNewsSentimentRequest,
    HistoricalNewsSentimentResponse,
    NewsSignalRequest,
    NewsSignalResponse,
    RatiosResponse,
    RefreshFundamentalsRequest,
    RefreshFundamentalsResponse,
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


# ============================================================================
# Fundamentals endpoints
# ============================================================================


@router.post("/fundamentals", response_model=FundamentalsResponse)
def get_fundamentals(
    request: FundamentalsRequest,
) -> FundamentalsResponse:
    """Get CURRENT fundamental ratios for inference.

    This endpoint fetches the most recent available fundamentals for each symbol
    using yfinance. No rate limits, no caching needed.

    Data source: yfinance (ticker.info)
    """
    as_of = date.today().isoformat()
    results: list[CurrentRatiosResponse] = []

    for symbol in request.symbols:
        try:
            ratios = get_yfinance_ratios(symbol, as_of)
            results.append(
                CurrentRatiosResponse(
                    symbol=symbol,
                    ratios=ratios,
                    error=None if ratios else "No data available",
                )
            )
        except Exception as e:
            results.append(
                CurrentRatiosResponse(
                    symbol=symbol,
                    ratios=None,
                    error=str(e),
                )
            )

    return FundamentalsResponse(
        as_of_date=as_of,
        per_symbol=results,
    )


@router.post("/fundamentals/historical", response_model=HistoricalFundamentalsResponse)
def get_historical_fundamentals(
    request: HistoricalFundamentalsRequest,
    base_path: Annotated[Path, Depends(get_data_base_path)],
) -> HistoricalFundamentalsResponse:
    """Get HISTORICAL fundamental ratios for training (date range).

    Returns n symbols x m quarterly periods as a flat list. Each entry represents
    the financial ratios that would have been available at that point in time.

    Reads from cache ONLY. Use PUT /signals/fundamentals/historical to refresh cache.

    Data source: Local cache (originally from Alpha Vantage)
    """
    start_date = date.fromisoformat(request.start_date)
    end_date = date.fromisoformat(request.end_date)

    # Use the shared loader function (reads from cache only)
    fundamentals = load_historical_fundamentals_from_cache(
        symbols=request.symbols,
        start_date=start_date,
        end_date=end_date,
        base_path=base_path,
    )

    # Helper to convert NaN to None for JSON serialization
    def safe_float(value: float | None) -> float | None:
        import math

        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return value

    # Convert dict[str, DataFrame] to list[RatiosResponse]
    all_ratios: list[RatiosResponse] = []
    for symbol, df in fundamentals.items():
        for idx, row in df.iterrows():
            all_ratios.append(
                RatiosResponse(
                    symbol=symbol,
                    as_of_date=idx.strftime("%Y-%m-%d"),
                    gross_margin=safe_float(row.get("gross_margin")),
                    operating_margin=safe_float(row.get("operating_margin")),
                    net_margin=safe_float(row.get("net_margin")),
                    current_ratio=safe_float(row.get("current_ratio")),
                    debt_to_equity=safe_float(row.get("debt_to_equity")),
                )
            )

    return HistoricalFundamentalsResponse(
        start_date=request.start_date,
        end_date=request.end_date,
        data=all_ratios,
    )


@router.put("/fundamentals/historical", response_model=RefreshFundamentalsResponse)
def refresh_fundamentals(
    request: RefreshFundamentalsRequest,
    base_path: Annotated[Path, Depends(get_data_base_path)],
) -> RefreshFundamentalsResponse:
    """Refresh fundamentals for symbols not fetched today.

    This endpoint checks which symbols haven't been fetched today and
    fetches fresh data from Alpha Vantage API for those symbols only.

    Requires ALPHA_VANTAGE_API_KEY environment variable.

    Use this before calling POST /signals/fundamentals/historical to ensure
    data is fresh.
    """
    result = refresh_stale_fundamentals(
        symbols=request.symbols,
        base_path=base_path,
    )

    return RefreshFundamentalsResponse(
        refreshed=result.refreshed,
        skipped=result.skipped,
        failed=result.failed,
        api_status=ApiStatusResponse(
            calls_today=result.api_status.get("calls_today", 0),
            daily_limit=result.api_status.get("daily_limit", 25),
            remaining=result.api_status.get("remaining", 25),
        ),
    )
