"""Signal route handlers."""

from datetime import date
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends

from brain_api.core.fundamentals import (
    FundamentalsFetcher,
    compute_ratios,
    load_raw_response,
    parse_quarterly_statements,
)
from brain_api.core.news_sentiment import (
    NewsFetcher,
    SentimentScorer,
    process_news_sentiment,
)
from brain_api.routes.signals.dependencies import (
    get_data_base_path,
    get_fundamentals_fetcher,
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
            results.append(CurrentRatiosResponse(
                symbol=symbol,
                ratios=ratios,
                error=None if ratios else "No data available",
            ))
        except Exception as e:
            results.append(CurrentRatiosResponse(
                symbol=symbol,
                ratios=None,
                error=str(e),
            ))

    return FundamentalsResponse(
        as_of_date=as_of,
        per_symbol=results,
    )


@router.post("/fundamentals/historical", response_model=HistoricalFundamentalsResponse)
def get_historical_fundamentals(
    request: HistoricalFundamentalsRequest,
    fetcher: Annotated[FundamentalsFetcher, Depends(get_fundamentals_fetcher)],
) -> HistoricalFundamentalsResponse:
    """Get HISTORICAL fundamental ratios for training (date range).

    Returns n symbols × m quarterly periods as a flat list. Each entry represents
    the financial ratios that would have been available at that point in time.

    Data source: Alpha Vantage (INCOME_STATEMENT + BALANCE_SHEET)
    """
    try:
        all_ratios: list[RatiosResponse] = []

        for symbol in request.symbols:
            try:
                # Fetch data (uses cache if available)
                fetcher.fetch_symbol(
                    symbol=symbol,
                    force_refresh=request.force_refresh,
                )

                income_data = load_raw_response(
                    fetcher.base_path, symbol, "income_statement"
                )
                balance_data = load_raw_response(
                    fetcher.base_path, symbol, "balance_sheet"
                )

                if income_data is None and balance_data is None:
                    continue

                # Parse statements
                income_stmts = []
                balance_stmts = []

                if income_data:
                    income_stmts = parse_quarterly_statements(
                        symbol, "income_statement", income_data
                    )
                if balance_data:
                    balance_stmts = parse_quarterly_statements(
                        symbol, "balance_sheet", balance_data
                    )

                # Get all unique fiscal dates in range
                fiscal_dates = set()
                for stmt in income_stmts:
                    if request.start_date <= stmt.fiscal_date_ending <= request.end_date:
                        fiscal_dates.add(stmt.fiscal_date_ending)
                for stmt in balance_stmts:
                    if request.start_date <= stmt.fiscal_date_ending <= request.end_date:
                        fiscal_dates.add(stmt.fiscal_date_ending)

                # Compute ratios for each fiscal date
                for fiscal_date in sorted(fiscal_dates):
                    income_stmt = next(
                        (s for s in income_stmts if s.fiscal_date_ending == fiscal_date),
                        None,
                    )
                    balance_stmt = next(
                        (s for s in balance_stmts if s.fiscal_date_ending == fiscal_date),
                        None,
                    )

                    ratios = compute_ratios(income_stmt, balance_stmt)
                    if ratios:
                        all_ratios.append(RatiosResponse(
                            symbol=ratios.symbol,
                            as_of_date=ratios.as_of_date,
                            gross_margin=ratios.gross_margin,
                            operating_margin=ratios.operating_margin,
                            net_margin=ratios.net_margin,
                            current_ratio=ratios.current_ratio,
                            debt_to_equity=ratios.debt_to_equity,
                        ))

            except Exception:
                continue

        # Get API status
        api_status = fetcher.get_api_status()

        return HistoricalFundamentalsResponse(
            start_date=request.start_date,
            end_date=request.end_date,
            api_status=ApiStatusResponse(**api_status),
            data=all_ratios,
        )
    finally:
        fetcher.close()


