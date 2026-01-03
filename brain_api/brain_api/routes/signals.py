"""Signal endpoints for feature extraction (news sentiment, fundamentals, etc.)."""

import os
from datetime import date
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from brain_api.core.fundamentals import (
    FundamentalRatios,
    FundamentalsFetcher,
)
from brain_api.core.news_sentiment import (
    FinBERTScorer,
    NewsFetcher,
    NewsSentimentResult,
    SentimentScorer,
    SymbolSentiment,
    YFinanceNewsFetcher,
    process_news_sentiment,
)

router = APIRouter()


# ============================================================================
# Configuration constants
# ============================================================================

MAX_SYMBOLS = 50
MAX_ARTICLES_PER_SYMBOL = 30
MAX_RETURN_TOP_K = 10

DEFAULT_MAX_ARTICLES = 30
DEFAULT_RETURN_TOP_K = 10


# ============================================================================
# Request / Response models
# ============================================================================


class NewsSignalRequest(BaseModel):
    """Request model for news sentiment endpoint."""

    symbols: list[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_SYMBOLS,
        description=f"List of ticker symbols (1-{MAX_SYMBOLS})",
    )
    as_of_date: str | None = Field(
        None,
        description="Reference date (YYYY-MM-DD). Defaults to today.",
    )
    max_articles_per_symbol: int = Field(
        DEFAULT_MAX_ARTICLES,
        ge=1,
        le=MAX_ARTICLES_PER_SYMBOL,
        description=f"Max articles to fetch per symbol (1-{MAX_ARTICLES_PER_SYMBOL})",
    )
    return_top_k: int = Field(
        DEFAULT_RETURN_TOP_K,
        ge=1,
        le=MAX_RETURN_TOP_K,
        description=f"Number of top articles to return per symbol (1-{MAX_RETURN_TOP_K})",
    )
    run_id: str | None = Field(
        None,
        description="Run identifier. Defaults to paper:<as_of_date>",
    )
    attempt: int = Field(
        1,
        ge=1,
        description="Attempt number for the run",
    )


class ArticleResponse(BaseModel):
    """Article in the API response (subset of stored data)."""

    title: str
    publisher: str
    link: str
    published: str | None
    finbert_label: str
    finbert_p_pos: float
    finbert_p_neg: float
    finbert_p_neu: float
    article_score: float


class SymbolSentimentResponse(BaseModel):
    """Per-symbol sentiment in the API response."""

    symbol: str
    article_count_fetched: int
    article_count_used: int
    sentiment_score: float
    insufficient_news: bool
    top_k_articles: list[ArticleResponse]


class NewsSignalResponse(BaseModel):
    """Response model for news sentiment endpoint."""

    run_id: str
    attempt: int
    as_of_date: str
    from_cache: bool
    per_symbol: list[SymbolSentimentResponse]


# ============================================================================
# Dependency injection for testability
# ============================================================================


def get_news_fetcher() -> NewsFetcher:
    """Get the news fetcher implementation."""
    return YFinanceNewsFetcher()


def get_sentiment_scorer() -> SentimentScorer:
    """Get the sentiment scorer implementation."""
    return FinBERTScorer()


def get_data_base_path() -> Path:
    """Get the base path for data storage."""
    return Path("data")


# ============================================================================
# Helper functions
# ============================================================================


def _symbol_to_response(
    sentiment: SymbolSentiment,
    return_top_k: int,
) -> SymbolSentimentResponse:
    """Convert internal SymbolSentiment to API response format.

    Selects top K articles by article_score (most positive first).
    """
    # Sort articles by article_score descending
    sorted_articles = sorted(
        sentiment.scored_articles,
        key=lambda a: a.finbert.article_score,
        reverse=True,
    )

    # Take top K
    top_articles = sorted_articles[:return_top_k]

    # Convert to response format
    article_responses = [
        ArticleResponse(
            title=sa.article.title,
            publisher=sa.article.publisher,
            link=sa.article.link,
            published=sa.article.published.isoformat() if sa.article.published else None,
            finbert_label=sa.finbert.label,
            finbert_p_pos=sa.finbert.p_pos,
            finbert_p_neg=sa.finbert.p_neg,
            finbert_p_neu=sa.finbert.p_neu,
            article_score=sa.finbert.article_score,
        )
        for sa in top_articles
    ]

    return SymbolSentimentResponse(
        symbol=sentiment.symbol,
        article_count_fetched=sentiment.article_count_fetched,
        article_count_used=sentiment.article_count_used,
        sentiment_score=sentiment.sentiment_score,
        insufficient_news=sentiment.insufficient_news,
        top_k_articles=article_responses,
    )


def _result_to_response(
    result: NewsSentimentResult,
    return_top_k: int,
) -> NewsSignalResponse:
    """Convert internal result to API response format."""
    return NewsSignalResponse(
        run_id=result.run_id,
        attempt=result.attempt,
        as_of_date=result.as_of_date,
        from_cache=result.from_cache,
        per_symbol=[
            _symbol_to_response(s, return_top_k) for s in result.per_symbol
        ],
    )


# ============================================================================
# Endpoint
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

    Args:
        request: NewsSignalRequest with symbols and parameters

    Returns:
        NewsSignalResponse with per-symbol sentiment and top articles
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

    # Convert to response format
    return _result_to_response(result, return_top_k)


# ============================================================================
# Fundamentals configuration
# ============================================================================

MAX_FUNDAMENTALS_SYMBOLS = 20  # Match universe size


# ============================================================================
# Fundamentals request / response models
# ============================================================================


class FundamentalsRequest(BaseModel):
    """Request model for current fundamentals endpoint (inference via yfinance)."""

    symbols: list[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_FUNDAMENTALS_SYMBOLS,
        description=f"List of ticker symbols (1-{MAX_FUNDAMENTALS_SYMBOLS})",
    )


class HistoricalFundamentalsRequest(BaseModel):
    """Request model for historical fundamentals endpoint (training via Alpha Vantage)."""

    symbols: list[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_FUNDAMENTALS_SYMBOLS,
        description=f"List of ticker symbols (1-{MAX_FUNDAMENTALS_SYMBOLS})",
    )
    start_date: str = Field(
        ...,
        description="Start date for historical range (YYYY-MM-DD)",
    )
    end_date: str = Field(
        ...,
        description="End date for historical range (YYYY-MM-DD)",
    )
    force_refresh: bool = Field(
        False,
        description="If True, ignore cache and re-fetch from API (uses API quota)",
    )


class RatiosResponse(BaseModel):
    """Financial ratios for a symbol at a point in time.
    
    5 core ratios for PPO:
    - Profitability: gross_margin, operating_margin, net_margin
    - Liquidity: current_ratio
    - Leverage: debt_to_equity
    """

    symbol: str
    as_of_date: str
    gross_margin: float | None
    operating_margin: float | None
    net_margin: float | None
    current_ratio: float | None
    debt_to_equity: float | None


class CurrentRatiosResponse(BaseModel):
    """Per-symbol current fundamentals response (from yfinance)."""

    symbol: str
    ratios: RatiosResponse | None
    error: str | None = None


class ApiStatusResponse(BaseModel):
    """API usage status (for Alpha Vantage historical endpoint)."""

    calls_today: int
    daily_limit: int
    remaining: int


class FundamentalsResponse(BaseModel):
    """Response model for current fundamentals endpoint (yfinance)."""

    as_of_date: str
    per_symbol: list[CurrentRatiosResponse]


class HistoricalFundamentalsResponse(BaseModel):
    """Response model for historical fundamentals endpoint (n symbols × m dates)."""

    start_date: str
    end_date: str
    api_status: ApiStatusResponse
    data: list[RatiosResponse]  # Flat list: n symbols × m quarterly periods


# ============================================================================
# Fundamentals dependency injection (for historical endpoint only)
# ============================================================================


def get_alpha_vantage_api_key() -> str:
    """Get Alpha Vantage API key from environment."""
    key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
    return key


def get_fundamentals_fetcher(
    api_key: Annotated[str, Depends(get_alpha_vantage_api_key)],
    base_path: Annotated[Path, Depends(get_data_base_path)],
) -> FundamentalsFetcher:
    """Get the fundamentals fetcher with injected dependencies."""
    return FundamentalsFetcher(
        api_key=api_key,
        base_path=base_path,
        daily_limit=25,  # Alpha Vantage free tier
    )


# ============================================================================
# yfinance current fundamentals helper
# ============================================================================


def _get_yfinance_ratios(symbol: str, as_of_date: str) -> RatiosResponse | None:
    """Fetch current fundamental ratios from yfinance.
    
    yfinance ticker.info contains pre-computed ratios from the latest filings.
    No rate limits, no caching needed.
    """
    import yfinance as yf
    
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        
        # yfinance provides these as decimals (e.g., 0.45 for 45%)
        gross_margin = info.get("grossMargins")
        operating_margin = info.get("operatingMargins")
        net_margin = info.get("profitMargins")
        current_ratio = info.get("currentRatio")
        debt_to_equity = info.get("debtToEquity")
        
        # debtToEquity from yfinance is as percentage (e.g., 150 for 1.5x)
        # Normalize to ratio
        if debt_to_equity is not None and debt_to_equity > 10:
            debt_to_equity = debt_to_equity / 100
        
        return RatiosResponse(
            symbol=symbol,
            as_of_date=as_of_date,
            gross_margin=round(gross_margin, 4) if gross_margin else None,
            operating_margin=round(operating_margin, 4) if operating_margin else None,
            net_margin=round(net_margin, 4) if net_margin else None,
            current_ratio=round(current_ratio, 4) if current_ratio else None,
            debt_to_equity=round(debt_to_equity, 4) if debt_to_equity else None,
        )
    except Exception:
        return None


# ============================================================================
# Historical fundamentals helper
# ============================================================================


def _ratios_to_response(
    ratios: FundamentalRatios | None,
) -> RatiosResponse | None:
    """Convert internal FundamentalRatios to API response."""
    if ratios is None:
        return None
    return RatiosResponse(
        symbol=ratios.symbol,
        as_of_date=ratios.as_of_date,
        gross_margin=ratios.gross_margin,
        operating_margin=ratios.operating_margin,
        net_margin=ratios.net_margin,
        current_ratio=ratios.current_ratio,
        debt_to_equity=ratios.debt_to_equity,
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

    Returns:
        FundamentalsResponse with per-symbol current ratios
    """
    as_of = date.today().isoformat()
    results: list[CurrentRatiosResponse] = []

    for symbol in request.symbols:
        try:
            ratios = _get_yfinance_ratios(symbol, as_of)
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
    the financial ratios that would have been available at that point in time
    (point-in-time correctness for avoiding look-ahead bias in training).

    Data source: Alpha Vantage (INCOME_STATEMENT + BALANCE_SHEET)

    Rate limits:
    - Free tier: 25 API calls/day
    - Each symbol requires 2 calls (income + balance) on first fetch
    - Cached data is reused until force_refresh is True

    Returns:
        HistoricalFundamentalsResponse with flat list of ratios (n × m)
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

                # Get all quarterly periods within the date range
                from brain_api.core.fundamentals import (
                    load_raw_response,
                    parse_quarterly_statements,
                    compute_ratios,
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
                    # Find matching statements
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
                # Skip symbols with errors, continue with others
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


