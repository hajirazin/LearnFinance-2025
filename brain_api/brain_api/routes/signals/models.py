"""Request and response models for signal endpoints."""

from pydantic import BaseModel, Field


# ============================================================================
# Configuration constants
# ============================================================================

MAX_SYMBOLS = 50
MAX_ARTICLES_PER_SYMBOL = 30
MAX_RETURN_TOP_K = 10
DEFAULT_MAX_ARTICLES = 30
DEFAULT_RETURN_TOP_K = 10
MAX_FUNDAMENTALS_SYMBOLS = 20
MAX_HISTORICAL_SENTIMENT_SYMBOLS = 20


# ============================================================================
# News sentiment models
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
# Historical news sentiment models
# ============================================================================


class HistoricalNewsSentimentRequest(BaseModel):
    """Request model for historical news sentiment endpoint (training via parquet)."""

    symbols: list[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_HISTORICAL_SENTIMENT_SYMBOLS,
        description=f"List of ticker symbols (1-{MAX_HISTORICAL_SENTIMENT_SYMBOLS})",
    )
    start_date: str = Field(
        ...,
        description="Start date for historical range (YYYY-MM-DD)",
    )
    end_date: str = Field(
        ...,
        description="End date for historical range (YYYY-MM-DD)",
    )


class SentimentDataPoint(BaseModel):
    """Historical sentiment data for a symbol on a specific date."""

    symbol: str
    date: str
    sentiment_score: float  # -1 to 1, 0.0 = neutral (default for missing)
    article_count: int | None  # None if neutral fallback (no data)
    p_pos_avg: float | None
    p_neg_avg: float | None


class HistoricalNewsSentimentResponse(BaseModel):
    """Response model for historical news sentiment endpoint."""

    start_date: str
    end_date: str
    data: list[SentimentDataPoint]


# ============================================================================
# Fundamentals models
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
    data: list[RatiosResponse]


