"""Request and response models for signal endpoints."""

from datetime import date

from pydantic import BaseModel, Field

# ============================================================================
# Configuration constants
# ============================================================================

MAX_SYMBOLS = 50
MAX_ARTICLES_PER_SYMBOL = 30
MAX_RETURN_TOP_K = 10
DEFAULT_MAX_ARTICLES = 30
DEFAULT_RETURN_TOP_K = 10
MAX_PRICE_SYMBOLS = 30
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
# Prices (closes) models -- SAC momentum_1w/4w/12_1
# ============================================================================

MIN_MOMENTUM_LOOKBACK_BARS = 253  # MOM_12_1_LOOKBACK_BARS(252) + 1 (P_t itself)


class ClosesRequest(BaseModel):
    """Request model for raw daily closes (SAC momentum signals)."""

    symbols: list[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_PRICE_SYMBOLS,
        description=f"List of ticker symbols (1-{MAX_PRICE_SYMBOLS})",
    )
    as_of_date: str = Field(
        ...,
        description="Decision date (YYYY-MM-DD); closes are as of/before this date",
    )
    lookback_bars: int = Field(
        MIN_MOMENTUM_LOOKBACK_BARS,
        ge=MIN_MOMENTUM_LOOKBACK_BARS,
        description=(
            "Trailing trading-day bars to return per symbol, oldest first. "
            "Defaults to momentum_12_1's requirement (skip 21 + lookback 252)."
        ),
    )


class ClosesResponse(BaseModel):
    """Point-in-time adjusted closes for SAC feature construction."""

    as_of_date: str
    adjusted_closes: dict[str, list[float]]
    provenance: dict[str, object]


class MarketHistoryRequest(BaseModel):
    """Request aligned SPY/VIX observations for causal HMM continuation."""

    start_date: date = Field(
        ...,
        description="First required post-training-cutoff market session",
    )
    as_of_date: date = Field(
        ...,
        description="Pre-open decision date; only earlier completed XNYS sessions",
    )


class MarketHistoryRow(BaseModel):
    """One aligned raw market observation row."""

    date: date
    spy_adjusted_close: float = Field(..., gt=0)
    vix_close: float = Field(..., gt=0)


class MarketHistoryResponse(BaseModel):
    """Gap-sensitive raw market history consumed by Brain's HMM filter."""

    start_date: date
    as_of_date: date
    rows: list[MarketHistoryRow]
    provenance: dict[str, object]
