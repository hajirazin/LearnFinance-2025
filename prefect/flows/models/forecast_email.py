"""Models for weekly forecast email flow.

These models represent the responses from brain_api endpoints
used in the weekly forecast email workflow.
"""

from pydantic import BaseModel

# ============================================================================
# Alpaca Endpoint Models
# ============================================================================


class PositionModel(BaseModel):
    """A single position in the portfolio."""

    symbol: str
    qty: float
    market_value: float


class AlpacaPortfolioResponse(BaseModel):
    """Response from GET /alpaca/portfolio."""

    cash: float
    positions: list[PositionModel]
    open_orders_count: int


class OrderSubmitResult(BaseModel):
    """Result of a single order submission."""

    id: str | None = None
    client_order_id: str
    symbol: str
    status: str
    error: str | None = None


class SubmitOrdersResponse(BaseModel):
    """Response from POST /alpaca/submit-orders."""

    account: str
    orders_submitted: int
    orders_failed: int
    skipped: bool = False
    results: list[OrderSubmitResult]


class OrderHistoryItem(BaseModel):
    """A single order from Alpaca order history."""

    id: str
    client_order_id: str
    symbol: str
    side: str
    status: str
    filled_qty: str | None = None
    filled_avg_price: str | None = None


# ============================================================================
# Inference Endpoint Models
# ============================================================================


class LSTMPrediction(BaseModel):
    """A single LSTM prediction."""

    symbol: str
    predicted_weekly_return_pct: float
    direction: str
    has_enough_history: bool


class LSTMInferenceResponse(BaseModel):
    """Response from POST /inference/lstm."""

    predictions: list[LSTMPrediction]
    model_version: str
    as_of_date: str
    target_week_start: str | None = None
    target_week_end: str | None = None


class PatchTSTPrediction(BaseModel):
    """A single PatchTST prediction."""

    symbol: str
    predicted_weekly_return_pct: float
    direction: str
    has_enough_history: bool


class PatchTSTInferenceResponse(BaseModel):
    """Response from POST /inference/patchtst."""

    predictions: list[PatchTSTPrediction]
    model_version: str
    as_of_date: str
    signals_used: list[str] = []
    target_week_start: str | None = None
    target_week_end: str | None = None


class PPOInferenceResponse(BaseModel):
    """Response from POST /inference/ppo."""

    target_weights: dict[str, float]
    turnover: float
    model_version: str
    target_week_start: str | None = None
    target_week_end: str | None = None


class SACInferenceResponse(BaseModel):
    """Response from POST /inference/sac."""

    target_weights: dict[str, float]
    turnover: float
    model_version: str
    target_week_start: str | None = None
    target_week_end: str | None = None


class HRPAllocationResponse(BaseModel):
    """Response from POST /allocation/hrp."""

    percentage_weights: dict[str, float]
    symbols_used: int
    symbols_excluded: list[str] = []
    as_of_date: str


# ============================================================================
# Signals Endpoint Models
# ============================================================================


class NewsArticle(BaseModel):
    """A news article with sentiment."""

    title: str
    url: str | None = None
    sentiment_score: float | None = None


class PerSymbolNews(BaseModel):
    """News sentiment for a single symbol."""

    symbol: str
    sentiment_score: float
    article_count: int = 0
    top_k_articles: list[NewsArticle] = []


class NewsSignalResponse(BaseModel):
    """Response from POST /signals/news."""

    per_symbol: list[PerSymbolNews]
    as_of_date: str


class FundamentalRatios(BaseModel):
    """Financial ratios for a stock."""

    gross_margin: float | None = None
    operating_margin: float | None = None
    net_margin: float | None = None
    current_ratio: float | None = None
    debt_to_equity: float | None = None


class PerSymbolFundamentals(BaseModel):
    """Fundamentals for a single symbol."""

    symbol: str
    ratios: FundamentalRatios | None = None


class FundamentalsResponse(BaseModel):
    """Response from POST /signals/fundamentals."""

    per_symbol: list[PerSymbolFundamentals]
    as_of_date: str


# ============================================================================
# Orders Endpoint Models
# ============================================================================


class OrderModel(BaseModel):
    """A single order to submit to Alpaca."""

    client_order_id: str
    symbol: str
    side: str
    qty: float
    type: str
    limit_price: float
    time_in_force: str


class OrderSummary(BaseModel):
    """Summary of generated orders."""

    buys: int
    sells: int
    total_buy_value: float
    total_sell_value: float
    turnover_pct: float
    skipped_small_orders: int


class GenerateOrdersResponse(BaseModel):
    """Response from POST /orders/generate."""

    orders: list[OrderModel]
    summary: OrderSummary
    prices_used: dict[str, float]


# ============================================================================
# Experience Endpoint Models
# ============================================================================


class StoreExperienceResponse(BaseModel):
    """Response from POST /experience/store."""

    record_id: str
    stored: bool
    model_type: str


class UpdateExecutionResponse(BaseModel):
    """Response from POST /experience/update-execution."""

    run_id: str
    updated: bool
    orders_filled: int
    orders_partial: int
    orders_expired: int


# ============================================================================
# LLM and Email Endpoint Models
# ============================================================================


class WeeklySummaryResponse(BaseModel):
    """Response from POST /llm/weekly-summary."""

    summary: dict[str, str]
    provider: str
    model_used: str
    tokens_used: int | None = None


class WeeklyReportEmailResponse(BaseModel):
    """Response from POST /email/weekly-report."""

    is_success: bool
    subject: str
    body: str


# ============================================================================
# Skipped Placeholder Models
# ============================================================================


class SkippedAllocation(BaseModel):
    """Placeholder for skipped allocator response."""

    skipped: bool = True
    algorithm: str
    target_weights: dict[str, float] = {}
    reason: str = "Open orders exist"
    turnover: float = 0.0
    model_version: str = "skipped"


class SkippedOrdersResponse(BaseModel):
    """Placeholder for skipped order generation response."""

    skipped: bool = True
    algorithm: str
    orders: list = []
    reason: str = "Open orders exist"


class SkippedSubmitResponse(BaseModel):
    """Placeholder for skipped order submission response."""

    account: str
    orders_submitted: int = 0
    orders_failed: int = 0
    skipped: bool = True
    results: list = []
