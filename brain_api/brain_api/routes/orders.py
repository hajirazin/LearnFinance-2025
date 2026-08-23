"""Order generation endpoints for Alpaca paper trading.

Converts allocation weights into actionable limit orders.
"""

import logging
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from brain_api.core.orders import (
    GenerateOrdersResult,
    OrderPriceError,
    PortfolioInput,
    PositionInput,
    convert_weights_to_whole_shares,
    generate_orders,
)
from brain_api.core.orders import (
    PaperAllocationResult as CorePaperAllocationResult,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# Request / Response models
# ============================================================================


class PositionModel(BaseModel):
    """A single position in the portfolio."""

    symbol: str = Field(..., description="Stock symbol")
    qty: float = Field(..., ge=0, description="Quantity of shares")
    market_value: float = Field(
        ..., ge=0, description="Current market value in dollars"
    )


class PortfolioModel(BaseModel):
    """Current portfolio state from Alpaca."""

    # Note: Cash can be slightly negative due to pending settlements
    cash: float = Field(..., description="Cash balance in dollars")
    currency: str = Field(default="USD", description="Currency of the cash balance")
    positions: list[PositionModel] = Field(
        default_factory=list, description="List of current positions"
    )


class GenerateOrdersRequest(BaseModel):
    """Request model for order generation endpoint."""

    target_weights: dict[str, float] = Field(
        ...,
        description="Target allocation weights (symbol -> weight, including CASH). Weights should sum to 1.0",
    )
    portfolio: PortfolioModel = Field(
        ...,
        description="Current portfolio state (cash + positions)",
    )
    run_id: str = Field(
        ...,
        description="Run identifier (e.g., 'paper:2026-01-20')",
    )
    attempt: int = Field(
        ...,
        ge=1,
        description="Attempt number (1, 2, 3, ...)",
    )
    algorithm: str = Field(
        ...,
        description="Algorithm name (e.g., 'sac', 'hrp')",
    )
    order_side: Literal["all", "sell", "buy"] = Field(
        default="all",
        description="Generate all rebalance legs or only one execution phase.",
    )


class OrderModel(BaseModel):
    """A single order to submit to Alpaca."""

    client_order_id: str = Field(
        ...,
        description="Deterministic order ID for idempotency",
    )
    symbol: str = Field(..., description="Stock symbol")
    side: str = Field(..., description="'buy' or 'sell'")
    qty: float = Field(
        ..., gt=0, description="Quantity of shares (supports fractional)"
    )
    type: str = Field(default="market", description="Order type")
    limit_price: float | None = Field(
        default=None, gt=0, description="Limit price (only for limit orders)"
    )
    time_in_force: str = Field(..., description="Time in force ('day')")
    currency: str = Field(default="USD", description="Currency of the order")
    # Display-only stop-loss reference. Computed in core/orders.py via
    # core/stop_loss.compute_stop_loss using ATR(14)*2 clamped to
    # [5%, 10%] of entry. Sells carry "sell_no_stop"; buys without ATR
    # carry "atr_unavailable" (no flat-percent fallback per AGENTS.md
    # rule #1).
    stop_loss_price: float | None = Field(
        default=None,
        description="Stop-loss reference price; None on sells / no ATR",
    )
    stop_loss_distance_pct: float | None = Field(
        default=None,
        description="Stop distance as a fraction of entry price",
    )
    stop_loss_reason: str = Field(
        default="atr_unavailable",
        description="'atr14' / 'atr_unavailable' / 'sell_no_stop'",
    )
    cash_qty: float | None = Field(
        default=None,
        description="Monetary value for fractional execution",
    )


class OrderSummaryModel(BaseModel):
    """Summary of generated orders."""

    buys: int = Field(..., ge=0, description="Number of buy orders")
    sells: int = Field(..., ge=0, description="Number of sell orders")
    total_buy_value: float = Field(..., ge=0, description="Total value of buy orders")
    total_sell_value: float = Field(..., ge=0, description="Total value of sell orders")
    turnover_pct: float = Field(..., ge=0, description="Portfolio turnover percentage")
    skipped_small_orders: int = Field(
        ..., ge=0, description="Orders skipped due to small value"
    )
    skipped_below_threshold: int = Field(
        ...,
        ge=0,
        description="Legs skipped: absolute weight change below min rebalance threshold",
    )


class GenerateOrdersResponse(BaseModel):
    """Response model for order generation endpoint."""

    orders: list[OrderModel] = Field(
        ...,
        description="List of orders to submit to Alpaca",
    )
    summary: OrderSummaryModel = Field(
        ...,
        description="Summary of generated orders",
    )
    prices_used: dict[str, float] = Field(
        ...,
        description="Prices used for calculations (symbol -> price)",
    )
    atr_used: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "ATR(14) per symbol (Wilder smoothing). Used by the email "
            "layer to render a stop-loss reference next to each buy. "
            "Symbols with insufficient history are absent (no silent "
            "fallback)."
        ),
    )


# ============================================================================
# Endpoint
# ============================================================================


@router.post("/generate", response_model=GenerateOrdersResponse)
def generate_orders_endpoint(request: GenerateOrdersRequest) -> GenerateOrdersResponse:
    # Log incoming request for debugging
    logger.debug(
        f"[Orders] Request received: algorithm={request.algorithm}, "
        f"run_id={request.run_id}, attempt={request.attempt}, "
        f"cash={request.portfolio.cash}, positions={len(request.portfolio.positions)}, "
        f"target_weights_count={len(request.target_weights)}"
    )
    """Generate orders to rebalance portfolio to target allocation.

    This endpoint converts target allocation weights into actionable market orders
    that can be submitted directly to Alpaca. It handles:

    1. **Idempotent client_order_id generation**: Deterministic IDs based on
       run_id, attempt, symbol, and side prevent duplicate orders.

    2. **Minimum weight change**: Legs with absolute weight delta below 1% of NAV
       are skipped, except full exits (target weight 0 with an open position).

    3. **Minimum trade filtering**: Orders below $10 value are skipped,
       except full-exit sells (target weight 0 with an open position).

    4. **Buy funding cap**: If total buy notional exceeds cash plus expected
       proceeds from generated sells, buy quantities are scaled down proportionally
       (or buys dropped if no buying power).

    5. **Sell qty cap**: Sell quantity is capped at position quantity to avoid
       requesting more shares than held. Full exits sell the entire broker qty
       (not notional / Yahoo price).

    6. **Fractional shares**: Partial rebalance quantities are rounded to 4
       decimal places. Full-exit sells keep the broker lot size unrounded.

    The generated orders can be submitted directly to Alpaca's POST /v2/orders
    endpoint. Alpaca will reject orders with duplicate client_order_ids,
    providing an additional safety layer.

    Args:
        request: GenerateOrdersRequest with target_weights, portfolio, run_id,
                 attempt, and algorithm

    Returns:
        GenerateOrdersResponse with orders, summary, and prices used

    Raises:
        HTTPException 400: if target_weights is empty or portfolio value is 0

    Example request:
    ```json
    {
      "target_weights": {"AAPL": 0.15, "MSFT": 0.10, "CASH": 0.75},
      "portfolio": {
        "cash": 10000,
        "positions": [{"symbol": "AAPL", "qty": 5, "market_value": 850}]
      },
      "run_id": "paper:2026-01-20",
      "attempt": 1,
      "algorithm": "sac",
      "order_side": "sell"
    }
    ```

    Example response:
    ```json
    {
      "orders": [
        {
          "client_order_id": "paper:2026-01-20:attempt-1:MSFT:BUY",
          "symbol": "MSFT",
          "side": "buy",
          "qty": 2.5,
          "type": "market",
          "time_in_force": "day"
        }
      ],
      "summary": {
        "buys": 1,
        "sells": 0,
        "total_buy_value": 1043.87,
        "total_sell_value": 0,
        "turnover_pct": 4.8,
        "skipped_small_orders": 0,
        "skipped_below_threshold": 0
      },
      "prices_used": {"AAPL": 170.00, "MSFT": 415.00}
    }
    ```
    """
    # Validate target_weights
    if not request.target_weights:
        raise HTTPException(
            status_code=400,
            detail="target_weights cannot be empty",
        )

    # Validate portfolio currency
    if request.portfolio.currency != "USD":
        raise HTTPException(
            status_code=400,
            detail=f"Portfolio currency must be USD for order generation, got {request.portfolio.currency}",
        )

    # Convert request models to core types
    positions = [
        PositionInput(
            symbol=p.symbol,
            qty=p.qty,
            market_value=p.market_value,
        )
        for p in request.portfolio.positions
    ]

    portfolio = PortfolioInput(
        cash=request.portfolio.cash,
        positions=positions,
    )

    # Validate portfolio has value
    if portfolio.total_value <= 0:
        raise HTTPException(
            status_code=400,
            detail="Portfolio total value must be greater than 0",
        )

    # Log request
    logger.info(
        f"[Orders] Generating orders for {request.algorithm}, "
        f"run_id={request.run_id}, attempt={request.attempt}, "
        f"portfolio_value=${portfolio.total_value:.2f}"
    )

    # Generate orders
    try:
        result: GenerateOrdersResult = generate_orders(
            target_weights=request.target_weights,
            portfolio=portfolio,
            run_id=request.run_id,
            attempt=request.attempt,
            algorithm=request.algorithm,
            order_side=request.order_side,
        )
    except OrderPriceError as e:
        raise HTTPException(status_code=422, detail=str(e)) from None
    except Exception as e:
        logger.error(f"[Orders] Order generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Order generation failed: {e!s}",
        ) from None

    # Convert to response models
    orders = [
        OrderModel(
            client_order_id=o.client_order_id,
            symbol=o.symbol,
            side=o.side,
            qty=o.qty,
            type=o.order_type,
            limit_price=o.limit_price,
            time_in_force=o.time_in_force,
            currency="USD",
            stop_loss_price=o.stop_loss_price,
            stop_loss_distance_pct=o.stop_loss_distance_pct,
            stop_loss_reason=o.stop_loss_reason,
            cash_qty=o.cash_qty,
        )
        for o in result.orders
    ]

    summary = OrderSummaryModel(
        buys=result.summary.buys,
        sells=result.summary.sells,
        total_buy_value=result.summary.total_buy_value,
        total_sell_value=result.summary.total_sell_value,
        turnover_pct=result.summary.turnover_pct,
        skipped_small_orders=result.summary.skipped_small_orders,
        skipped_below_threshold=result.summary.skipped_below_threshold,
    )

    return GenerateOrdersResponse(
        orders=orders,
        summary=summary,
        prices_used=result.prices_used,
        atr_used=result.atr_used,
    )


# ============================================================================
# Paper Allocation Models (whole shares, no order submission)
# ============================================================================


class AllocationDetailModel(BaseModel):
    """A single row in the paper-allocation table.

    ``stop_loss_*`` fields are display-only ATR references so India
    Stage 2 email tables can render the same stop column as the US
    order table. See ``brain_api.core.stop_loss``.
    """

    symbol: str = Field(..., description="Stock symbol")
    weight_pct: float = Field(..., description="Target weight percentage (0..100)")
    price: float = Field(..., gt=0, description="Current market price")
    whole_shares: int = Field(
        ..., ge=0, description="Whole shares (floored to integer)"
    )
    trade_value: float = Field(..., ge=0, description="Notional value of shares held")
    stop_loss_price: float | None = Field(
        None, description="ATR-based stop-loss price (display-only)"
    )
    stop_loss_distance_pct: float | None = Field(
        None, description="Stop distance as a fraction of entry price"
    )
    stop_loss_reason: str = Field(
        "atr_unavailable",
        description="atr14 | atr_unavailable (paper rows are never sells)",
    )


class PaperAllocationRequest(BaseModel):
    """Request model for the paper-only weight-to-shares endpoint.

    India is paper-only with no broker, so it has no portfolio to
    reference. The caller supplies the target weights and a notional
    NAV; the endpoint looks up current prices via yfinance and converts
    each weight to a whole-share quantity (floored, no fractions).
    """

    percentage_weights: dict[str, float] = Field(
        ...,
        description="Target allocation weights (symbol -> pct, where 100 = 100%)",
    )
    total_nav: float = Field(
        ..., gt=0, description="Notional portfolio NAV in local currency (e.g. INR)"
    )


class PaperAllocationResponse(BaseModel):
    """Response model for POST /orders/paper-allocation."""

    details: list[AllocationDetailModel] = Field(
        ..., description="Per-symbol allocation details sorted by weight descending"
    )
    total_nav: float = Field(..., description="Notional NAV used for conversion")
    prices_used: dict[str, float] = Field(
        ..., description="Prices used for conversion (symbol -> price)"
    )
    total_allocated_pct: float = Field(
        ..., description="Sum of allocated weights (should round to ~100%)"
    )


# ============================================================================
# Paper Allocation Endpoint
# ============================================================================


@router.post("/paper-allocation", response_model=PaperAllocationResponse)
def paper_allocation_endpoint(
    request: PaperAllocationRequest,
) -> PaperAllocationResponse:
    """Convert percentage weights to whole shares (paper-only, no order submission).

    India uses this endpoint to show what a theoretical portfolio would
    look like in whole shares at current market prices. There is no
    portfolio input, no broker interaction, and no order generation —
    just a lookup of current prices and integer-floor share math.

    The response is purely informational and is rendered in the India
    email report alongside the Stage 2 HRP allocation table.
    """
    if not request.percentage_weights:
        raise HTTPException(
            status_code=400,
            detail="percentage_weights cannot be empty",
        )

    try:
        result: CorePaperAllocationResult = convert_weights_to_whole_shares(
            percentage_weights=request.percentage_weights,
            total_nav=request.total_nav,
        )
    except Exception as e:
        logger.error(f"[PaperAllocation] Conversion failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Weight-to-shares conversion failed: {e!s}",
        ) from None

    details = [
        AllocationDetailModel(
            symbol=d.symbol,
            weight_pct=d.weight_pct,
            price=d.price,
            whole_shares=d.whole_shares,
            trade_value=d.trade_value,
            stop_loss_price=d.stop_loss_price,
            stop_loss_distance_pct=d.stop_loss_distance_pct,
            stop_loss_reason=d.stop_loss_reason,
        )
        for d in result.details
    ]

    return PaperAllocationResponse(
        details=details,
        total_nav=result.total_nav,
        prices_used=result.prices_used,
        total_allocated_pct=result.total_allocated_pct,
    )
