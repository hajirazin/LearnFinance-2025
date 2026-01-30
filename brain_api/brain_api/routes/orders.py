"""Order generation endpoints for Alpaca paper trading.

Converts allocation weights into actionable limit orders.
"""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from brain_api.core.orders import (
    GenerateOrdersResult,
    PortfolioInput,
    PositionInput,
    generate_orders,
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

    cash: float = Field(..., ge=0, description="Cash balance in dollars")
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
        description="Algorithm name (e.g., 'ppo', 'sac', 'hrp')",
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
    type: str = Field(..., description="Order type ('limit')")
    limit_price: float = Field(..., gt=0, description="Limit price in dollars")
    time_in_force: str = Field(..., description="Time in force ('day')")


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


# ============================================================================
# Endpoint
# ============================================================================


@router.post("/generate", response_model=GenerateOrdersResponse)
def generate_orders_endpoint(request: GenerateOrdersRequest) -> GenerateOrdersResponse:
    """Generate orders to rebalance portfolio to target allocation.

    This endpoint converts target allocation weights into actionable limit orders
    that can be submitted directly to Alpaca. It handles:

    1. **Idempotent client_order_id generation**: Deterministic IDs based on
       run_id, attempt, symbol, and side prevent duplicate orders.

    2. **Minimum trade filtering**: Orders below $10 value are skipped.

    3. **Limit price calculation**: Adds 1% buffer (above market for buys,
       below for sells) to improve fill probability.

    4. **Fractional shares**: Quantities are calculated to 4 decimal places.

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
      "algorithm": "ppo"
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
          "type": "limit",
          "limit_price": 419.55,
          "time_in_force": "day"
        }
      ],
      "summary": {
        "buys": 1,
        "sells": 0,
        "total_buy_value": 1043.87,
        "total_sell_value": 0,
        "turnover_pct": 4.8,
        "skipped_small_orders": 0
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
        )
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
    )

    return GenerateOrdersResponse(
        orders=orders,
        summary=summary,
        prices_used=result.prices_used,
    )
