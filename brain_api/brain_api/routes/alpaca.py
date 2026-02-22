"""Alpaca paper trading endpoints.

Provides endpoints to interact with Alpaca paper trading accounts:
- GET /alpaca/portfolio: Fetch account, positions, and open orders count
- POST /alpaca/submit-orders: Submit an array of orders
- GET /alpaca/order-history: Fetch order history for a date range
"""

import logging
import os
from enum import Enum

import httpx
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()

# Alpaca Paper Trading API base URL
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

# Timeout settings for Alpaca API calls
ALPACA_TIMEOUT = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)


# ============================================================================
# Enums
# ============================================================================


class AlpacaAccount(str, Enum):
    """Supported Alpaca paper trading accounts."""

    PPO = "ppo"
    SAC = "sac"
    HRP = "hrp"


# ============================================================================
# Request / Response Models
# ============================================================================


class PositionResponse(BaseModel):
    """A single position in the portfolio."""

    symbol: str = Field(..., description="Stock symbol")
    qty: float = Field(..., description="Quantity of shares")
    market_value: float = Field(..., description="Current market value in dollars")


class PortfolioResponse(BaseModel):
    """Portfolio data from Alpaca account."""

    cash: float = Field(..., description="Cash balance in dollars")
    positions: list[PositionResponse] = Field(
        default_factory=list, description="List of current positions"
    )
    open_orders_count: int = Field(
        ..., description="Number of open orders (used for skip logic)"
    )


class OrderToSubmit(BaseModel):
    """A single order to submit to Alpaca."""

    symbol: str = Field(..., description="Stock symbol")
    qty: float = Field(..., gt=0, description="Quantity of shares")
    side: str = Field(..., description="'buy' or 'sell'")
    type: str = Field(default="limit", description="Order type")
    time_in_force: str = Field(default="day", description="Time in force")
    limit_price: float = Field(..., gt=0, description="Limit price in dollars")
    client_order_id: str = Field(..., description="Idempotent order ID")


class SubmitOrdersRequest(BaseModel):
    """Request model for submitting multiple orders."""

    account: AlpacaAccount = Field(..., description="Trading account (ppo, sac, hrp)")
    orders: list[OrderToSubmit] = Field(
        default_factory=list, description="Orders to submit"
    )


class OrderSubmitResult(BaseModel):
    """Result of a single order submission."""

    id: str | None = Field(None, description="Alpaca order ID (if successful)")
    client_order_id: str = Field(..., description="Client order ID")
    symbol: str = Field(..., description="Stock symbol")
    status: str = Field(..., description="Order status or error")
    error: str | None = Field(None, description="Error message if failed")


class SubmitOrdersResponse(BaseModel):
    """Response model for order submission."""

    account: str = Field(..., description="Trading account")
    orders_submitted: int = Field(..., ge=0, description="Number of orders submitted")
    orders_failed: int = Field(..., ge=0, description="Number of orders failed")
    skipped: bool = Field(default=False, description="Whether submission was skipped")
    results: list[OrderSubmitResult] = Field(
        default_factory=list, description="Individual order results"
    )


class OrderHistoryItem(BaseModel):
    """A single order from Alpaca order history."""

    id: str = Field(..., description="Alpaca order ID")
    client_order_id: str = Field(..., description="Client order ID")
    symbol: str = Field(..., description="Stock symbol")
    side: str = Field(..., description="'buy' or 'sell'")
    status: str = Field(..., description="Order status (filled, canceled, etc.)")
    filled_qty: str | None = Field(None, description="Filled quantity")
    filled_avg_price: str | None = Field(None, description="Average fill price")


# ============================================================================
# Helper Functions
# ============================================================================


def get_alpaca_credentials(account: AlpacaAccount) -> tuple[str, str]:
    """Get Alpaca API credentials for a specific account.

    Args:
        account: The trading account (ppo, sac, hrp)

    Returns:
        Tuple of (api_key, api_secret)

    Raises:
        HTTPException: If credentials are not configured
    """
    account_upper = account.value.upper()
    key_var = f"ALPACA_{account_upper}_KEY"
    secret_var = f"ALPACA_{account_upper}_SECRET"

    api_key = os.environ.get(key_var)
    api_secret = os.environ.get(secret_var)

    if not api_key or not api_secret:
        logger.error(f"Missing Alpaca credentials for account {account.value}")
        raise HTTPException(
            status_code=500,
            detail=f"Alpaca credentials not configured for account {account.value}. "
            f"Set {key_var} and {secret_var} environment variables.",
        )

    return api_key, api_secret


def get_alpaca_headers(account: AlpacaAccount) -> dict[str, str]:
    """Get HTTP headers for Alpaca API authentication."""
    api_key, api_secret = get_alpaca_credentials(account)
    return {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
    }


# ============================================================================
# Endpoints
# ============================================================================


@router.get("/portfolio", response_model=PortfolioResponse)
def get_portfolio(
    account: AlpacaAccount = Query(..., description="Trading account (ppo, sac, hrp)"),
) -> PortfolioResponse:
    """Get portfolio data for a specific Alpaca account.

    Fetches account balance, positions, and open orders count from Alpaca.
    The open_orders_count is used to determine if an algorithm should be
    skipped (if > 0, there are pending orders from a previous run).

    Args:
        account: The trading account (ppo, sac, hrp)

    Returns:
        PortfolioResponse with cash, positions, and open_orders_count

    Raises:
        HTTPException 500: If credentials are missing
        HTTPException 503: If Alpaca API is unavailable
    """
    logger.info(f"Fetching portfolio for account {account.value}")
    headers = get_alpaca_headers(account)

    try:
        with httpx.Client(
            base_url=ALPACA_BASE_URL, headers=headers, timeout=ALPACA_TIMEOUT
        ) as client:
            # Fetch account data
            account_response = client.get("/v2/account")
            account_response.raise_for_status()
            account_data = account_response.json()

            # Fetch positions
            positions_response = client.get("/v2/positions")
            positions_response.raise_for_status()
            positions_data = positions_response.json()

            # Fetch open orders count
            orders_response = client.get("/v2/orders", params={"status": "open"})
            orders_response.raise_for_status()
            orders_data = orders_response.json()

    except httpx.TimeoutException as e:
        logger.error(f"Alpaca API timeout for account {account.value}: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Alpaca API timeout for account {account.value}",
        ) from e
    except httpx.HTTPStatusError as e:
        logger.error(f"Alpaca API error for account {account.value}: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Alpaca API error: {e.response.status_code}",
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error fetching portfolio: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Failed to fetch portfolio: {e!s}",
        ) from e

    # Normalize positions
    positions = [
        PositionResponse(
            symbol=p["symbol"],
            qty=float(p["qty"]),
            market_value=float(p["market_value"]),
        )
        for p in positions_data
    ]

    return PortfolioResponse(
        cash=float(account_data["cash"]),
        positions=positions,
        open_orders_count=len(orders_data),
    )


@router.post("/submit-orders", response_model=SubmitOrdersResponse)
def submit_orders(request: SubmitOrdersRequest) -> SubmitOrdersResponse:
    """Submit multiple orders to Alpaca.

    Submits each order in the request to Alpaca and collects results.
    Partial failures are handled gracefully - successful orders are counted
    even if some orders fail.

    Args:
        request: SubmitOrdersRequest with account and orders array

    Returns:
        SubmitOrdersResponse with submission counts and individual results

    Raises:
        HTTPException 500: If credentials are missing
    """
    logger.info(
        f"Submitting {len(request.orders)} orders for account {request.account.value}"
    )

    if not request.orders:
        return SubmitOrdersResponse(
            account=request.account.value,
            orders_submitted=0,
            orders_failed=0,
            skipped=False,
            results=[],
        )

    headers = get_alpaca_headers(request.account)
    results: list[OrderSubmitResult] = []
    orders_submitted = 0
    orders_failed = 0

    with httpx.Client(
        base_url=ALPACA_BASE_URL, headers=headers, timeout=ALPACA_TIMEOUT
    ) as client:
        sorted_orders = sorted(
            request.orders, key=lambda o: (o.side != "sell", o.symbol)
        )
        for order in sorted_orders:
            try:
                response = client.post(
                    "/v2/orders",
                    json={
                        "symbol": order.symbol,
                        "qty": str(order.qty),
                        "side": order.side,
                        "type": order.type,
                        "time_in_force": order.time_in_force,
                        "limit_price": str(order.limit_price),
                        "client_order_id": order.client_order_id,
                    },
                )
                response.raise_for_status()
                data = response.json()

                results.append(
                    OrderSubmitResult(
                        id=data.get("id"),
                        client_order_id=order.client_order_id,
                        symbol=order.symbol,
                        status=data.get("status", "accepted"),
                        error=None,
                    )
                )
                orders_submitted += 1
                logger.debug(f"Order submitted: {order.client_order_id}")

            except httpx.HTTPStatusError as e:
                error_msg = f"HTTP {e.response.status_code}"
                try:
                    error_data = e.response.json()
                    error_msg = error_data.get("message", error_msg)
                except Exception:
                    pass

                results.append(
                    OrderSubmitResult(
                        id=None,
                        client_order_id=order.client_order_id,
                        symbol=order.symbol,
                        status="rejected",
                        error=error_msg,
                    )
                )
                orders_failed += 1
                logger.warning(
                    f"Order failed: {order.client_order_id}, error: {error_msg}"
                )

            except Exception as e:
                results.append(
                    OrderSubmitResult(
                        id=None,
                        client_order_id=order.client_order_id,
                        symbol=order.symbol,
                        status="error",
                        error=str(e),
                    )
                )
                orders_failed += 1
                logger.error(f"Order error: {order.client_order_id}, error: {e}")

    logger.info(
        f"Order submission complete for {request.account.value}: "
        f"{orders_submitted} submitted, {orders_failed} failed"
    )

    return SubmitOrdersResponse(
        account=request.account.value,
        orders_submitted=orders_submitted,
        orders_failed=orders_failed,
        skipped=False,
        results=results,
    )


@router.get("/order-history", response_model=list[OrderHistoryItem])
def get_order_history(
    account: AlpacaAccount = Query(..., description="Trading account (ppo, sac, hrp)"),
    after: str = Query(
        ..., description="ISO date to fetch orders after (e.g., 2026-02-03)"
    ),
) -> list[OrderHistoryItem]:
    """Get order history from Alpaca for a specific account.

    Fetches all orders after the specified date, used to match intended
    orders with actual execution results.

    Args:
        account: The trading account (ppo, sac, hrp)
        after: ISO date string to fetch orders after

    Returns:
        List of OrderHistoryItem with order details

    Raises:
        HTTPException 500: If credentials are missing
        HTTPException 503: If Alpaca API is unavailable
    """
    logger.info(f"Fetching order history for account {account.value} after {after}")
    headers = get_alpaca_headers(account)

    try:
        with httpx.Client(
            base_url=ALPACA_BASE_URL, headers=headers, timeout=ALPACA_TIMEOUT
        ) as client:
            response = client.get(
                "/v2/orders",
                params={
                    "status": "all",
                    "after": after,
                    "limit": 100,
                },
            )
            response.raise_for_status()
            orders_data = response.json()

    except httpx.TimeoutException as e:
        logger.error(f"Alpaca API timeout for order history: {e}")
        raise HTTPException(
            status_code=503,
            detail="Alpaca API timeout fetching order history",
        ) from e
    except httpx.HTTPStatusError as e:
        logger.error(f"Alpaca API error fetching order history: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Alpaca API error: {e.response.status_code}",
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error fetching order history: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Failed to fetch order history: {e!s}",
        ) from e

    # Convert to response models
    orders = [
        OrderHistoryItem(
            id=o["id"],
            client_order_id=o.get("client_order_id", ""),
            symbol=o["symbol"],
            side=o["side"],
            status=o["status"],
            filled_qty=o.get("filled_qty"),
            filled_avg_price=o.get("filled_avg_price"),
        )
        for o in orders_data
    ]

    logger.info(f"Fetched {len(orders)} orders from history")
    return orders
