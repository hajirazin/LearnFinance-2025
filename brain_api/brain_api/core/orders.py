"""Order generation logic for Alpaca paper trading.

Converts allocation weights into actionable limit orders with:
- Idempotent client_order_id generation
- Minimum trade value filtering
- Limit price calculation with buffer
"""

from dataclasses import dataclass

import yfinance as yf

# ============================================================================
# Configuration constants
# ============================================================================

# Skip orders smaller than this value (in dollars)
MIN_TRADE_VALUE: float = 10.0

# Buffer for limit price (2% above last price for buys, 2% below for sells)
# This gives ~95%+ fill rate while still providing price protection
LIMIT_PRICE_BUFFER_PCT: float = 0.02


# ============================================================================
# Data classes
# ============================================================================


@dataclass
class PositionInput:
    """Input position from Alpaca portfolio."""

    symbol: str
    qty: float
    market_value: float


@dataclass
class PortfolioInput:
    """Input portfolio state from Alpaca."""

    cash: float
    positions: list[PositionInput]

    @property
    def total_value(self) -> float:
        """Total portfolio value (cash + positions)."""
        return self.cash + sum(p.market_value for p in self.positions)


@dataclass
class Order:
    """A single order to submit to Alpaca."""

    client_order_id: str
    symbol: str
    side: str  # "buy" or "sell"
    qty: float
    order_type: str  # "limit"
    limit_price: float
    time_in_force: str  # "day"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "side": self.side,
            "qty": self.qty,
            "type": self.order_type,
            "limit_price": self.limit_price,
            "time_in_force": self.time_in_force,
        }


@dataclass
class OrderSummary:
    """Summary of generated orders."""

    buys: int
    sells: int
    total_buy_value: float
    total_sell_value: float
    turnover_pct: float
    skipped_small_orders: int

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "buys": self.buys,
            "sells": self.sells,
            "total_buy_value": round(self.total_buy_value, 2),
            "total_sell_value": round(self.total_sell_value, 2),
            "turnover_pct": round(self.turnover_pct, 2),
            "skipped_small_orders": self.skipped_small_orders,
        }


@dataclass
class GenerateOrdersResult:
    """Result of order generation."""

    orders: list[Order]
    summary: OrderSummary
    prices_used: dict[str, float]  # symbol -> price used for calculations


# ============================================================================
# Helper functions
# ============================================================================


def generate_client_order_id(
    run_id: str,
    attempt: int,
    symbol: str,
    side: str,
) -> str:
    """Generate deterministic client_order_id for idempotency.

    Format: paper:YYYY-MM-DD:attempt-<N>:<SYMBOL>:<SIDE>
    Example: paper:2026-01-20:attempt-1:AAPL:BUY

    Args:
        run_id: Run identifier (e.g., "paper:2026-01-20")
        attempt: Attempt number (1, 2, 3, ...)
        symbol: Stock symbol
        side: "buy" or "sell"

    Returns:
        Deterministic client order ID
    """
    return f"{run_id}:attempt-{attempt}:{symbol}:{side.upper()}"


def fetch_current_prices(symbols: list[str]) -> dict[str, float]:
    """Fetch current prices for symbols using yfinance.

    Args:
        symbols: List of stock symbols

    Returns:
        Dict mapping symbol -> current price (last close)
    """
    prices: dict[str, float] = {}

    if not symbols:
        return prices

    try:
        # Use batch download for efficiency
        tickers_str = " ".join(symbols)
        data = yf.download(
            tickers_str,
            period="1d",
            progress=False,
        )

        if data is not None and not data.empty:
            if len(symbols) == 1:
                # Single ticker returns flat DataFrame
                symbol = symbols[0]
                if "Close" in data.columns:
                    last_close = data["Close"].iloc[-1]
                    if last_close > 0:
                        prices[symbol] = float(last_close)
            else:
                # Multiple tickers: get Close prices
                if "Close" in data.columns:
                    close_data = data["Close"]
                    for symbol in symbols:
                        if symbol in close_data.columns:
                            last_close = close_data[symbol].iloc[-1]
                            if last_close > 0:
                                prices[symbol] = float(last_close)

    except Exception as e:
        print(f"[Orders] Batch price fetch failed: {e}")

    # Fallback: fetch missing symbols individually
    missing = [s for s in symbols if s not in prices]
    for symbol in missing:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.fast_info
            if hasattr(info, "last_price") and info.last_price and info.last_price > 0:
                prices[symbol] = float(info.last_price)
            else:
                # Fallback to history
                hist = ticker.history(period="1d")
                if hist is not None and not hist.empty and "Close" in hist.columns:
                    prices[symbol] = float(hist["Close"].iloc[-1])
        except Exception as e:
            print(f"[Orders] Failed to fetch price for {symbol}: {e}")

    return prices


def calculate_limit_price(current_price: float, side: str) -> float:
    """Calculate limit price with buffer.

    For buys: limit = current_price * (1 + buffer)
    For sells: limit = current_price * (1 - buffer)

    Args:
        current_price: Current market price
        side: "buy" or "sell"

    Returns:
        Limit price rounded to 2 decimal places
    """
    if side == "buy":
        return round(current_price * (1 + LIMIT_PRICE_BUFFER_PCT), 2)
    else:
        return round(current_price * (1 - LIMIT_PRICE_BUFFER_PCT), 2)


# ============================================================================
# Main order generation function
# ============================================================================


def generate_orders(
    target_weights: dict[str, float],
    portfolio: PortfolioInput,
    run_id: str,
    attempt: int,
    algorithm: str,
    prices: dict[str, float] | None = None,
) -> GenerateOrdersResult:
    """Generate orders to rebalance portfolio to target weights.

    Args:
        target_weights: Target allocation weights (symbol -> weight, including CASH)
        portfolio: Current portfolio state (cash + positions)
        run_id: Run identifier (e.g., "paper:2026-01-20")
        attempt: Attempt number
        algorithm: Algorithm name (for logging)
        prices: Optional pre-fetched prices (if None, will fetch)

    Returns:
        GenerateOrdersResult with orders, summary, and prices used
    """
    total_value = portfolio.total_value

    if total_value <= 0:
        return GenerateOrdersResult(
            orders=[],
            summary=OrderSummary(
                buys=0,
                sells=0,
                total_buy_value=0.0,
                total_sell_value=0.0,
                turnover_pct=0.0,
                skipped_small_orders=0,
            ),
            prices_used={},
        )

    # Get all symbols we need prices for (excluding CASH)
    symbols_needed = [s for s in target_weights if s != "CASH"]

    # Add current positions that might need to be sold
    current_positions = {p.symbol: p for p in portfolio.positions}
    for symbol in current_positions:
        if symbol not in symbols_needed:
            symbols_needed.append(symbol)

    # Fetch prices if not provided
    if prices is None:
        prices = fetch_current_prices(symbols_needed)

    # Build current weights
    current_weights: dict[str, float] = {"CASH": portfolio.cash / total_value}
    for pos in portfolio.positions:
        current_weights[pos.symbol] = pos.market_value / total_value

    # Calculate required trades
    orders: list[Order] = []
    skipped_small_orders = 0
    total_buy_value = 0.0
    total_sell_value = 0.0

    # Process all symbols (both in target and current)
    all_symbols = set(target_weights.keys()) | set(current_weights.keys())
    all_symbols.discard("CASH")

    for symbol in sorted(all_symbols):
        current_weight = current_weights.get(symbol, 0.0)
        target_weight = target_weights.get(symbol, 0.0)
        weight_diff = target_weight - current_weight

        # Skip if no change needed
        if abs(weight_diff) < 0.0001:  # Less than 0.01% change
            continue

        # Skip if we don't have a price
        if symbol not in prices:
            print(f"[Orders] Skipping {symbol}: no price available")
            continue

        current_price = prices[symbol]
        if current_price <= 0:
            print(f"[Orders] Skipping {symbol}: invalid price {current_price}")
            continue

        # Calculate trade value
        trade_value = abs(weight_diff) * total_value

        # Skip small trades
        if trade_value < MIN_TRADE_VALUE:
            skipped_small_orders += 1
            continue

        # Calculate quantity
        qty = trade_value / current_price

        # Determine side
        if weight_diff > 0:
            side = "buy"
            total_buy_value += trade_value
        else:
            side = "sell"
            total_sell_value += trade_value

        # Calculate limit price
        limit_price = calculate_limit_price(current_price, side)

        # Generate client order ID
        client_order_id = generate_client_order_id(run_id, attempt, symbol, side)

        # Create order
        order = Order(
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            qty=round(qty, 4),  # Alpaca supports fractional shares
            order_type="limit",
            limit_price=limit_price,
            time_in_force="day",
        )
        orders.append(order)

    # Calculate turnover
    turnover = (total_buy_value + total_sell_value) / 2 / total_value * 100

    # Count buys and sells
    buys = sum(1 for o in orders if o.side == "buy")
    sells = sum(1 for o in orders if o.side == "sell")

    summary = OrderSummary(
        buys=buys,
        sells=sells,
        total_buy_value=total_buy_value,
        total_sell_value=total_sell_value,
        turnover_pct=turnover,
        skipped_small_orders=skipped_small_orders,
    )

    print(
        f"[Orders] {algorithm}: Generated {len(orders)} orders "
        f"({buys} buys, {sells} sells), turnover={turnover:.1f}%"
    )

    return GenerateOrdersResult(
        orders=orders,
        summary=summary,
        prices_used=prices,
    )
