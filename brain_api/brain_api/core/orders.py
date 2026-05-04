"""Order generation logic for Alpaca paper trading.

Converts allocation weights into actionable market orders with:
- Idempotent client_order_id generation
- Minimum absolute weight change (1% of NAV per leg; full exit exempt)
- Minimum trade value filtering
- Buy funding cap (scale buys to cash + surviving sell proceeds)
- Sell qty capped at position quantity
"""

from dataclasses import dataclass, field

import pandas as pd
import yfinance as yf

from brain_api.core.stop_loss import compute_stop_loss, stop_loss_for_sell

# ============================================================================
# Configuration constants
# ============================================================================

# Skip orders smaller than this value (in dollars)
MIN_TRADE_VALUE: float = 10.0

# Skip rebalance legs smaller than this absolute weight delta (fraction of NAV; 0.01 = 1%)
MIN_REBALANCE_WEIGHT_DELTA: float = 0.01

# ATR lookback period (Wilder's standard 14 daily bars).
ATR_PERIOD: int = 14

# Daily-bar window pulled from yfinance for ATR computation.
# 40 days >> ATR_PERIOD so weekend/holiday gaps still leave 14+ usable bars.
ATR_FETCH_DAYS: int = 40


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
    """A single order to submit to Alpaca.

    ``stop_loss_*`` fields are display-only references computed at
    generation time so the email layer (and any other downstream
    renderer) gets the math without re-implementing it. See
    :mod:`brain_api.core.stop_loss` for the formula. Sells carry the
    canonical ``"sell_no_stop"`` sentinel; buys with no ATR available
    carry ``"atr_unavailable"`` (never a flat-percent fallback per
    AGENTS.md rule #1).
    """

    client_order_id: str
    symbol: str
    side: str  # "buy" or "sell"
    qty: float
    order_type: str  # "market"
    time_in_force: str  # "day"
    limit_price: float | None = None
    stop_loss_price: float | None = None
    stop_loss_distance_pct: float | None = None
    stop_loss_reason: str = "atr_unavailable"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "side": self.side,
            "qty": self.qty,
            "type": self.order_type,
            "time_in_force": self.time_in_force,
            "stop_loss_price": self.stop_loss_price,
            "stop_loss_distance_pct": self.stop_loss_distance_pct,
            "stop_loss_reason": self.stop_loss_reason,
        }
        if self.limit_price is not None:
            result["limit_price"] = self.limit_price
        return result


@dataclass
class OrderSummary:
    """Summary of generated orders."""

    buys: int
    sells: int
    total_buy_value: float
    total_sell_value: float
    turnover_pct: float
    skipped_small_orders: int
    skipped_below_threshold: int

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "buys": self.buys,
            "sells": self.sells,
            "total_buy_value": round(self.total_buy_value, 2),
            "total_sell_value": round(self.total_sell_value, 2),
            "turnover_pct": round(self.turnover_pct, 2),
            "skipped_small_orders": self.skipped_small_orders,
            "skipped_below_threshold": self.skipped_below_threshold,
        }


@dataclass
class GenerateOrdersResult:
    """Result of order generation."""

    orders: list[Order]
    summary: OrderSummary
    prices_used: dict[str, float]  # symbol -> price used for calculations
    atr_used: dict[str, float] = field(default_factory=dict)  # symbol -> ATR(14)


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


def fetch_ohlc_window(
    symbols: list[str],
    days: int = ATR_FETCH_DAYS,
) -> dict[str, list[tuple[float, float, float]]]:
    """Fetch ``(high, low, close)`` daily bars for each symbol.

    Used as the input to :func:`compute_atr_map`. Returns whatever
    yfinance gives back; symbols with no data are simply absent from
    the result dict (no silent fake-data fallback per AGENTS.md rule
    #1).

    Math invariant: each emitted ``(high, low, close)`` tuple comes
    from the **same trading date**. NaN handling is a joint
    ``dropna(how="any")`` over the three columns, so a missing high
    cannot pair up with the next day's low when the True Range is
    computed downstream.

    Args:
        symbols: Tickers to fetch.
        days: Calendar-day window (>> ATR_PERIOD so non-trading days
            don't starve the ATR computation).

    Returns:
        Dict mapping symbol -> ordered list of ``(high, low, close)``
        tuples, one per available trading day, oldest first.
    """
    bars: dict[str, list[tuple[float, float, float]]] = {}

    if not symbols:
        return bars

    try:
        tickers_str = " ".join(symbols)
        data = yf.download(
            tickers_str,
            period=f"{days}d",
            progress=False,
        )
    except Exception as e:
        print(f"[Orders] OHLC batch fetch failed: {e}")
        return bars

    if data is None or data.empty:
        return bars

    if len(symbols) == 1:
        symbol = symbols[0]
        if {"High", "Low", "Close"}.issubset(data.columns):
            sub = data[["High", "Low", "Close"]].dropna(how="any")
            bars[symbol] = [
                (float(row.High), float(row.Low), float(row.Close))
                for row in sub.itertuples(index=False)
            ]
            if not bars[symbol]:
                bars.pop(symbol)
        return bars

    if (
        "High" not in data.columns
        or "Low" not in data.columns
        or "Close" not in data.columns
    ):
        return bars

    high_df = data["High"]
    low_df = data["Low"]
    close_df = data["Close"]
    for symbol in symbols:
        if symbol not in high_df.columns:
            continue
        # Joint dropna: every surviving row carries H/L/C for the
        # same date. Per-column dropna would silently misalign
        # columns and produce a fictional True Range.
        sub = pd.concat(
            [
                high_df[symbol].rename("High"),
                low_df[symbol].rename("Low"),
                close_df[symbol].rename("Close"),
            ],
            axis=1,
        ).dropna(how="any")
        if sub.empty:
            continue
        bars[symbol] = [
            (float(row.High), float(row.Low), float(row.Close))
            for row in sub.itertuples(index=False)
        ]

    return bars


def compute_atr_map(
    ohlc: dict[str, list[tuple[float, float, float]]],
    period: int = ATR_PERIOD,
) -> dict[str, float]:
    """Compute ATR(14) per symbol using Wilder's smoothing.

    True Range for bar ``t`` is::

        TR_t = max(high_t - low_t,
                   |high_t - close_{t-1}|,
                   |low_t  - close_{t-1}|)

    ATR is the Wilder-smoothed TR series. We use the simple-MA seed
    (mean of the first ``period`` TRs) and then the recursive
    Wilder update ``ATR_t = (ATR_{t-1} * (period-1) + TR_t) / period``.

    Symbols with fewer than ``period + 1`` bars are silently absent
    from the output (caller surfaces the missing entry verbatim as
    "atr_unavailable" -- never substituted with a flat percent).

    Args:
        ohlc: ``{symbol: [(high, low, close), ...]}`` oldest-first.
        period: ATR window (default 14).

    Returns:
        ``{symbol: atr_14}`` for every symbol with enough history.
    """
    atrs: dict[str, float] = {}

    for symbol, bars in ohlc.items():
        if len(bars) < period + 1:
            continue

        true_ranges: list[float] = []
        for i in range(1, len(bars)):
            high, low, _ = bars[i]
            prev_close = bars[i - 1][2]
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close),
            )
            true_ranges.append(tr)

        if len(true_ranges) < period:
            continue

        seed = sum(true_ranges[:period]) / period
        atr = seed
        for tr in true_ranges[period:]:
            atr = (atr * (period - 1) + tr) / period

        if atr > 0:
            atrs[symbol] = atr

    return atrs


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
    atr_map: dict[str, float] | None = None,
) -> GenerateOrdersResult:
    """Generate orders to rebalance portfolio to target weights.

    Args:
        target_weights: Target allocation weights (symbol -> weight, including CASH)
        portfolio: Current portfolio state (cash + positions)
        run_id: Run identifier (e.g., "paper:2026-01-20")
        attempt: Attempt number
        algorithm: Algorithm name (for logging)
        prices: Optional pre-fetched prices (if None, will fetch)
        atr_map: Optional pre-computed ATR(14) per symbol. When ``None``,
            ATR is fetched alongside prices so the email layer can render
            a stop-loss reference next to each buy. Symbols with
            insufficient history are silently absent (caller surfaces
            "atr_unavailable" verbatim per AGENTS.md rule #1).

    Returns:
        GenerateOrdersResult with orders, summary, prices, and ATR map.
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
                skipped_below_threshold=0,
            ),
            prices_used={},
            atr_used={},
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

    # Compute ATR(14) alongside prices so the email layer has a
    # stop-loss reference per buy without an additional yfinance call.
    if atr_map is None:
        try:
            ohlc = fetch_ohlc_window(symbols_needed)
            atr_map = compute_atr_map(ohlc)
        except Exception as e:
            print(f"[Orders] ATR computation failed: {e}")
            atr_map = {}

    # Build current weights
    current_weights: dict[str, float] = {"CASH": portfolio.cash / total_value}
    for pos in portfolio.positions:
        current_weights[pos.symbol] = pos.market_value / total_value

    # Calculate required trades
    orders: list[Order] = []
    skipped_small_orders = 0
    skipped_below_threshold = 0
    total_buy_value = 0.0
    total_sell_value = 0.0

    # Process all symbols (both in target and current)
    all_symbols = set(target_weights.keys()) | set(current_weights.keys())
    all_symbols.discard("CASH")

    for symbol in sorted(all_symbols):
        current_weight = current_weights.get(symbol, 0.0)
        target_weight = target_weights.get(symbol, 0.0)
        weight_diff = target_weight - current_weight

        # Skip negligible change (always allow full exit: target 0 with a position)
        is_full_exit = target_weight == 0.0 and current_weight > 0.0
        if abs(weight_diff) < MIN_REBALANCE_WEIGHT_DELTA and not is_full_exit:
            skipped_below_threshold += 1
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

        # Determine side
        if weight_diff > 0:
            side = "buy"
            total_buy_value += trade_value
        else:
            side = "sell"
            total_sell_value += trade_value

        qty = trade_value / current_price

        # Cap sell qty at position to avoid requesting more shares than held
        if side == "sell":
            position = current_positions.get(symbol)
            if position:
                qty = min(qty, position.qty)

        client_order_id = generate_client_order_id(run_id, attempt, symbol, side)

        # Stop-loss reference: computed once at generation time so all
        # downstream renderers (emails, dashboards) read the same math.
        # Sells get the sell-no-stop sentinel; buys with no ATR get
        # the atr_unavailable sentinel (no silent flat-percent fallback).
        if side == "sell":
            stop = stop_loss_for_sell()
        else:
            stop = compute_stop_loss(current_price, atr_map.get(symbol))

        order = Order(
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            qty=round(qty, 4),
            order_type="market",
            time_in_force="day",
            stop_loss_price=stop.price,
            stop_loss_distance_pct=stop.distance_pct,
            stop_loss_reason=stop.reason,
        )
        orders.append(order)

    # Cap buys so total notional <= cash + expected sell proceeds (skipped sells reduce cash)
    available_for_buys = portfolio.cash + total_sell_value
    if available_for_buys <= 0:
        orders = [o for o in orders if o.side == "sell"]
        total_buy_value = 0.0
    elif total_buy_value > available_for_buys:
        scale = available_for_buys / total_buy_value
        for o in orders:
            if o.side == "buy":
                o.qty = round(o.qty * scale, 4)
        orders = [o for o in orders if o.side == "sell" or o.qty > 0]
        total_buy_value = sum(
            o.qty * prices[o.symbol]
            for o in orders
            if o.side == "buy" and o.symbol in prices
        )

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
        skipped_below_threshold=skipped_below_threshold,
    )

    print(
        f"[Orders] {algorithm}: Generated {len(orders)} orders "
        f"({buys} buys, {sells} sells), turnover={turnover:.1f}%"
    )

    return GenerateOrdersResult(
        orders=orders,
        summary=summary,
        prices_used=prices,
        atr_used=atr_map,
    )
