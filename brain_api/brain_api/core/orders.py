"""Order generation logic for Alpaca paper trading.

Converts allocation weights into actionable market orders with:
- Idempotent client_order_id generation
- Minimum absolute weight change (1% of NAV per leg; full exit exempt)
- Minimum trade value filtering (full-exit sells exempt)
- Full-exit flatten uses broker position qty, not notional / Yahoo price
- Buy funding cap (scale buys to cash + surviving sell proceeds)
- Partial sell qty capped at position quantity
"""

import math
from dataclasses import dataclass, field
from datetime import date, timedelta

from brain_api.core.prices import load_prices_yfinance
from brain_api.core.stop_loss import compute_stop_loss, stop_loss_for_sell

# ============================================================================
# Configuration constants
# ============================================================================

# Skip non-flatten orders smaller than this value (in dollars).
# Full-exit sells (target weight 0 with an open position) are exempt.
MIN_TRADE_VALUE: float = 10.0

# Skip rebalance legs smaller than this absolute weight delta (fraction of NAV; 0.01 = 1%)
MIN_REBALANCE_WEIGHT_DELTA: float = 0.01

# ATR lookback period (Wilder's standard 14 daily bars).
ATR_PERIOD: int = 14

# Daily-bar window pulled from yfinance for ATR computation.
# 40 days >> ATR_PERIOD so weekend/holiday gaps still leave 14+ usable bars.
ATR_FETCH_DAYS: int = 40


class OrderPriceError(ValueError):
    """Raised when a material rebalance leg has no safe current price."""


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
    cash_qty: float | None = None

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


def is_full_exit(target_weight: float, current_weight: float) -> bool:
    """True when a held name is targeted to zero (flatten / sell-all)."""
    return target_weight == 0.0 and current_weight > 0.0


def full_exit_sell_qty(position: PositionInput) -> float:
    """Broker share quantity to flatten a position to target weight 0.

    Uses the held lot size as-is. Sizing from notional / Yahoo price
    undershoots when marks differ; rounding to 4 decimals leaves dust.
    """
    if position.qty <= 0:
        raise ValueError(
            f"full-exit sell for {position.symbol} requires qty > 0; got {position.qty}"
        )
    return position.qty


def _full_exit_order(
    *,
    symbol: str,
    position: PositionInput,
    run_id: str,
    attempt: int,
) -> Order:
    """Market sell of the entire broker lot. Qty is not rounded."""
    stop = stop_loss_for_sell()
    return Order(
        client_order_id=generate_client_order_id(run_id, attempt, symbol, "sell"),
        symbol=symbol,
        side="sell",
        qty=full_exit_sell_qty(position),
        order_type="market",
        time_in_force="day",
        stop_loss_price=stop.price,
        stop_loss_distance_pct=stop.distance_pct,
        stop_loss_reason=stop.reason,
        cash_qty=round(position.market_value, 2),
    )


def fetch_current_prices(symbols: list[str]) -> dict[str, float]:
    """Fetch current prices for symbols using the shared yfinance loader.

    Args:
        symbols: List of stock symbols

    Returns:
        Dict mapping symbol -> current price (last close)
    """
    prices: dict[str, float] = {}
    if not symbols:
        return prices

    today = date.today()
    frames = load_prices_yfinance(
        symbols,
        today - timedelta(days=7),
        today + timedelta(days=1),
        log_prefix="[Orders]",
    )
    for symbol, frame in frames.items():
        last_close = float(frame["close"].iloc[-1])
        if last_close > 0:
            prices[symbol] = last_close
    return prices


def fetch_ohlc_window(
    symbols: list[str],
    days: int = ATR_FETCH_DAYS,
) -> dict[str, list[tuple[float, float, float]]]:
    """Fetch ``(high, low, close)`` daily bars for each symbol.

    Used as the input to :func:`compute_atr_map`. Returns whatever
    the shared loader gives back; symbols with no data are simply
    absent from the result dict (no silent fake-data fallback per
    AGENTS.md rule #1).

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

    today = date.today()
    frames = load_prices_yfinance(
        symbols,
        today - timedelta(days=days),
        today + timedelta(days=1),
        log_prefix="[Orders]",
    )
    for symbol, frame in frames.items():
        sub = frame[["high", "low", "close"]].dropna(how="any")
        if sub.empty:
            continue
        bars[symbol] = [
            (float(row.high), float(row.low), float(row.close))
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
    order_side: str = "all",
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
        order_side: Generate ``all``, ``sell``, or ``buy`` legs only.
        prices: Optional pre-fetched prices (if None, will fetch). Full-exit
            sells do not require a price; they flatten at ``position.qty``.
        atr_map: Optional pre-computed ATR(14) per symbol. When ``None``,
            ATR is fetched alongside prices so the email layer can render
            a stop-loss reference next to each buy. Symbols with
            insufficient history are silently absent (caller surfaces
            "atr_unavailable" verbatim per AGENTS.md rule #1).

    Returns:
        GenerateOrdersResult with orders, summary, prices, and ATR map.
    """
    total_value = portfolio.total_value

    if order_side not in {"all", "sell", "buy"}:
        raise ValueError("order_side must be one of: all, sell, buy")

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

    current_positions = {p.symbol: p for p in portfolio.positions}
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

    material_symbols: list[str] = []
    material_sides: dict[str, str] = {}
    for symbol in sorted(all_symbols):
        current_weight = current_weights.get(symbol, 0.0)
        target_weight = target_weights.get(symbol, 0.0)
        weight_diff = target_weight - current_weight
        side = "buy" if weight_diff > 0 else "sell"
        if order_side != "all" and side != order_side:
            continue
        if is_full_exit(target_weight, current_weight):
            # Flatten uses broker qty; do not require a Yahoo price.
            continue
        if abs(weight_diff) < MIN_REBALANCE_WEIGHT_DELTA:
            continue
        if abs(weight_diff) * total_value < MIN_TRADE_VALUE:
            continue
        material_symbols.append(symbol)
        material_sides[symbol] = side

    # Each execution phase fetches its own current prices inside Brain. In the
    # sell-wait-buy workflow this means the buy call happens after the durable
    # wait and cannot reuse decision-time or sell-time prices.
    if prices is None:
        prices = fetch_current_prices(material_symbols)

    invalid_prices = []
    for symbol in material_symbols:
        try:
            price = float(prices[symbol])
        except (KeyError, TypeError, ValueError):
            invalid_prices.append(symbol)
            continue
        if not math.isfinite(price) or price <= 0:
            invalid_prices.append(symbol)
            continue
        prices[symbol] = price
    if invalid_prices:
        raise OrderPriceError(
            "finite positive current prices are required for all material "
            f"order legs; missing_or_invalid={invalid_prices}"
        )

    # Compute ATR(14) only after every material leg has a validated price.
    # ATR remains display-only and best-effort, but an invalid current-price
    # snapshot must fail before any secondary provider call is attempted.
    if atr_map is None:
        try:
            buy_symbols = [
                symbol for symbol in material_symbols if material_sides[symbol] == "buy"
            ]
            ohlc = fetch_ohlc_window(buy_symbols)
            atr_map = compute_atr_map(ohlc)
        except Exception as e:
            print(f"[Orders] ATR computation failed: {e}")
            atr_map = {}

    for symbol in sorted(all_symbols):
        current_weight = current_weights.get(symbol, 0.0)
        target_weight = target_weights.get(symbol, 0.0)
        weight_diff = target_weight - current_weight
        side = "buy" if weight_diff > 0 else "sell"
        if order_side != "all" and side != order_side:
            continue

        if is_full_exit(target_weight, current_weight):
            position = current_positions.get(symbol)
            if position is None:
                raise ValueError(f"full-exit sell for {symbol} has no open position")
            orders.append(
                _full_exit_order(
                    symbol=symbol,
                    position=position,
                    run_id=run_id,
                    attempt=attempt,
                )
            )
            total_sell_value += position.market_value
            continue

        # Skip negligible change (full exits handled above)
        if abs(weight_diff) < MIN_REBALANCE_WEIGHT_DELTA:
            skipped_below_threshold += 1
            continue

        # Calculate trade value
        trade_value = abs(weight_diff) * total_value

        # Skip small trades (buys and partial trims; full exits already returned)
        if trade_value < MIN_TRADE_VALUE:
            skipped_small_orders += 1
            continue

        current_price = prices[symbol]

        # Determine side
        if side == "buy":
            total_buy_value += trade_value
        else:
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
            cash_qty=round(qty * current_price, 2),
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
                if o.cash_qty is not None and o.symbol in prices:
                    o.cash_qty = round(o.qty * prices[o.symbol], 2)
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


# ============================================================================
# Paper allocation (whole shares only, no order submission)
# ============================================================================


@dataclass
class AllocationDetail:
    """A single row in the paper-allocation table.

    Shows the theoretical whole-share quantity for a given weight
    target and NAV, using the current market price. No order is
    generated or submitted.

    ``stop_loss_*`` fields are display-only references computed at
    conversion time so India Stage 2 email tables can render the same
    ATR stop column as the US order table. Every paper row is a hold
    / buy reference (never ``sell_no_stop``). See
    :mod:`brain_api.core.stop_loss` for the formula.
    """

    symbol: str
    weight_pct: float
    price: float
    whole_shares: int
    trade_value: float
    stop_loss_price: float | None = None
    stop_loss_distance_pct: float | None = None
    stop_loss_reason: str = "atr_unavailable"


@dataclass
class PaperAllocationResult:
    """Result of converting weights to whole shares (paper-only).

    The ``details`` list is sorted by weight descending so the email
    table renders the largest allocation first.
    """

    details: list[AllocationDetail]
    total_nav: float
    prices_used: dict[str, float]
    total_allocated_pct: float


def convert_weights_to_whole_shares(
    percentage_weights: dict[str, float],
    total_nav: float,
    prices: dict[str, float] | None = None,
    atr_map: dict[str, float] | None = None,
) -> PaperAllocationResult:
    """Convert percentage weights to whole shares at current prices.

    This is a paper-only computation with no order submission. It
    reuses the same price-fetching logic as :func:`generate_orders`
    but uses floor-to-integer (``int()``) instead of fractional shares.

    Stop-loss references are attached per row via
    :func:`compute_stop_loss` so India weekly emails can surface the
    same ATR(14)x2 / 5%-10% clamp column as the US order table.

    Args:
        percentage_weights: ``{symbol: weight_pct}`` where 100 = 100%.
        total_nav: Notional portfolio value in the local currency
            (e.g. 1 000 000 INR).
        prices: Optional pre-fetched prices. If ``None``, fetches via
            yfinance.
        atr_map: Optional pre-computed ATR(14) per symbol. When
            ``None``, fetches an OHLC window and computes ATR the same
            way as :func:`generate_orders` (injectable for tests).

    Returns:
        PaperAllocationResult with whole-share quantities and stop-loss
        fields on each detail row.
    """
    symbols = [s for s in percentage_weights if s and percentage_weights[s] > 0]

    if prices is None:
        prices = fetch_current_prices(symbols)

    # Compute ATR(14) alongside prices so the India Stage 2 email
    # table has a stop-loss reference per row without a second round-trip.
    if atr_map is None:
        try:
            ohlc = fetch_ohlc_window(symbols)
            atr_map = compute_atr_map(ohlc)
        except Exception as e:
            print(f"[PaperAllocation] ATR computation failed: {e}")
            atr_map = {}

    details: list[AllocationDetail] = []
    total_allocated_pct = 0.0

    for symbol in sorted(symbols, key=lambda s: percentage_weights[s], reverse=True):
        wt = percentage_weights[symbol]
        if wt <= 0:
            continue
        total_allocated_pct += wt
        price = prices.get(symbol, 0.0)
        if price <= 0:
            print(f"[PaperAllocation] Skipping {symbol}: no valid price ({price})")
            continue
        trade_value = (wt / 100.0) * total_nav
        if trade_value < MIN_TRADE_VALUE:
            continue
        whole_shares = int(trade_value / price)
        stop = compute_stop_loss(price, atr_map.get(symbol))
        details.append(
            AllocationDetail(
                symbol=symbol,
                weight_pct=wt,
                price=price,
                whole_shares=whole_shares,
                trade_value=round(whole_shares * price, 2),
                stop_loss_price=stop.price,
                stop_loss_distance_pct=stop.distance_pct,
                stop_loss_reason=stop.reason,
            )
        )

    return PaperAllocationResult(
        details=details,
        total_nav=total_nav,
        prices_used=prices,
        total_allocated_pct=round(total_allocated_pct, 2),
    )
