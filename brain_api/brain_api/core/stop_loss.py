"""ATR-based stop-loss reference for weekly-rebalanced US strategies.

Display-only helper. The four US weekly emails surface a recommended
stop-loss price next to each BUY order so the human reader has a manual
exit reference. No Alpaca-side bracket orders are submitted; the system
remains a paper-by-default rebalancer.

Formula
-------
::

    stop_distance = clamp(2 * ATR_14, MIN_PCT * entry, MAX_PCT * entry)
    stop_price    = entry_price - stop_distance

with ``MIN_PCT = 0.05`` (5%) and ``MAX_PCT = 0.10`` (10%). This is
Wilder's swing-stop convention (J. Welles Wilder, *New Concepts in
Technical Trading Systems*, 1978) with a percent floor so noisy
low-vol names don't get a fragile 1% stop and a percent ceiling so a
single name cannot lose >10% before the auto-exit reference triggers.

Why ATR (daily) for a weekly horizon
------------------------------------
The position is exposed to gaps and intraday wicks throughout the
week, not just close-to-close moves. ATR(14) on daily bars captures
both, which is precisely why it remains the industry-standard input
for stop-loss design even when the rebalance cadence is weekly.

No silent fallback
------------------
When ATR is missing (insufficient history, fetch failure), the helper
returns ``StopLoss(price=None, reason="atr_unavailable")``. Callers
must surface that reason verbatim in the email rather than substituting
a flat percent -- per AGENTS.md rule #1 (no silent fallbacks). The
sell side returns ``reason="sell_no_stop"`` because exits don't carry
a stop reference.
"""

from __future__ import annotations

from dataclasses import dataclass

# Stop-distance floor (5% of entry). Below this, even high-ATR names
# give the position effectively no breathing room against routine noise.
MIN_STOP_PCT: float = 0.05

# Stop-distance ceiling (10% of entry). Above this, a single 15-name
# slot can lose >10% before the operator's manual exit reference fires;
# capping protects the portfolio's drawdown profile.
MAX_STOP_PCT: float = 0.10

# Wilder's standard ATR multiplier for swing/position trading.
ATR_MULTIPLIER: float = 2.0


@dataclass(frozen=True)
class StopLoss:
    """A computed stop-loss reference for one order row.

    Attributes
    ----------
    price
        Stop price in dollars, or ``None`` when ATR is unavailable
        or the order is a sell.
    distance_pct
        Stop distance as a fraction of entry price, or ``None``.
    reason
        ``"atr14"`` on the happy path; ``"atr_unavailable"`` when ATR
        is missing; ``"sell_no_stop"`` for sells.
    """

    price: float | None
    distance_pct: float | None
    reason: str


def compute_stop_loss(entry_price: float, atr_14: float | None) -> StopLoss:
    """Compute the ATR(14)-based stop-loss reference for a buy order.

    Returns a ``StopLoss`` with ``price=None`` and
    ``reason="atr_unavailable"`` when ATR cannot be computed, rather
    than substituting a flat percent. The caller is expected to render
    the reason verbatim in the email so the operator sees why the row
    has no stop.

    Args:
        entry_price: Current market price used as the entry reference.
        atr_14: 14-period ATR (Wilder smoothing), or ``None`` if missing.

    Returns:
        StopLoss with ``price`` set when ATR is positive, else ``None``.
    """
    if entry_price <= 0:
        return StopLoss(price=None, distance_pct=None, reason="atr_unavailable")
    if atr_14 is None or atr_14 <= 0:
        return StopLoss(price=None, distance_pct=None, reason="atr_unavailable")

    raw_distance = ATR_MULTIPLIER * atr_14
    floor_distance = MIN_STOP_PCT * entry_price
    ceiling_distance = MAX_STOP_PCT * entry_price

    distance = max(raw_distance, floor_distance)
    distance = min(distance, ceiling_distance)

    return StopLoss(
        price=entry_price - distance,
        distance_pct=distance / entry_price,
        reason="atr14",
    )


def stop_loss_for_sell() -> StopLoss:
    """Return the canonical "no stop on a sell" sentinel.

    Sell orders close exposure; they don't carry a stop-loss. The
    sentinel is centralised so renderers don't reinvent the literal
    ``"sell_no_stop"`` reason string in multiple places.
    """
    return StopLoss(price=None, distance_pct=None, reason="sell_no_stop")
