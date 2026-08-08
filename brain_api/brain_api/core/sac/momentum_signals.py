"""Shared momentum + earnings-yield math for SAC per-stock signals.

Single source of truth so SAC train (``portfolio_rl.data_loading``) and
Monday inference (``temporal/activities/sac_context.py``) cannot drift on
encoding. Per AGENTS.md "no silent fallbacks", every function here raises
:class:`MomentumSignalError` on insufficient bars, missing/non-finite
prices, or missing/non-finite EPS -- callers must not zero-fill.

Encodings (index ``t`` is "as of" -- the most recent bar available):
- ``momentum_1w``   = P[t] / P[t-5]   - 1  (5 trading-day lookback)
- ``momentum_4w``   = P[t] / P[t-20]  - 1  (20 trading-day lookback)
- ``momentum_12_1`` = P[t-21] / P[t-252] - 1  (skip the most recent ~month,
  look back ~12 months -- the classic Jegadeesh/Titman "12-1" horizon)
- ``earnings_yield`` = eps_diluted / as_of_close (never raw P/E, never
  EBIT/EV -- those live under ``universe/stock_filter.py`` for a
  different, unrelated purpose)
"""

from __future__ import annotations

import math
from collections.abc import Sequence

MOM_1W_BARS = 5
MOM_4W_BARS = 20
MOM_12_1_SKIP_BARS = 21
MOM_12_1_LOOKBACK_BARS = 252

# Calendar-day buffer callers must add BEFORE a training/inference window's
# nominal start date when fetching daily closes, so the earliest week in
# the window still has MOM_12_1_SKIP_BARS + MOM_12_1_LOOKBACK_BARS (273)
# trading bars of history behind it. ~365/252 calendar days per trading
# day, plus slack for weekends/holidays. This is a wider price fetch on
# an existing call, not a new ETL pipeline.
MOM_12_1_CALENDAR_BUFFER_DAYS = 420


class MomentumSignalError(ValueError):
    """Raised when momentum/earnings-yield inputs are insufficient or invalid."""


def _price_at(closes: Sequence[float], index: int, *, field: str) -> float:
    if index < 0:
        raise MomentumSignalError(
            f"{field} requires more price history ({-index} more closes needed)"
        )
    if index >= len(closes):
        raise MomentumSignalError(f"{field} index {index} out of range")
    value = closes[index]
    if value is None or not math.isfinite(value) or value <= 0:
        raise MomentumSignalError(
            f"{field} price at index {index} must be finite and positive, got {value!r}"
        )
    return float(value)


def compute_momentum_1w(closes: Sequence[float], *, as_of_index: int) -> float:
    """Simple 1-week return: P[t] / P[t-5] - 1 (5 trading bars)."""
    p_t = _price_at(closes, as_of_index, field="momentum_1w.P_t")
    p_lag = _price_at(closes, as_of_index - MOM_1W_BARS, field="momentum_1w.P_t-5")
    return p_t / p_lag - 1.0


def compute_momentum_4w(closes: Sequence[float], *, as_of_index: int) -> float:
    """Simple 4-week return: P[t] / P[t-20] - 1 (20 trading bars)."""
    p_t = _price_at(closes, as_of_index, field="momentum_4w.P_t")
    p_lag = _price_at(closes, as_of_index - MOM_4W_BARS, field="momentum_4w.P_t-20")
    return p_t / p_lag - 1.0


def compute_momentum_12_1(closes: Sequence[float], *, as_of_index: int) -> float:
    """Classic 12-1 momentum: P[t-21] / P[t-252] - 1.

    Skips the most recent 21 trading bars (~1 month) and looks back a
    further 252 trading bars (~12 months) -- the Jegadeesh/Titman
    intermediate-horizon momentum factor. Do not collapse this into
    ``momentum_1w``/``momentum_4w`` math; the skip-month is intentional
    and mathematically distinct (short-term reversal avoidance).
    """
    p_skip = _price_at(
        closes,
        as_of_index - MOM_12_1_SKIP_BARS,
        field="momentum_12_1.P_t-21",
    )
    p_lookback = _price_at(
        closes,
        as_of_index - MOM_12_1_LOOKBACK_BARS,
        field="momentum_12_1.P_t-252",
    )
    return p_skip / p_lookback - 1.0


def compute_earnings_yield(*, eps_diluted: float, as_of_close: float) -> float:
    """Earnings yield: eps_diluted / as_of_close.

    ``eps_diluted`` must come from SEC point-in-time diluted EPS
    (``EarningsPerShareDiluted``, Basic fallback) -- never raw P/E and
    never the universe ``stock_filter`` EBIT/EV earnings yield, which
    is a different metric for a different purpose (see AGENTS.md
    universe pipeline invariants).
    """
    if eps_diluted is None or not math.isfinite(eps_diluted):
        raise MomentumSignalError(
            f"earnings_yield requires a finite eps_diluted, got {eps_diluted!r}"
        )
    if as_of_close is None or not math.isfinite(as_of_close) or as_of_close <= 0:
        raise MomentumSignalError(
            f"earnings_yield requires a positive finite close, got {as_of_close!r}"
        )
    return eps_diluted / as_of_close
