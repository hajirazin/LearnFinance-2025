"""IBKR Singapore Tiered transaction-cost model for SAC.

This module replaces the legacy flat ``cost_bps`` * turnover formula
in :mod:`brain_api.core.portfolio_rl.rewards` with a per-symbol,
per-leg cost model calibrated to Interactive Brokers Singapore's
**Tiered** pricing tier for US stocks (≤300k shares/month volume).

The cost model is broker-specific by name on purpose: SAC trains under
IBKR economics so that the policy learns to respect IBKR's per-order
minimum, sell-side regulatory schedule, and per-share clearing fees.
The order-submission code path (Alpaca client / routes/ibkr.py) is
untouched -- this module only feeds the **reward** computation.

In scope (modelled per leg, summed across symbols):

- Commission: USD 0.0035 / share, **min USD 0.35 / order**, **max
  1% of trade value**.
- NSCC/DTC clearing: USD 0.00020 / share, both sides.
- FINRA CAT: USD 0.000033 / share, both sides.
- SEC Transaction Fee: 0.0000206 * sale notional (sells only).
- FINRA TAF: 0.000195 * shares sold (sells only, capped USD 9.27).
- NYSE / FINRA pass-through: commission * (0.000175 + 0.000565)
  (tiny but included for completeness).

Out of scope (intentionally not modelled):

- FX (SGD↔USD): assume USD-funded account.
- US dividend WHT: ``symbol_returns`` stay price-only.
- Account / inactivity / platform fees: zero for our setup.

Math invariant
--------------
The dollar cost is converted to a return fraction so the existing
log-space reward formula in
:func:`compute_blended_reward` is unchanged in shape::

    tc_fraction = total_dollar_cost / nav_usd
    reward = reward_scale * (log(1 + r) - log(1 + tc_fraction)) + DSR

The min/max-per-order caps make realised bps a function of NAV and
per-symbol notional, which is the whole point of moving away from a
flat ``cost_bps`` constant.

No silent fallbacks (per AGENTS.md rule #1)
-------------------------------------------
- Missing price for a non-zero-delta symbol → :class:`ValueError`.
- ``nav_usd <= 0`` → :class:`ValueError`.
- Negative shares / negative notional → :class:`ValueError`.

A symbol with zero weight delta produces a zero-cost
:class:`LegCost` (no leg, no min charge); that is correctness, not a
fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np

Side = Literal["buy", "sell"]


@dataclass(frozen=True)
class IBKRSingaporeCostConfig:
    """All IBKR-SG tiered rates / caps used by the SAC reward.

    Defaults reflect the published IBKR Singapore Tiered schedule for
    US stocks at ≤300k shares/month (the only tier we care about for
    a small SAC portfolio). Every value is overridable so future rate
    changes (SEC fee in particular updates yearly) can be patched
    without code edits.

    The ``nav_usd`` field is the **assumed portfolio value** used by
    training-time cost computation. It is intentionally a single
    knob: the cost model needs a notional anchor to convert weight
    deltas into dollar trades, and the $0.35 per-order minimum binds
    very differently at $10k vs $100k. Calibration in tests targets
    ``nav_usd=10_000``.
    """

    # === Commission (per-order) ===
    commission_per_share: float = 0.0035
    commission_min: float = 0.35
    commission_max_pct: float = 0.01  # 1% of trade value

    # === Clearing (both sides, per share) ===
    clearing_per_share: float = 0.00020

    # === FINRA CAT (both sides, per share) ===
    cat_per_share: float = 0.000033

    # === Sell-side regulatory ===
    sec_fee_rate: float = 0.0000206  # * sale notional
    finra_taf_per_share: float = 0.000195  # * shares sold
    finra_taf_cap: float = 9.27  # cap per trade

    # === Pass-through on commission ===
    nyse_pass_through_rate: float = 0.000175
    finra_pass_through_rate: float = 0.000565

    # === Calibration anchor ===
    nav_usd: float = 10_000.0

    @classmethod
    def default(cls) -> IBKRSingaporeCostConfig:
        """Return the default IBKR-SG Tiered config (calibrated to $10k NAV)."""
        return cls()

    def with_nav(self, nav_usd: float) -> IBKRSingaporeCostConfig:
        """Return a copy with a different NAV anchor."""
        if nav_usd <= 0:
            raise ValueError(f"nav_usd must be > 0, got {nav_usd}")
        return replace(self, nav_usd=nav_usd)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a dict (config_hash + storage round-trip)."""
        return {
            "commission_per_share": self.commission_per_share,
            "commission_min": self.commission_min,
            "commission_max_pct": self.commission_max_pct,
            "clearing_per_share": self.clearing_per_share,
            "cat_per_share": self.cat_per_share,
            "sec_fee_rate": self.sec_fee_rate,
            "finra_taf_per_share": self.finra_taf_per_share,
            "finra_taf_cap": self.finra_taf_cap,
            "nyse_pass_through_rate": self.nyse_pass_through_rate,
            "finra_pass_through_rate": self.finra_pass_through_rate,
            "nav_usd": self.nav_usd,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IBKRSingaporeCostConfig:
        """Inverse of :meth:`to_dict`."""
        return cls(**data)


@dataclass(frozen=True)
class LegCost:
    """Itemised cost of a single per-symbol per-leg trade in USD.

    A "leg" is one symbol's buy or sell at one rebalance. The six
    fields sum to :attr:`total`; we keep the breakdown so the SAC
    weekly summary email and audit logs can show *why* a particular
    rebalance was expensive.
    """

    symbol: str
    side: Side
    notional_usd: float
    shares: float
    commission: float
    sec_fee: float
    finra_taf: float
    finra_cat: float
    clearing: float
    pass_through: float

    @property
    def total(self) -> float:
        """Sum of all six cost components in USD."""
        return (
            self.commission
            + self.sec_fee
            + self.finra_taf
            + self.finra_cat
            + self.clearing
            + self.pass_through
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise for logs / audit trail."""
        return {
            "symbol": self.symbol,
            "side": self.side,
            "notional_usd": self.notional_usd,
            "shares": self.shares,
            "commission": self.commission,
            "sec_fee": self.sec_fee,
            "finra_taf": self.finra_taf,
            "finra_cat": self.finra_cat,
            "clearing": self.clearing,
            "pass_through": self.pass_through,
            "total": self.total,
        }


@dataclass(frozen=True)
class RebalanceCost:
    """Aggregate cost of one weekly rebalance across all symbols.

    Used by :class:`PortfolioEnv.step` to convert dollars → return
    fraction (``total_usd / nav_usd``) before subtracting from the
    portfolio log return.
    """

    legs: list[LegCost] = field(default_factory=list)
    nav_usd: float = 0.0

    @property
    def total_usd(self) -> float:
        """Sum of every leg's :attr:`LegCost.total`."""
        return sum(leg.total for leg in self.legs)

    @property
    def total_fraction(self) -> float:
        """Total cost as a fraction of NAV (the value the reward subtracts)."""
        if self.nav_usd <= 0:
            raise ValueError(
                f"nav_usd must be > 0 to compute fraction, got {self.nav_usd}"
            )
        return self.total_usd / self.nav_usd

    def breakdown(self) -> dict[str, float]:
        """Aggregate the six cost components for logging / email summaries."""
        agg = {
            "commission": 0.0,
            "sec_fee": 0.0,
            "finra_taf": 0.0,
            "finra_cat": 0.0,
            "clearing": 0.0,
            "pass_through": 0.0,
        }
        for leg in self.legs:
            agg["commission"] += leg.commission
            agg["sec_fee"] += leg.sec_fee
            agg["finra_taf"] += leg.finra_taf
            agg["finra_cat"] += leg.finra_cat
            agg["clearing"] += leg.clearing
            agg["pass_through"] += leg.pass_through
        return agg


def compute_ibkr_leg_cost(
    symbol: str,
    notional_usd: float,
    shares: float,
    side: Side,
    cfg: IBKRSingaporeCostConfig,
) -> LegCost:
    """Compute IBKR-SG cost for a single per-symbol per-leg trade.

    Math
    ----
    Commission base = ``commission_per_share * shares``. The min
    floor (``commission_min``) is applied first, then the 1%-of-
    notional ceiling (``commission_max_pct * notional_usd``). The
    max-cap takes precedence over the min-floor when they conflict --
    that matches IBKR's published examples for tiny trades where
    the per-order minimum of $0.35 would otherwise exceed 1% of a
    $5 notional. (Test case ``test_max_1pct_cap_binds`` documents
    this branch.)

    Sells get SEC fee + FINRA TAF (capped). Both sides get clearing
    + FINRA CAT + pass-through on commission.

    Args:
        symbol: Stock symbol (kept for the leg's audit trail).
        notional_usd: Trade value in USD (always positive; side
            determines buy vs sell).
        shares: Share count (always positive). Fractional shares are
            allowed; IBKR Pro Tiered supports them at the same per-
            share rate.
        side: ``"buy"`` or ``"sell"``.
        cfg: IBKR-SG rate schedule.

    Returns:
        A fully populated :class:`LegCost`.

    Raises:
        ValueError: if ``shares`` or ``notional_usd`` is non-positive,
            or ``side`` is not ``"buy"``/``"sell"``. Per AGENTS.md
            rule #1, callers must surface these instead of silently
            zero-costing the leg.
    """
    if shares <= 0:
        raise ValueError(
            f"shares must be > 0 for a leg, got {shares} (symbol={symbol!r}); "
            "callers should skip zero-delta symbols entirely instead of "
            "creating a zero-share leg"
        )
    if notional_usd <= 0:
        raise ValueError(
            f"notional_usd must be > 0 for a leg, got {notional_usd} "
            f"(symbol={symbol!r})"
        )
    if side not in ("buy", "sell"):
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")

    # Commission with min floor + 1% max ceiling (max wins when they conflict).
    commission_raw = cfg.commission_per_share * shares
    commission = max(commission_raw, cfg.commission_min)
    commission_cap = cfg.commission_max_pct * notional_usd
    if commission > commission_cap:
        commission = commission_cap

    # Sell-side regulatory.
    if side == "sell":
        sec_fee = cfg.sec_fee_rate * notional_usd
        finra_taf = min(cfg.finra_taf_per_share * shares, cfg.finra_taf_cap)
    else:
        sec_fee = 0.0
        finra_taf = 0.0

    finra_cat = cfg.cat_per_share * shares
    clearing = cfg.clearing_per_share * shares
    pass_through = commission * (
        cfg.nyse_pass_through_rate + cfg.finra_pass_through_rate
    )

    return LegCost(
        symbol=symbol,
        side=side,
        notional_usd=notional_usd,
        shares=shares,
        commission=commission,
        sec_fee=sec_fee,
        finra_taf=finra_taf,
        finra_cat=finra_cat,
        clearing=clearing,
        pass_through=pass_through,
    )


def compute_ibkr_rebalance_cost(
    symbol_order: list[str],
    current_weights: np.ndarray,
    target_weights: np.ndarray,
    prices: np.ndarray,
    cfg: IBKRSingaporeCostConfig,
    *,
    weight_epsilon: float = 0.005,
) -> RebalanceCost:
    """Compute IBKR-SG cost for one weekly rebalance.

    Convention: ``current_weights`` and ``target_weights`` are arrays
    of shape ``(n_stocks + 1,)`` with **CASH as the last element**
    (matching :class:`PortfolioEnv`). CASH is excluded from trading
    cost -- it is just the residual; we only charge for the stock
    legs.

    For each stock symbol with a non-zero weight delta, we:

    1. Convert the delta to dollars: ``delta_usd = |Δw| * NAV``.
    2. Convert dollars to shares using **today's price**:
       ``shares = delta_usd / price``.
    3. Pick a side: positive Δw → buy, negative Δw → sell.
    4. Call :func:`compute_ibkr_leg_cost`.

    Args:
        symbol_order: Ordered list of stock symbols (length
            ``n_stocks``; CASH is implicit at index ``n_stocks``).
        current_weights: Pre-rebalance weights, shape
            ``(n_stocks + 1,)``.
        target_weights: Post-rebalance weights, same shape.
        prices: Today's per-symbol close prices, shape
            ``(n_stocks,)`` (CASH has no price).
        cfg: IBKR-SG rate schedule + ``nav_usd`` anchor.
        weight_epsilon: Numerical-noise threshold below which a delta
            is treated as zero (skip the leg). Avoids creating spam
            $0.35 legs for tiny rounding artefacts in the softmax /
            constraint-enforcement pipeline.

    Returns:
        A :class:`RebalanceCost` aggregating every per-symbol leg.

    Raises:
        ValueError: if ``cfg.nav_usd <= 0``, or shapes are
            inconsistent, or a non-zero-delta symbol has a missing /
            non-positive price (per AGENTS.md rule #1: prefer a hard
            failure to silently free trades).
    """
    if cfg.nav_usd <= 0:
        raise ValueError(f"cfg.nav_usd must be > 0, got {cfg.nav_usd}")

    n_stocks = len(symbol_order)
    if current_weights.shape != (n_stocks + 1,):
        raise ValueError(
            f"current_weights shape {current_weights.shape} does not match "
            f"n_stocks+1 = {n_stocks + 1}"
        )
    if target_weights.shape != (n_stocks + 1,):
        raise ValueError(
            f"target_weights shape {target_weights.shape} does not match "
            f"n_stocks+1 = {n_stocks + 1}"
        )
    if prices.shape != (n_stocks,):
        raise ValueError(
            f"prices shape {prices.shape} does not match n_stocks = {n_stocks}"
        )

    delta = target_weights[:n_stocks] - current_weights[:n_stocks]

    legs: list[LegCost] = []
    for stock_idx in range(n_stocks):
        delta_w = float(delta[stock_idx])
        if abs(delta_w) < weight_epsilon:
            continue

        symbol = symbol_order[stock_idx]
        price = float(prices[stock_idx])
        if not np.isfinite(price) or price <= 0:
            raise ValueError(
                f"price for {symbol!r} must be > 0 and finite to size the "
                f"trade leg, got {price} (delta_w={delta_w}); per AGENTS.md "
                "rule #1 we refuse to silently zero-cost a real trade"
            )

        notional_usd = abs(delta_w) * cfg.nav_usd
        shares = notional_usd / price
        side: Side = "buy" if delta_w > 0 else "sell"

        legs.append(
            compute_ibkr_leg_cost(
                symbol=symbol,
                notional_usd=notional_usd,
                shares=shares,
                side=side,
                cfg=cfg,
            )
        )

    return RebalanceCost(legs=legs, nav_usd=cfg.nav_usd)
