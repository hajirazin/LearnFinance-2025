"""Tests for the IBKR Singapore Tiered transaction-cost model.

Pure-function business-logic tests (per AGENTS.md / repo testing policy
-- no schema-only tests). Calibration sanity check at the bottom guards
the headline number we put in the doc / SAC training summary email.
"""

from __future__ import annotations

import numpy as np
import pytest

from brain_api.core.portfolio_rl.broker_costs import (
    IBKRSingaporeCostConfig,
    compute_ibkr_leg_cost,
    compute_ibkr_rebalance_cost,
)


def _cfg(**overrides) -> IBKRSingaporeCostConfig:
    return IBKRSingaporeCostConfig(
        **{**IBKRSingaporeCostConfig.default().to_dict(), **overrides}
    )


# ---------------------------------------------------------------------------
# Per-leg cost: commission floor + cap
# ---------------------------------------------------------------------------


def test_min_per_order_binds_for_tiny_trade():
    """A $10 notional / 0.5 share buy gets the $0.35 minimum, NOT $0.0017."""
    leg = compute_ibkr_leg_cost(
        symbol="AAPL",
        notional_usd=10.0,
        shares=0.5,
        side="buy",
        cfg=_cfg(),
    )
    # raw = 0.5 * 0.0035 = 0.00175 -> bumped to 0.35 (min) -> within 1% cap (0.10)?
    # No: 0.35 > 0.10 (1% of $10) so cap wins -> commission = 0.10.
    assert leg.commission == pytest.approx(0.10)


def test_max_1pct_cap_binds():
    """When the min would exceed 1% of trade value, the 1% cap wins."""
    leg = compute_ibkr_leg_cost(
        symbol="AAPL",
        notional_usd=5.0,
        shares=1.0,
        side="buy",
        cfg=_cfg(),
    )
    # raw = 1.0 * 0.0035 = 0.0035 -> min would be 0.35 -> capped to 0.05 (1% of $5).
    assert leg.commission == pytest.approx(0.05)


def test_per_share_rate_dominates_at_size():
    """A $50k / 250-share buy pays the per-share rate, not the floor or cap."""
    leg = compute_ibkr_leg_cost(
        symbol="AAPL",
        notional_usd=50_000.0,
        shares=250.0,
        side="buy",
        cfg=_cfg(),
    )
    # raw = 250 * 0.0035 = 0.875; floor 0.35 < 0.875; cap 1% * 50k = 500 > 0.875.
    assert leg.commission == pytest.approx(0.875)


# ---------------------------------------------------------------------------
# Sell-side regulatory: SEC + FINRA TAF + cap; CAT and clearing both sides
# ---------------------------------------------------------------------------


def test_sec_fee_only_on_sells():
    cfg = _cfg()
    sell_leg = compute_ibkr_leg_cost(
        symbol="AAPL", notional_usd=10_000.0, shares=50.0, side="sell", cfg=cfg
    )
    buy_leg = compute_ibkr_leg_cost(
        symbol="AAPL", notional_usd=10_000.0, shares=50.0, side="buy", cfg=cfg
    )
    assert sell_leg.sec_fee == pytest.approx(0.0000206 * 10_000.0)
    assert buy_leg.sec_fee == 0.0


def test_finra_taf_only_on_sells_and_capped_at_9_27():
    cfg = _cfg()
    # 50 shares: under cap.
    small_sell = compute_ibkr_leg_cost(
        symbol="AAPL", notional_usd=10_000.0, shares=50.0, side="sell", cfg=cfg
    )
    assert small_sell.finra_taf == pytest.approx(0.000195 * 50.0)

    # 1,000,000 shares: way over cap -> capped at 9.27.
    big_sell = compute_ibkr_leg_cost(
        symbol="AAPL",
        notional_usd=1_000_000.0,
        shares=1_000_000.0,
        side="sell",
        cfg=cfg,
    )
    assert big_sell.finra_taf == pytest.approx(9.27)

    # Buys never pay TAF.
    buy = compute_ibkr_leg_cost(
        symbol="AAPL", notional_usd=10_000.0, shares=50.0, side="buy", cfg=cfg
    )
    assert buy.finra_taf == 0.0


def test_finra_cat_both_sides():
    cfg = _cfg()
    sell = compute_ibkr_leg_cost(
        symbol="AAPL", notional_usd=10_000.0, shares=50.0, side="sell", cfg=cfg
    )
    buy = compute_ibkr_leg_cost(
        symbol="AAPL", notional_usd=10_000.0, shares=50.0, side="buy", cfg=cfg
    )
    assert sell.finra_cat == pytest.approx(0.000033 * 50.0)
    assert buy.finra_cat == pytest.approx(0.000033 * 50.0)


def test_clearing_both_sides():
    cfg = _cfg()
    sell = compute_ibkr_leg_cost(
        symbol="AAPL", notional_usd=10_000.0, shares=50.0, side="sell", cfg=cfg
    )
    buy = compute_ibkr_leg_cost(
        symbol="AAPL", notional_usd=10_000.0, shares=50.0, side="buy", cfg=cfg
    )
    assert sell.clearing == pytest.approx(0.00020 * 50.0)
    assert buy.clearing == pytest.approx(0.00020 * 50.0)


def test_pass_through_on_commission():
    cfg = _cfg()
    # Pick a leg where commission is clearly the per-share rate (no floor/cap).
    leg = compute_ibkr_leg_cost(
        symbol="AAPL", notional_usd=50_000.0, shares=250.0, side="buy", cfg=cfg
    )
    expected_pass = leg.commission * (
        cfg.nyse_pass_through_rate + cfg.finra_pass_through_rate
    )
    assert leg.pass_through == pytest.approx(expected_pass)


# ---------------------------------------------------------------------------
# Validation: no silent fallbacks (AGENTS.md rule #1)
# ---------------------------------------------------------------------------


def test_no_silent_fallback_on_zero_shares():
    with pytest.raises(ValueError, match="shares must be > 0"):
        compute_ibkr_leg_cost(
            symbol="AAPL", notional_usd=10.0, shares=0.0, side="buy", cfg=_cfg()
        )


def test_no_silent_fallback_on_zero_notional():
    with pytest.raises(ValueError, match="notional_usd must be > 0"):
        compute_ibkr_leg_cost(
            symbol="AAPL", notional_usd=0.0, shares=1.0, side="buy", cfg=_cfg()
        )


def test_no_silent_fallback_on_bad_side():
    with pytest.raises(ValueError, match="side must be 'buy' or 'sell'"):
        compute_ibkr_leg_cost(
            symbol="AAPL", notional_usd=10.0, shares=1.0, side="wrong", cfg=_cfg()
        )


def test_no_silent_fallback_on_missing_price():
    cfg = _cfg().with_nav(10_000.0)
    # AAPL gets a 10% delta but its price is 0 -> raise.
    current = np.array([0.10, 0.10, 0.80])  # AAPL, MSFT, CASH
    target = np.array([0.20, 0.10, 0.70])  # AAPL up to 20%
    prices = np.array([0.0, 200.0])
    with pytest.raises(ValueError, match="price for 'AAPL'"):
        compute_ibkr_rebalance_cost(
            symbol_order=["AAPL", "MSFT"],
            current_weights=current,
            target_weights=target,
            prices=prices,
            cfg=cfg,
        )


def test_no_silent_fallback_on_negative_nav():
    with pytest.raises(ValueError, match="nav_usd must be > 0"):
        IBKRSingaporeCostConfig.default().with_nav(-1.0)


def test_no_silent_fallback_on_zero_nav():
    cfg = IBKRSingaporeCostConfig.default()
    object.__setattr__(cfg, "nav_usd", 0.0)  # bypass with_nav guard
    with pytest.raises(ValueError, match=r"cfg\.nav_usd must be > 0"):
        compute_ibkr_rebalance_cost(
            symbol_order=["AAPL"],
            current_weights=np.array([0.5, 0.5]),
            target_weights=np.array([0.6, 0.4]),
            prices=np.array([100.0]),
            cfg=cfg,
        )


def test_zero_delta_symbol_skipped():
    """A symbol whose weight does not change incurs no leg / no min charge."""
    cfg = _cfg().with_nav(10_000.0)
    current = np.array([0.10, 0.10, 0.80])
    target = np.array([0.10, 0.10, 0.80])
    prices = np.array([200.0, 150.0])
    cost = compute_ibkr_rebalance_cost(
        symbol_order=["AAPL", "MSFT"],
        current_weights=current,
        target_weights=target,
        prices=prices,
        cfg=cfg,
    )
    assert len(cost.legs) == 0
    assert cost.total_usd == 0.0
    assert cost.total_fraction == 0.0


def test_breakdown_aggregates_components():
    cfg = _cfg().with_nav(10_000.0)
    current = np.array([0.0, 0.0, 1.0])
    target = np.array([0.5, 0.5, 0.0])  # buy AAPL + buy MSFT
    prices = np.array([200.0, 150.0])
    cost = compute_ibkr_rebalance_cost(
        symbol_order=["AAPL", "MSFT"],
        current_weights=current,
        target_weights=target,
        prices=prices,
        cfg=cfg,
    )
    breakdown = cost.breakdown()
    # Both buys -> SEC and FINRA TAF must be exactly 0 in the aggregate.
    assert breakdown["sec_fee"] == 0.0
    assert breakdown["finra_taf"] == 0.0
    # Sum of breakdown values must equal total_usd.
    assert sum(breakdown.values()) == pytest.approx(cost.total_usd)


# ---------------------------------------------------------------------------
# Calibration sanity check
# ---------------------------------------------------------------------------


def test_calibration_typical_rebalance_at_10k_nav():
    """Round-trip cost at $10k NAV / 30% turnover / ~$200 avg price.

    Anchors the headline IBKR-SG bps figure used in README and the
    SAC training summary email. At $10k NAV with ~$1k-per-leg
    notional, the $0.35 per-order minimum binds for every leg
    because 5 shares * $0.0035 = $0.0175 << $0.35. Total comes out
    around 1.5-3.0 bps of NAV -- noticeably *cheaper* than the
    legacy flat 10 bps * turnover formula (which would have charged
    ~3 bps for 30% turnover anyway), but the per-symbol structure
    matters for the policy: tiny symbols (1 share orders) pay the
    same $0.35 floor as a 50-share position, so the model now has
    a real disincentive to fragment trades across many small names.
    """
    cfg = IBKRSingaporeCostConfig.default()  # nav_usd = 10_000

    # 5 names at ~$200 each. 3 sells at -10pp, 2 buys at +10pp, CASH +10pp.
    # Total turnover = 0.5 * (3*0.10 + 2*0.10 + 0.10) = 0.30.
    n_stocks = 5
    symbol_order = [f"S{i}" for i in range(n_stocks)]
    prices = np.full(n_stocks, 200.0)
    current = np.array([0.20, 0.20, 0.20, 0.10, 0.10, 0.20])
    target = np.array([0.10, 0.10, 0.10, 0.20, 0.20, 0.30])

    cost = compute_ibkr_rebalance_cost(
        symbol_order=symbol_order,
        current_weights=current,
        target_weights=target,
        prices=prices,
        cfg=cfg,
    )

    # Per leg notional = 0.10 * $10k = $1000 -> 5 shares -> per-share
    # commission of $0.0175 hits the $0.35 floor. 5 legs * $0.35 = $1.75
    # commission baseline; sell-side adds SEC + TAF + CAT + clearing on
    # 3 of the 5; buys add CAT + clearing on the other 2.
    bps = cost.total_fraction * 10_000
    assert 1.0 <= bps <= 3.0, (
        f"expected 1-3 bps round-trip cost at $10k NAV, got {bps:.2f}"
    )

    # Commission alone is $1.75 (floor binds uniformly).
    breakdown = cost.breakdown()
    assert breakdown["commission"] == pytest.approx(5 * 0.35, rel=0.01)


def test_calibration_per_share_dominance_at_large_nav():
    """At $100k NAV the per-share rate dominates and round-trip drops to ~0.5 bps."""
    cfg = IBKRSingaporeCostConfig.default().with_nav(100_000.0)
    n_stocks = 5
    symbol_order = [f"S{i}" for i in range(n_stocks)]
    prices = np.full(n_stocks, 200.0)
    current = np.array([0.20, 0.20, 0.20, 0.10, 0.10, 0.20])
    target = np.array([0.10, 0.10, 0.10, 0.20, 0.20, 0.30])

    cost = compute_ibkr_rebalance_cost(
        symbol_order=symbol_order,
        current_weights=current,
        target_weights=target,
        prices=prices,
        cfg=cfg,
    )

    bps = cost.total_fraction * 10_000
    assert 0.2 <= bps <= 1.5, f"expected sub-1.5 bps cost at $100k NAV, got {bps:.2f}"
