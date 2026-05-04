"""Tests for the ATR-based stop-loss helper.

Covers the formula edge cases that the email path depends on:

* ATR(14)*2 with 5% floor when ATR is small
* 10% ceiling when ATR is large
* "atr_unavailable" sentinel when ATR is missing -- never a flat
  percent fallback per AGENTS.md rule #1
* "sell_no_stop" sentinel for sells (no stop on exits)
"""

import pytest

from brain_api.core.stop_loss import (
    ATR_MULTIPLIER,
    MAX_STOP_PCT,
    MIN_STOP_PCT,
    StopLoss,
    compute_stop_loss,
    stop_loss_for_sell,
)


class TestComputeStopLoss:
    """Per-formula edge-case tests for compute_stop_loss."""

    def test_normal_atr_falls_inside_floor_and_ceiling(self):
        # entry=$100, ATR=$3 -> raw = 6 (6%) which is inside [5%, 10%]
        result = compute_stop_loss(entry_price=100.0, atr_14=3.0)
        assert result.reason == "atr14"
        assert result.price == pytest.approx(94.0)
        assert result.distance_pct == pytest.approx(0.06)

    def test_low_atr_clamped_to_5pct_floor(self):
        # entry=$100, ATR=$0.5 -> raw = 1 (1%) which is below the 5% floor.
        # Should clamp to 5% -> stop at $95.
        result = compute_stop_loss(entry_price=100.0, atr_14=0.5)
        assert result.reason == "atr14"
        assert result.price == pytest.approx(95.0)
        assert result.distance_pct == pytest.approx(MIN_STOP_PCT)

    def test_high_atr_clamped_to_10pct_ceiling(self):
        # entry=$100, ATR=$8 -> raw = 16 (16%) above the 10% ceiling.
        # Should clamp to 10% -> stop at $90.
        result = compute_stop_loss(entry_price=100.0, atr_14=8.0)
        assert result.reason == "atr14"
        assert result.price == pytest.approx(90.0)
        assert result.distance_pct == pytest.approx(MAX_STOP_PCT)

    def test_atr_at_exact_floor_boundary(self):
        # ATR_MULTIPLIER * ATR == 5% * entry -> floor wins ties without
        # double-applying.
        atr = (MIN_STOP_PCT / ATR_MULTIPLIER) * 100  # 2.5
        result = compute_stop_loss(entry_price=100.0, atr_14=atr)
        assert result.reason == "atr14"
        assert result.distance_pct == pytest.approx(MIN_STOP_PCT)

    def test_atr_at_exact_ceiling_boundary(self):
        atr = (MAX_STOP_PCT / ATR_MULTIPLIER) * 100  # 5.0
        result = compute_stop_loss(entry_price=100.0, atr_14=atr)
        assert result.reason == "atr14"
        assert result.distance_pct == pytest.approx(MAX_STOP_PCT)

    def test_none_atr_returns_unavailable_sentinel(self):
        result = compute_stop_loss(entry_price=100.0, atr_14=None)
        assert result == StopLoss(
            price=None, distance_pct=None, reason="atr_unavailable"
        )

    def test_zero_atr_returns_unavailable_sentinel(self):
        # Zero ATR is meaningless -- never silently fall back to a flat
        # percent (AGENTS.md rule #1).
        result = compute_stop_loss(entry_price=100.0, atr_14=0.0)
        assert result.reason == "atr_unavailable"
        assert result.price is None

    def test_negative_atr_returns_unavailable_sentinel(self):
        result = compute_stop_loss(entry_price=100.0, atr_14=-1.0)
        assert result.reason == "atr_unavailable"
        assert result.price is None

    def test_zero_entry_price_returns_unavailable(self):
        result = compute_stop_loss(entry_price=0.0, atr_14=2.0)
        assert result.reason == "atr_unavailable"
        assert result.price is None

    def test_negative_entry_price_returns_unavailable(self):
        result = compute_stop_loss(entry_price=-1.0, atr_14=2.0)
        assert result.reason == "atr_unavailable"

    def test_stop_price_never_negative(self):
        # Even at the 10% ceiling, the stop is always inside (0, entry).
        result = compute_stop_loss(entry_price=100.0, atr_14=1000.0)
        assert result.price is not None
        assert result.price > 0
        assert result.price < 100.0


class TestStopLossForSell:
    """Sell sentinel: exits don't carry a stop-loss reference."""

    def test_sell_returns_no_stop_sentinel(self):
        result = stop_loss_for_sell()
        assert result.price is None
        assert result.distance_pct is None
        assert result.reason == "sell_no_stop"
