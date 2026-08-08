"""Tests for ATR(14) + stop-loss wiring in core/orders.py.

We feed canned OHLC tuples into ``compute_atr_map`` directly so we
don't hit the network. Empty / short-history paths must absent the
symbol from the result (no silent zero-fill per AGENTS.md rule #1).

We also exercise ``generate_orders`` end-to-end with a synthetic
``atr_map`` to assert that each generated ``Order`` carries the
stop-loss fields populated by :func:`brain_api.core.stop_loss.compute_stop_loss`.
This is the integration point that lets the email layer render the
stop-loss column without re-implementing the math.

Finally, ``fetch_ohlc_window`` is exercised against synthetic
``yf.download`` outputs to assert that NaN rows in any of H/L/C
remove the **whole row** (joint ``dropna``) -- per-column dropna
would silently misalign columns and break the True Range math.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from brain_api.core.orders import (
    ATR_PERIOD,
    PortfolioInput,
    PositionInput,
    compute_atr_map,
    convert_weights_to_whole_shares,
    fetch_ohlc_window,
    generate_orders,
)


def _flat_bars(
    n: int, high: float, low: float, close: float
) -> list[tuple[float, float, float]]:
    """Synthetic flat OHLC bars used to exercise the smoothing seed."""
    return [(high, low, close) for _ in range(n)]


class TestComputeATRMap:
    def test_empty_input_returns_empty(self):
        assert compute_atr_map({}) == {}

    def test_short_history_silently_omits_symbol(self):
        # ATR(14) needs >= 15 bars; 14 bars must NOT produce a fake-zero
        # entry -- the email shows "atr_unavailable" instead.
        ohlc = {"AAPL": _flat_bars(ATR_PERIOD, 102.0, 98.0, 100.0)}
        assert compute_atr_map(ohlc) == {}

    def test_constant_range_yields_constant_atr(self):
        # 30 identical bars with HL range = $4 -> every TR is 4 (since
        # close_{t-1} == close_t the gap term doesn't dominate). Wilder
        # smoothing of a constant series equals the constant.
        ohlc = {"AAPL": _flat_bars(30, 104.0, 100.0, 102.0)}
        atrs = compute_atr_map(ohlc)
        assert "AAPL" in atrs
        assert atrs["AAPL"] == pytest.approx(4.0)

    def test_uses_gap_term_when_close_jumps(self):
        # Construct a series where day-over-day close gap exceeds the
        # bar's HL range, so |high - prev_close| or |low - prev_close|
        # drives TR. ATR should reflect the gap, not just HL.
        bars: list[tuple[float, float, float]] = []
        # 20 bars with a structural gap of +5 between every close
        prev_close = 100.0
        for _ in range(20):
            high = prev_close + 6.0
            low = prev_close + 4.0
            close = prev_close + 5.0
            bars.append((high, low, close))
            prev_close = close
        atrs = compute_atr_map({"AAPL": bars})
        # TR_t for these bars = max(2, |6|, |4|) = 6 -> ATR converges to 6.
        assert atrs["AAPL"] == pytest.approx(6.0, rel=1e-3)

    def test_zero_atr_silently_omitted(self):
        # If smoothing collapses to exactly zero (bars all identical
        # with H==L==C), the symbol is omitted rather than emitting a
        # zero ATR that would round-trip as an unusable stop reference.
        ohlc = {"AAPL": [(100.0, 100.0, 100.0)] * 20}
        assert compute_atr_map(ohlc) == {}

    def test_multiple_symbols_independent(self):
        ohlc = {
            "AAPL": _flat_bars(30, 102.0, 98.0, 100.0),
            "MSFT": _flat_bars(30, 210.0, 190.0, 200.0),
        }
        atrs = compute_atr_map(ohlc)
        assert atrs["AAPL"] == pytest.approx(4.0)
        assert atrs["MSFT"] == pytest.approx(20.0)

    def test_short_symbol_omitted_alongside_long_symbol(self):
        ohlc = {
            "AAPL": _flat_bars(30, 102.0, 98.0, 100.0),  # has enough
            "TINY": _flat_bars(5, 10.0, 9.0, 9.5),  # too short
        }
        atrs = compute_atr_map(ohlc)
        assert "AAPL" in atrs
        assert "TINY" not in atrs

    def test_atr_period_parameter_respected(self):
        # period=5 needs >= 6 bars
        ohlc = {"AAPL": _flat_bars(6, 102.0, 98.0, 100.0)}
        atrs = compute_atr_map(ohlc, period=5)
        assert "AAPL" in atrs
        # Same flat bars: ATR == constant TR == 4.
        assert atrs["AAPL"] == pytest.approx(4.0)


class TestGenerateOrdersStopLossFields:
    """``generate_orders`` populates stop-loss on each Order.

    The stop-loss math itself is unit-tested in test_stop_loss.py;
    here we only assert the wiring (each Order carries the three
    fields, sells get the sentinel, buys without ATR get
    ``"atr_unavailable"`` and never a flat-percent fallback).
    """

    def test_buy_with_atr_carries_atr14_stop(self):
        target_weights = {"AAPL": 1.0, "CASH": 0.0}
        portfolio = PortfolioInput(cash=10000.0, positions=[])
        result = generate_orders(
            target_weights=target_weights,
            portfolio=portfolio,
            run_id="paper:2026-04-27",
            attempt=1,
            algorithm="alpha_hrp",
            prices={"AAPL": 100.0},
            atr_map={"AAPL": 3.0},
        )
        assert len(result.orders) == 1
        order = result.orders[0]
        assert order.side == "buy"
        assert order.stop_loss_reason == "atr14"
        # ATR=3 -> raw=6 (6%) inside [5%, 10%] -> stop at $94.
        assert order.stop_loss_price == pytest.approx(94.0)
        assert order.stop_loss_distance_pct == pytest.approx(0.06)

    def test_buy_without_atr_carries_unavailable_sentinel(self):
        target_weights = {"AAPL": 1.0, "CASH": 0.0}
        portfolio = PortfolioInput(cash=10000.0, positions=[])
        result = generate_orders(
            target_weights=target_weights,
            portfolio=portfolio,
            run_id="paper:2026-04-27",
            attempt=1,
            algorithm="alpha_hrp",
            prices={"AAPL": 100.0},
            atr_map={},  # no ATR -> never substitute a flat percent
        )
        assert len(result.orders) == 1
        order = result.orders[0]
        assert order.side == "buy"
        assert order.stop_loss_reason == "atr_unavailable"
        assert order.stop_loss_price is None
        assert order.stop_loss_distance_pct is None

    def test_sell_carries_sell_no_stop_sentinel(self):
        target_weights = {"AAPL": 0.0, "CASH": 1.0}
        portfolio = PortfolioInput(
            cash=0.0,
            positions=[PositionInput(symbol="AAPL", qty=10.0, market_value=1000.0)],
        )
        result = generate_orders(
            target_weights=target_weights,
            portfolio=portfolio,
            run_id="paper:2026-04-27",
            attempt=1,
            algorithm="alpha_hrp",
            prices={"AAPL": 100.0},
            atr_map={"AAPL": 3.0},  # ATR is irrelevant on the sell side
        )
        assert len(result.orders) == 1
        order = result.orders[0]
        assert order.side == "sell"
        assert order.stop_loss_reason == "sell_no_stop"
        assert order.stop_loss_price is None
        assert order.stop_loss_distance_pct is None

    def test_to_dict_includes_stop_loss_fields(self):
        target_weights = {"AAPL": 1.0, "CASH": 0.0}
        portfolio = PortfolioInput(cash=10000.0, positions=[])
        result = generate_orders(
            target_weights=target_weights,
            portfolio=portfolio,
            run_id="paper:2026-04-27",
            attempt=1,
            algorithm="alpha_hrp",
            prices={"AAPL": 100.0},
            atr_map={"AAPL": 3.0},
        )
        payload = result.orders[0].to_dict()
        assert payload["stop_loss_reason"] == "atr14"
        assert payload["stop_loss_price"] == pytest.approx(94.0)
        assert payload["stop_loss_distance_pct"] == pytest.approx(0.06)


class TestConvertWeightsStopLossFields:
    """``convert_weights_to_whole_shares`` populates stop-loss on each row.

    India Stage 2 emails read these fields from paper_allocation; the
    math is unit-tested in test_stop_loss.py -- here we only assert
    wiring (injected atr_map, missing ATR sentinel, .NS symbols).
    """

    def test_row_with_atr_carries_atr14_stop(self):
        result = convert_weights_to_whole_shares(
            percentage_weights={"RELIANCE.NS": 50.0},
            total_nav=1_000_000.0,
            prices={"RELIANCE.NS": 100.0},
            atr_map={"RELIANCE.NS": 3.0},
        )
        assert len(result.details) == 1
        detail = result.details[0]
        assert detail.symbol == "RELIANCE.NS"
        assert detail.stop_loss_reason == "atr14"
        # ATR=3 -> raw=6 (6%) inside [5%, 10%] -> stop at 94.
        assert detail.stop_loss_price == pytest.approx(94.0)
        assert detail.stop_loss_distance_pct == pytest.approx(0.06)

    def test_row_without_atr_carries_unavailable_sentinel(self):
        result = convert_weights_to_whole_shares(
            percentage_weights={"TCS.NS": 50.0},
            total_nav=1_000_000.0,
            prices={"TCS.NS": 100.0},
            atr_map={},  # no ATR -> never substitute a flat percent
        )
        assert len(result.details) == 1
        detail = result.details[0]
        assert detail.stop_loss_reason == "atr_unavailable"
        assert detail.stop_loss_price is None
        assert detail.stop_loss_distance_pct is None

    def test_mixed_atr_availability_per_symbol(self):
        result = convert_weights_to_whole_shares(
            percentage_weights={"HAS.NS": 40.0, "MISS.NS": 40.0},
            total_nav=1_000_000.0,
            prices={"HAS.NS": 100.0, "MISS.NS": 200.0},
            atr_map={"HAS.NS": 3.0},
        )
        by_symbol = {d.symbol: d for d in result.details}
        assert by_symbol["HAS.NS"].stop_loss_reason == "atr14"
        assert by_symbol["HAS.NS"].stop_loss_price == pytest.approx(94.0)
        assert by_symbol["MISS.NS"].stop_loss_reason == "atr_unavailable"
        assert by_symbol["MISS.NS"].stop_loss_price is None


class TestFetchOhlcWindowNaNAlignment:
    """Joint ``dropna`` keeps every emitted (H, L, C) tuple on the same date.

    Independent per-column ``dropna`` would silently shift columns
    against each other when any one of H/L/C is NaN for a date, and
    the downstream True-Range computation
    ``max(high - low, |high - prev_close|, |low - prev_close|)`` would
    pair the wrong day's bars and emit a fictional TR.
    """

    def _build_multi_symbol_frame(self) -> pd.DataFrame:
        """Two symbols, one with a NaN High on day 2.

        Date layout (oldest first):
            2026-04-23: AAPL H=102 L=98  C=100  | MSFT H=210 L=200 C=205
            2026-04-24: AAPL H=NaN L=99  C=101  | MSFT H=212 L=201 C=208  <- NaN
            2026-04-25: AAPL H=104 L=100 C=103  | MSFT H=213 L=205 C=210
            2026-04-26: AAPL H=105 L=101 C=104  | MSFT H=214 L=206 C=212

        After joint dropna on AAPL columns, day 2 must be removed
        from AAPL entirely (so close=101 is dropped too). MSFT must
        be unaffected because its own columns are clean.
        """
        index = pd.to_datetime(["2026-04-23", "2026-04-24", "2026-04-25", "2026-04-26"])
        cols = pd.MultiIndex.from_product([["High", "Low", "Close"], ["AAPL", "MSFT"]])
        df = pd.DataFrame(
            [
                [102.0, 210.0, 98.0, 200.0, 100.0, 205.0],
                [float("nan"), 212.0, 99.0, 201.0, 101.0, 208.0],
                [104.0, 213.0, 100.0, 205.0, 103.0, 210.0],
                [105.0, 214.0, 101.0, 206.0, 104.0, 212.0],
            ],
            index=index,
            columns=cols,
        )
        return df

    def test_multi_symbol_drops_nan_row_from_one_symbol_only(self):
        df = self._build_multi_symbol_frame()
        with patch("brain_api.core.orders.yf.download", return_value=df):
            bars = fetch_ohlc_window(["AAPL", "MSFT"])

        # AAPL: day 2 (NaN High) is removed entirely. The surviving 3
        # rows must all carry the close from the SAME date as the H/L,
        # so the close 101 (day 2) must NOT appear in any tuple.
        aapl_closes = [c for _h, _l, c in bars["AAPL"]]
        assert 101.0 not in aapl_closes
        assert aapl_closes == [100.0, 103.0, 104.0]
        # H/L/C alignment per row:
        assert bars["AAPL"] == [
            (102.0, 98.0, 100.0),
            (104.0, 100.0, 103.0),
            (105.0, 101.0, 104.0),
        ]
        # MSFT rows untouched because none of its columns had a NaN.
        assert bars["MSFT"] == [
            (210.0, 200.0, 205.0),
            (212.0, 201.0, 208.0),
            (213.0, 205.0, 210.0),
            (214.0, 206.0, 212.0),
        ]

    def test_single_symbol_drops_nan_row(self):
        index = pd.to_datetime(["2026-04-23", "2026-04-24", "2026-04-25", "2026-04-26"])
        df = pd.DataFrame(
            {
                "High": [102.0, float("nan"), 104.0, 105.0],
                "Low": [98.0, 99.0, 100.0, 101.0],
                "Close": [100.0, 101.0, 103.0, 104.0],
            },
            index=index,
        )
        with patch("brain_api.core.orders.yf.download", return_value=df):
            bars = fetch_ohlc_window(["AAPL"])

        # Day 2's close (101) must be dropped along with the NaN high.
        assert bars["AAPL"] == [
            (102.0, 98.0, 100.0),
            (104.0, 100.0, 103.0),
            (105.0, 101.0, 104.0),
        ]
