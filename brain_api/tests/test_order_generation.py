"""Tests for order generation logic.

This module tests:
- Market order generation from target weights
- Sell qty capped at position quantity
- Client order ID generation
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from brain_api.core.orders import (
    OrderPriceError,
    PortfolioInput,
    PositionInput,
    generate_orders,
)
from brain_api.main import app

client = TestClient(app)


class TestOrderGeneration:
    """Tests for market order generation from target weights."""

    @pytest.fixture(autouse=True)
    def synthetic_atr_history(self, monkeypatch):
        monkeypatch.setattr(
            "brain_api.core.orders.fetch_ohlc_window",
            lambda symbols: {symbol: [(101.0, 99.0, 100.0)] * 20 for symbol in symbols},
        )

    def test_generate_buy_orders_from_cash(self):
        """Buy orders sized at market price."""
        target_weights = {"AAPL": 0.5, "CASH": 0.5}
        portfolio = PortfolioInput(
            cash=10000.0,
            positions=[],
        )
        mock_prices = {"AAPL": 100.0}

        with patch(
            "brain_api.core.orders.fetch_current_prices", return_value=mock_prices
        ):
            result = generate_orders(
                target_weights=target_weights,
                portfolio=portfolio,
                run_id="paper:2026-01-20",
                attempt=1,
                algorithm="sac",
            )

        assert len(result.orders) == 1
        order = result.orders[0]
        assert order.side == "buy"
        assert order.symbol == "AAPL"
        expected_qty = round(5000.0 / 100.0, 4)
        assert order.qty == expected_qty

    @pytest.mark.parametrize("prices", [{}, {"AAPL": 0.0}, {"AAPL": float("nan")}])
    def test_material_leg_requires_finite_positive_price(self, prices):
        with pytest.raises(OrderPriceError, match=r"missing_or_invalid=\['AAPL'\]"):
            generate_orders(
                target_weights={"AAPL": 0.5, "CASH": 0.5},
                portfolio=PortfolioInput(cash=1000.0, positions=[]),
                run_id="paper:2026-01-20",
                attempt=1,
                algorithm="sac",
                prices=prices,
                atr_map={},
            )

    def test_missing_price_is_allowed_for_non_material_leg(self):
        result = generate_orders(
            target_weights={"AAPL": 0.005, "CASH": 0.995},
            portfolio=PortfolioInput(cash=1000.0, positions=[]),
            run_id="paper:2026-01-20",
            attempt=1,
            algorithm="sac",
            prices={},
            atr_map={},
        )

        assert result.orders == []
        assert result.summary.skipped_below_threshold == 1

    def test_missing_price_is_allowed_for_below_minimum_full_exit(self):
        """Sub-$10 full exit still sells exact broker qty with no Yahoo price."""
        result = generate_orders(
            target_weights={"AAPL": 0.0, "CASH": 1.0},
            portfolio=PortfolioInput(
                cash=995.0,
                positions=[PositionInput(symbol="AAPL", qty=0.05, market_value=5.0)],
            ),
            run_id="paper:2026-01-20",
            attempt=1,
            algorithm="sac",
            prices={},
            atr_map={},
        )

        assert len(result.orders) == 1
        order = result.orders[0]
        assert order.side == "sell"
        assert order.symbol == "AAPL"
        assert order.qty == 0.05
        assert result.summary.skipped_small_orders == 0

    def test_full_exit_with_price_sells_exact_qty_below_ten_dollars(self):
        """AMD-class dust: sub-$10 full exit with a mark still flattens."""
        result = generate_orders(
            target_weights={"AMD": 0.0, "CASH": 1.0},
            portfolio=PortfolioInput(
                cash=993.66,
                positions=[PositionInput(symbol="AMD", qty=0.0134, market_value=6.34)],
            ),
            run_id="paper:2026-01-20",
            attempt=1,
            algorithm="sac",
            prices={"AMD": 473.25},
            atr_map={},
        )

        assert len(result.orders) == 1
        assert result.orders[0].side == "sell"
        assert result.orders[0].qty == 0.0134

    def test_full_exit_uses_position_qty_not_notional_over_yahoo(self):
        """CRWD-class: Yahoo mark above Alpaca MV must not undershoot flatten qty."""
        position_qty = 25.6862
        alpaca_mv = 2568.62
        yahoo_price = 100.4  # 0.4% above the Alpaca mark of $100
        result = generate_orders(
            target_weights={"CRWD": 0.0, "CASH": 1.0},
            portfolio=PortfolioInput(
                cash=7431.38,
                positions=[
                    PositionInput(
                        symbol="CRWD", qty=position_qty, market_value=alpaca_mv
                    )
                ],
            ),
            run_id="paper:2026-01-20",
            attempt=1,
            algorithm="sac",
            prices={"CRWD": yahoo_price},
            atr_map={},
        )

        assert len(result.orders) == 1
        assert result.orders[0].qty == position_qty
        undershot = round(alpaca_mv / yahoo_price, 4)
        assert result.orders[0].qty != undershot

    def test_full_exit_does_not_round_qty_to_four_decimals(self):
        """Broker qty longer than 4 decimals is not truncated on flatten."""
        result = generate_orders(
            target_weights={"SNDK": 0.0, "CASH": 1.0},
            portfolio=PortfolioInput(
                cash=9999.84,
                positions=[
                    PositionInput(symbol="SNDK", qty=0.00012, market_value=0.16)
                ],
            ),
            run_id="paper:2026-01-20",
            attempt=1,
            algorithm="sac",
            prices={},
            atr_map={},
        )

        assert len(result.orders) == 1
        assert result.orders[0].qty == 0.00012
        assert result.orders[0].qty != round(0.00012, 4)

    def test_buy_only_qty_still_uses_notional_over_price_rounded_four_dp(self):
        """Buy path is unchanged: trade_value / price rounded to 4 decimals."""
        result = generate_orders(
            target_weights={"AAPL": 0.5, "CASH": 0.5},
            portfolio=PortfolioInput(cash=10000.0, positions=[]),
            run_id="paper:2026-01-20",
            attempt=1,
            algorithm="sac",
            prices={"AAPL": 150.0},
            atr_map={},
        )

        assert len(result.orders) == 1
        assert result.orders[0].side == "buy"
        assert result.orders[0].qty == round(5000.0 / 150.0, 4)

    def test_full_exit_omitted_when_order_side_is_buy(self):
        """Sell-wait-buy: the buy phase must not emit a flatten sell."""
        result = generate_orders(
            target_weights={"AAPL": 0.0, "CASH": 1.0},
            portfolio=PortfolioInput(
                cash=0.0,
                positions=[PositionInput(symbol="AAPL", qty=10.0, market_value=1000.0)],
            ),
            run_id="paper:2026-01-20",
            attempt=1,
            algorithm="sac",
            order_side="buy",
            prices={},
            atr_map={},
        )

        assert result.orders == []

    def test_full_exit_emitted_when_order_side_is_sell(self):
        result = generate_orders(
            target_weights={"AAPL": 0.0, "CASH": 1.0},
            portfolio=PortfolioInput(
                cash=0.0,
                positions=[PositionInput(symbol="AAPL", qty=10.0, market_value=1000.0)],
            ),
            run_id="paper:2026-01-20",
            attempt=1,
            algorithm="sac",
            order_side="sell",
            prices={},
            atr_map={},
        )

        assert len(result.orders) == 1
        assert result.orders[0].side == "sell"
        assert result.orders[0].qty == 10.0

    def test_generate_sell_orders_to_cash(self):
        """Full position sell: qty capped at position qty."""
        target_weights = {"AAPL": 0.0, "CASH": 1.0}
        portfolio = PortfolioInput(
            cash=0.0,
            positions=[
                PositionInput(symbol="AAPL", qty=100, market_value=10000.0),
            ],
        )
        mock_prices = {"AAPL": 100.0}

        with patch(
            "brain_api.core.orders.fetch_current_prices", return_value=mock_prices
        ):
            result = generate_orders(
                target_weights=target_weights,
                portfolio=portfolio,
                run_id="paper:2026-01-20",
                attempt=1,
                algorithm="sac",
            )

        assert len(result.orders) == 1
        order = result.orders[0]
        assert order.side == "sell"
        assert order.symbol == "AAPL"
        assert order.qty == 100.0

    def test_orders_are_market_type(self):
        """All generated orders use market type with no limit price."""
        target_weights = {"AAPL": 0.5, "MSFT": 0.3, "CASH": 0.2}
        portfolio = PortfolioInput(
            cash=10000.0,
            positions=[],
        )
        mock_prices = {"AAPL": 150.0, "MSFT": 300.0}

        with patch(
            "brain_api.core.orders.fetch_current_prices", return_value=mock_prices
        ):
            result = generate_orders(
                target_weights=target_weights,
                portfolio=portfolio,
                run_id="paper:2026-01-20",
                attempt=1,
                algorithm="sac",
            )

        for order in result.orders:
            assert order.order_type == "market"
            assert order.limit_price is None

    def test_buy_qty_sized_for_market_price(self):
        """Buy qty = trade_value / market_price."""
        target_weights = {"AAPL": 0.8, "CASH": 0.2}
        portfolio = PortfolioInput(
            cash=10000.0,
            positions=[],
        )
        mock_prices = {"AAPL": 100.0}

        with patch(
            "brain_api.core.orders.fetch_current_prices", return_value=mock_prices
        ):
            result = generate_orders(
                target_weights=target_weights,
                portfolio=portfolio,
                run_id="paper:2026-01-20",
                attempt=1,
                algorithm="sac",
            )

        order = result.orders[0]
        trade_value = 0.8 * 10000.0
        expected_qty = round(trade_value / 100.0, 4)
        assert order.qty == expected_qty

    def test_sell_qty_uses_market_price(self):
        """Partial sell: qty = trade_value / market_price."""
        target_weights = {"AAPL": 0.2, "CASH": 0.8}
        portfolio = PortfolioInput(
            cash=2000.0,
            positions=[
                PositionInput(symbol="AAPL", qty=80, market_value=8000.0),
            ],
        )
        mock_prices = {"AAPL": 100.0}

        with patch(
            "brain_api.core.orders.fetch_current_prices", return_value=mock_prices
        ):
            result = generate_orders(
                target_weights=target_weights,
                portfolio=portfolio,
                run_id="paper:2026-01-20",
                attempt=1,
                algorithm="sac",
            )

        order = result.orders[0]
        assert order.side == "sell"
        trade_value = 0.6 * 10000.0  # 80% -> 20%
        expected_qty = round(trade_value / 100.0, 4)
        assert order.qty == expected_qty

    def test_sell_qty_capped_at_position_qty(self):
        """Sell qty never exceeds actual position held."""
        target_weights = {"AAPL": 0.0, "CASH": 1.0}
        portfolio = PortfolioInput(
            cash=500.0,
            positions=[
                PositionInput(symbol="AAPL", qty=50, market_value=5000.0),
            ],
        )
        mock_prices = {"AAPL": 100.0}

        with patch(
            "brain_api.core.orders.fetch_current_prices", return_value=mock_prices
        ):
            result = generate_orders(
                target_weights=target_weights,
                portfolio=portfolio,
                run_id="paper:2026-01-20",
                attempt=1,
                algorithm="sac",
            )

        order = result.orders[0]
        assert order.side == "sell"
        assert order.qty <= 50.0

    def test_full_rebalance_sell_qty_matches_position(self):
        """Full rebalance: sell qty = position qty (not inflated)."""
        target_weights = {"MSFT": 0.49, "GOOGL": 0.49, "CASH": 0.02}
        portfolio = PortfolioInput(
            cash=200.0,
            positions=[
                PositionInput(symbol="AAPL", qty=49, market_value=4900.0),
                PositionInput(symbol="TSLA", qty=49, market_value=4900.0),
            ],
        )
        mock_prices = {"AAPL": 100.0, "TSLA": 100.0, "MSFT": 100.0, "GOOGL": 100.0}

        with patch(
            "brain_api.core.orders.fetch_current_prices", return_value=mock_prices
        ):
            result = generate_orders(
                target_weights=target_weights,
                portfolio=portfolio,
                run_id="paper:2026-01-20",
                attempt=1,
                algorithm="sac",
            )

        sell_orders = [o for o in result.orders if o.side == "sell"]
        for order in sell_orders:
            assert order.qty == 49.0, (
                f"{order.symbol}: sell qty {order.qty} != position qty 49"
            )

    def test_skip_small_orders(self):
        """Orders at or above 1% weight delta but below $10 notional are skipped."""
        # NAV 500, weight delta 1.9% => $9.50 trade (below $10 min)
        target_weights = {"AAPL": 0.519, "CASH": 0.481}
        portfolio = PortfolioInput(
            cash=250.0,
            positions=[
                PositionInput(symbol="AAPL", qty=2.5, market_value=250.0),
            ],
        )
        mock_prices = {"AAPL": 100.0}

        with patch(
            "brain_api.core.orders.fetch_current_prices", return_value=mock_prices
        ):
            result = generate_orders(
                target_weights=target_weights,
                portfolio=portfolio,
                run_id="paper:2026-01-20",
                attempt=1,
                algorithm="sac",
            )

        assert result.summary.skipped_small_orders == 1
        assert len(result.orders) == 0
        assert result.summary.skipped_below_threshold == 0

    def test_skip_below_weight_threshold(self):
        """Legs with absolute weight delta under 1% are skipped (not full exit)."""
        target_weights = {"AAPL": 0.505, "CASH": 0.495}
        portfolio = PortfolioInput(
            cash=5000.0,
            positions=[
                PositionInput(symbol="AAPL", qty=50, market_value=5000.0),
            ],
        )
        mock_prices = {"AAPL": 100.0}

        with patch(
            "brain_api.core.orders.fetch_current_prices", return_value=mock_prices
        ):
            result = generate_orders(
                target_weights=target_weights,
                portfolio=portfolio,
                run_id="paper:2026-01-20",
                attempt=1,
                algorithm="sac",
            )

        assert result.summary.skipped_below_threshold == 1
        assert len(result.orders) == 0

    def test_full_exit_below_one_percent_weight_generates_sell(self):
        """Target weight 0 with an open position always generates a sell."""
        target_weights = {"AAPL": 0.0, "CASH": 1.0}
        portfolio = PortfolioInput(
            cash=9950.0,
            positions=[
                PositionInput(symbol="AAPL", qty=1.0, market_value=50.0),
            ],
        )
        mock_prices = {"AAPL": 100.0}

        with patch(
            "brain_api.core.orders.fetch_current_prices", return_value=mock_prices
        ):
            result = generate_orders(
                target_weights=target_weights,
                portfolio=portfolio,
                run_id="paper:2026-01-20",
                attempt=1,
                algorithm="sac",
            )

        assert len(result.orders) == 1
        assert result.orders[0].side == "sell"
        assert result.orders[0].symbol == "AAPL"
        assert result.summary.skipped_below_threshold == 0

    def test_buy_cap_scales_buys_proportionally(self):
        """When buy notional exceeds cash, scale all buy qty down proportionally."""
        target_weights = {"AAPL": 0.6, "MSFT": 0.5}
        portfolio = PortfolioInput(cash=1000.0, positions=[])
        mock_prices = {"AAPL": 100.0, "MSFT": 100.0}

        with patch(
            "brain_api.core.orders.fetch_current_prices", return_value=mock_prices
        ):
            result = generate_orders(
                target_weights=target_weights,
                portfolio=portfolio,
                run_id="paper:2026-01-20",
                attempt=1,
                algorithm="sac",
            )

        buys = [o for o in result.orders if o.side == "buy"]
        assert len(buys) == 2
        scale = 1000.0 / 1100.0
        by_sym = {o.symbol: o.qty for o in buys}
        assert by_sym["AAPL"] == round(6.0 * scale, 4)
        assert by_sym["MSFT"] == round(5.0 * scale, 4)
        total_notional = sum(o.qty * mock_prices[o.symbol] for o in buys)
        assert abs(total_notional - 1000.0) < 0.02
        assert result.summary.total_buy_value == pytest.approx(1000.0, abs=0.05)

    def test_buy_cap_not_applied_when_buys_fit_cash(self):
        """No scaling when total buy notional is within cash."""
        target_weights = {"AAPL": 0.45, "MSFT": 0.45}
        portfolio = PortfolioInput(cash=1000.0, positions=[])
        mock_prices = {"AAPL": 100.0, "MSFT": 100.0}

        with patch(
            "brain_api.core.orders.fetch_current_prices", return_value=mock_prices
        ):
            result = generate_orders(
                target_weights=target_weights,
                portfolio=portfolio,
                run_id="paper:2026-01-20",
                attempt=1,
                algorithm="sac",
            )

        buys = [o for o in result.orders if o.side == "buy"]
        assert len(buys) == 2
        assert {o.symbol: o.qty for o in buys} == {"AAPL": 4.5, "MSFT": 4.5}

    def test_buy_cap_uses_cash_plus_expected_sell_proceeds(self):
        """Buying power includes current cash and generated sell notionals."""
        target_weights = {"AAPL": 0.0, "MSFT": 0.8, "CASH": 0.2}
        portfolio = PortfolioInput(
            cash=200.0,
            positions=[
                PositionInput(symbol="AAPL", qty=8.0, market_value=800.0),
            ],
        )
        mock_prices = {"AAPL": 100.0, "MSFT": 50.0}

        with patch(
            "brain_api.core.orders.fetch_current_prices", return_value=mock_prices
        ):
            result = generate_orders(
                target_weights=target_weights,
                portfolio=portfolio,
                run_id="paper:2026-01-20",
                attempt=1,
                algorithm="sac",
            )

        buy = next(o for o in result.orders if o.side == "buy")
        assert buy.symbol == "MSFT"
        assert buy.qty == pytest.approx(16.0, rel=1e-5)
        assert result.summary.total_buy_value == pytest.approx(800.0, rel=1e-5)

    def test_to_dict_omits_limit_price_for_market(self):
        """Market order to_dict() does not include limit_price."""
        target_weights = {"AAPL": 0.5, "CASH": 0.5}
        portfolio = PortfolioInput(
            cash=10000.0,
            positions=[],
        )
        mock_prices = {"AAPL": 100.0}

        with patch(
            "brain_api.core.orders.fetch_current_prices", return_value=mock_prices
        ):
            result = generate_orders(
                target_weights=target_weights,
                portfolio=portfolio,
                run_id="paper:2026-01-20",
                attempt=1,
                algorithm="sac",
            )

        order_dict = result.orders[0].to_dict()
        assert "limit_price" not in order_dict
        assert order_dict["type"] == "market"
