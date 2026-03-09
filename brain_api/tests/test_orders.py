"""Tests for order generation logic.

This module tests:
- Market order generation from target weights
- Sell qty capped at position quantity
- Client order ID generation
"""

from unittest.mock import patch

from brain_api.core.orders import (
    PortfolioInput,
    PositionInput,
    generate_client_order_id,
    generate_orders,
)


class TestOrderGeneration:
    """Tests for market order generation from target weights."""

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
                algorithm="ppo",
            )

        assert len(result.orders) == 1
        order = result.orders[0]
        assert order.side == "buy"
        assert order.symbol == "AAPL"
        expected_qty = round(5000.0 / 100.0, 4)
        assert order.qty == expected_qty

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
                algorithm="ppo",
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
                algorithm="ppo",
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
                algorithm="ppo",
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
                algorithm="ppo",
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
                algorithm="ppo",
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
                algorithm="ppo",
            )

        sell_orders = [o for o in result.orders if o.side == "sell"]
        for order in sell_orders:
            assert order.qty == 49.0, (
                f"{order.symbol}: sell qty {order.qty} != position qty 49"
            )

    def test_skip_small_orders(self):
        """Orders below minimum value are skipped."""
        target_weights = {"AAPL": 0.501, "CASH": 0.499}
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
                algorithm="ppo",
            )

        assert result.summary.skipped_small_orders >= 0

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
                algorithm="ppo",
            )

        order_dict = result.orders[0].to_dict()
        assert "limit_price" not in order_dict
        assert order_dict["type"] == "market"


class TestClientOrderId:
    """Tests for client order ID generation."""

    def test_generate_deterministic_id(self):
        """Client order IDs are deterministic."""
        id1 = generate_client_order_id("paper:2026-01-20", 1, "AAPL", "buy")
        id2 = generate_client_order_id("paper:2026-01-20", 1, "AAPL", "buy")

        assert id1 == id2

    def test_id_format(self):
        """Client order ID format matches expected pattern."""
        order_id = generate_client_order_id("paper:2026-01-20", 1, "AAPL", "buy")

        assert order_id == "paper:2026-01-20:attempt-1:AAPL:BUY"

    def test_different_attempts_different_ids(self):
        """Different attempts produce different IDs."""
        id1 = generate_client_order_id("paper:2026-01-20", 1, "AAPL", "buy")
        id2 = generate_client_order_id("paper:2026-01-20", 2, "AAPL", "buy")

        assert id1 != id2

    def test_different_symbols_different_ids(self):
        """Different symbols produce different IDs."""
        id1 = generate_client_order_id("paper:2026-01-20", 1, "AAPL", "buy")
        id2 = generate_client_order_id("paper:2026-01-20", 1, "MSFT", "buy")

        assert id1 != id2

    def test_different_sides_different_ids(self):
        """Different sides produce different IDs."""
        id1 = generate_client_order_id("paper:2026-01-20", 1, "AAPL", "buy")
        id2 = generate_client_order_id("paper:2026-01-20", 1, "AAPL", "sell")

        assert id1 != id2
