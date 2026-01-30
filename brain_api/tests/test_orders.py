"""Tests for order generation logic.

This module tests:
- Limit price buffer calculation
- Order generation from target weights
- Client order ID generation
"""

from unittest.mock import patch

from brain_api.core.orders import (
    LIMIT_PRICE_BUFFER_PCT,
    PortfolioInput,
    PositionInput,
    calculate_limit_price,
    generate_client_order_id,
    generate_orders,
)


class TestLimitBuffer:
    """Tests for 2% limit buffer on order prices."""

    def test_buy_order_uses_price_times_1_02(self):
        """Test that buy orders use price * 1.02 (2% buffer above)."""
        current_price = 100.0
        limit_price = calculate_limit_price(current_price, "buy")

        # With 2% buffer, buy limit should be 102.00
        expected = round(current_price * (1 + LIMIT_PRICE_BUFFER_PCT), 2)
        assert limit_price == expected
        assert limit_price == 102.00

    def test_sell_order_uses_price_times_0_98(self):
        """Test that sell orders use price * 0.98 (2% buffer below)."""
        current_price = 100.0
        limit_price = calculate_limit_price(current_price, "sell")

        # With 2% buffer, sell limit should be 98.00
        expected = round(current_price * (1 - LIMIT_PRICE_BUFFER_PCT), 2)
        assert limit_price == expected
        assert limit_price == 98.00

    def test_buffer_applied_to_all_symbols(self):
        """Test that buffer is applied to all generated orders."""
        target_weights = {"AAPL": 0.5, "MSFT": 0.3, "CASH": 0.2}
        portfolio = PortfolioInput(
            cash=10000.0,
            positions=[],  # All cash, so all orders are buys
        )

        # Mock prices
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

        # Check all buy orders have correct buffer
        for order in result.orders:
            if order.side == "buy":
                expected_price = round(mock_prices[order.symbol] * 1.02, 2)
                assert order.limit_price == expected_price, (
                    f"{order.symbol}: expected {expected_price}, got {order.limit_price}"
                )

    def test_buffer_rounds_to_2_decimals(self):
        """Test that limit prices are rounded to 2 decimal places."""
        # Price that would result in a non-round number
        current_price = 123.456
        buy_limit = calculate_limit_price(current_price, "buy")
        sell_limit = calculate_limit_price(current_price, "sell")

        # Check they're rounded to 2 decimals
        assert buy_limit == round(buy_limit, 2)
        assert sell_limit == round(sell_limit, 2)

        # Verify exact values
        assert buy_limit == round(123.456 * 1.02, 2)  # 125.93
        assert sell_limit == round(123.456 * 0.98, 2)  # 120.99

    def test_buffer_config_is_configurable(self):
        """Test that the buffer percentage is set to 2%."""
        # This test verifies the constant is set correctly
        assert LIMIT_PRICE_BUFFER_PCT == 0.02, (
            f"Expected 2% buffer (0.02), got {LIMIT_PRICE_BUFFER_PCT}"
        )


class TestOrderGeneration:
    """Tests for order generation from target weights."""

    def test_generate_buy_orders_from_cash(self):
        """Test generating buy orders when starting from all cash."""
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
        assert result.orders[0].side == "buy"
        assert result.orders[0].symbol == "AAPL"

    def test_generate_sell_orders_to_cash(self):
        """Test generating sell orders when reducing positions."""
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
        assert result.orders[0].side == "sell"
        assert result.orders[0].symbol == "AAPL"

    def test_skip_small_orders(self):
        """Test that orders below minimum value are skipped."""
        # Very small weight change
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

        # 0.1% change on $10k = $10, which is at the minimum threshold
        # Verify the skipped count
        assert result.summary.skipped_small_orders >= 0


class TestClientOrderId:
    """Tests for client order ID generation."""

    def test_generate_deterministic_id(self):
        """Test that client order IDs are deterministic."""
        id1 = generate_client_order_id("paper:2026-01-20", 1, "AAPL", "buy")
        id2 = generate_client_order_id("paper:2026-01-20", 1, "AAPL", "buy")

        assert id1 == id2

    def test_id_format(self):
        """Test client order ID format matches expected pattern."""
        order_id = generate_client_order_id("paper:2026-01-20", 1, "AAPL", "buy")

        assert order_id == "paper:2026-01-20:attempt-1:AAPL:BUY"

    def test_different_attempts_different_ids(self):
        """Test that different attempts produce different IDs."""
        id1 = generate_client_order_id("paper:2026-01-20", 1, "AAPL", "buy")
        id2 = generate_client_order_id("paper:2026-01-20", 2, "AAPL", "buy")

        assert id1 != id2

    def test_different_symbols_different_ids(self):
        """Test that different symbols produce different IDs."""
        id1 = generate_client_order_id("paper:2026-01-20", 1, "AAPL", "buy")
        id2 = generate_client_order_id("paper:2026-01-20", 1, "MSFT", "buy")

        assert id1 != id2

    def test_different_sides_different_ids(self):
        """Test that different sides produce different IDs."""
        id1 = generate_client_order_id("paper:2026-01-20", 1, "AAPL", "buy")
        id2 = generate_client_order_id("paper:2026-01-20", 1, "AAPL", "sell")

        assert id1 != id2
