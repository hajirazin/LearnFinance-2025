"""Tests for order generation logic.

This module tests:
- Market order generation from target weights
- Sell qty capped at position quantity
- Client order ID generation
"""

from unittest.mock import patch

from fastapi.testclient import TestClient

from brain_api.core.orders import (
    convert_weights_to_whole_shares,
    generate_client_order_id,
)
from brain_api.main import app

client = TestClient(app)


class TestGenerateOrdersEndpoint:
    """Tests for POST /orders/generate endpoint."""

    def test_rejects_non_usd_portfolio_currency(self):
        """A portfolio with currency != USD returns 400."""
        payload = {
            "target_weights": {"AAPL": 0.5, "MSFT": 0.5},
            "portfolio": {"cash": 1000.0, "currency": "SGD", "positions": []},
            "run_id": "test",
            "attempt": 1,
            "algorithm": "test",
        }
        response = client.post("/orders/generate", json=payload)
        assert response.status_code == 400
        assert "Portfolio currency must be USD" in response.json()["detail"]

    def test_sell_phase_fetches_only_material_sell_prices(self):
        payload = {
            "target_weights": {"AAPL": 0.0, "MSFT": 0.5, "CASH": 0.5},
            "portfolio": {
                "cash": 0.0,
                "currency": "USD",
                "positions": [{"symbol": "AAPL", "qty": 10, "market_value": 1000}],
            },
            "run_id": "paper:2026-01-20",
            "attempt": 1,
            "algorithm": "sac",
            "order_side": "sell",
        }

        with patch(
            "brain_api.core.orders.fetch_current_prices", return_value={"AAPL": 100.0}
        ) as fetch:
            response = client.post("/orders/generate", json=payload)

        assert response.status_code == 200
        assert [order["side"] for order in response.json()["orders"]] == ["sell"]
        assert response.json()["orders"][0]["qty"] == 10.0
        # Full-exit sells flatten at broker qty and do not fetch Yahoo prices.
        fetch.assert_called_once_with([])

    def test_buy_phase_fetches_price_when_phase_runs(self):
        payload = {
            "target_weights": {"AAPL": 0.5, "CASH": 0.5},
            "portfolio": {"cash": 1000.0, "currency": "USD", "positions": []},
            "run_id": "paper:2026-01-20",
            "attempt": 1,
            "algorithm": "sac",
            "order_side": "buy",
        }

        with (
            patch(
                "brain_api.core.orders.fetch_current_prices",
                return_value={"AAPL": 100.0},
            ) as fetch,
            patch("brain_api.core.orders.fetch_ohlc_window", return_value={}),
        ):
            response = client.post("/orders/generate", json=payload)

        assert response.status_code == 200
        assert response.json()["prices_used"] == {"AAPL": 100.0}
        fetch.assert_called_once_with(["AAPL"])

    def test_sac_rejects_missing_price_for_material_leg(self):
        payload = {
            "target_weights": {"AAPL": 0.5, "CASH": 0.5},
            "portfolio": {"cash": 1000.0, "currency": "USD", "positions": []},
            "run_id": "paper:2026-01-20",
            "attempt": 1,
            "algorithm": "sac",
            "order_side": "buy",
        }

        with (
            patch("brain_api.core.orders.fetch_current_prices", return_value={}),
            patch("brain_api.core.orders.fetch_ohlc_window") as fetch_ohlc,
        ):
            response = client.post("/orders/generate", json=payload)

        assert response.status_code == 422
        assert "missing_or_invalid=['AAPL']" in response.json()["detail"]
        fetch_ohlc.assert_not_called()

    def test_full_exit_below_min_trade_value_returns_sell(self):
        """Router: a sub-$10 target-0 position still generates a flatten sell."""
        payload = {
            "target_weights": {"AAPL": 0.0, "CASH": 1.0},
            "portfolio": {
                "cash": 995.0,
                "currency": "USD",
                "positions": [{"symbol": "AAPL", "qty": 0.05, "market_value": 5.0}],
            },
            "run_id": "paper:2026-01-20",
            "attempt": 1,
            "algorithm": "sac",
        }

        with patch(
            "brain_api.core.orders.fetch_current_prices", return_value={}
        ) as fetch:
            response = client.post("/orders/generate", json=payload)

        assert response.status_code == 200
        orders = response.json()["orders"]
        assert len(orders) == 1
        assert orders[0]["side"] == "sell"
        assert orders[0]["symbol"] == "AAPL"
        assert orders[0]["qty"] == 0.05
        fetch.assert_called_once_with([])


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


class TestConvertWeightsToWholeShares:
    """Tests for paper-only whole-share allocation."""

    def test_normal_affordable_stock(self):
        """A stock with price below the per-stock NAV gets >0 shares."""
        weights = {"AAPL": 50.0}
        with patch(
            "brain_api.core.orders.fetch_current_prices",
            return_value={"AAPL": 100.0},
        ):
            result = convert_weights_to_whole_shares(
                weights, total_nav=100_000.0, atr_map={}
            )

        assert len(result.details) == 1
        d = result.details[0]
        assert d.symbol == "AAPL"
        assert d.price == 100.0
        assert d.whole_shares == 500  # (50% of 100k = 50k) / 100 = 500
        assert d.trade_value == 50000.0

    def test_high_price_stock_yields_zero_shares(self):
        """A stock whose per-share price exceeds the allocated NAV produces 0 shares rather than being hidden."""
        # 1% of 100k NAV = 1000; price 10_000 → whole_shares = 0
        weights = {"RELNR.NS": 1.0}
        with patch(
            "brain_api.core.orders.fetch_current_prices",
            return_value={"RELNR.NS": 10_000.0},
        ):
            result = convert_weights_to_whole_shares(
                weights, total_nav=100_000.0, atr_map={}
            )

        assert len(result.details) == 1
        d = result.details[0]
        assert d.symbol == "RELNR.NS"
        assert d.price == 10_000.0
        assert d.whole_shares == 0
        assert d.trade_value == 0.0

    def test_mixed_prices_some_zero_some_positive(self):
        """Mix of affordable and high-price stocks; both appear in the result."""
        weights = {"RELIANCE.NS": 10.0, "CHEAP.NS": 40.0, "HIGHPR.NS": 2.0}
        with patch(
            "brain_api.core.orders.fetch_current_prices",
            return_value={
                "RELIANCE.NS": 1_200.0,
                "CHEAP.NS": 50.0,
                "HIGHPR.NS": 200_000.0,  # 2% of 1M = 20k → 0 shares
            },
        ):
            result = convert_weights_to_whole_shares(
                weights, total_nav=1_000_000.0, atr_map={}
            )

        assert len(result.details) == 3

        by_symbol = {d.symbol: d for d in result.details}
        assert by_symbol["CHEAP.NS"].whole_shares == 8000
        assert by_symbol["RELIANCE.NS"].whole_shares == 83
        assert by_symbol["HIGHPR.NS"].whole_shares == 0
        assert by_symbol["HIGHPR.NS"].trade_value == 0.0

    def test_skips_zero_weight_and_below_min_trade(self):
        """Stocks with 0% weight or trade_value below MIN_TRADE_VALUE are excluded, not shown as 0."""
        weights = {"SKIP": 0.0, "AAPL": 50.0, "TINY": 0.0005}
        with patch(
            "brain_api.core.orders.fetch_current_prices",
            return_value={"SKIP": 100.0, "AAPL": 100.0, "TINY": 1.0},
        ):
            result = convert_weights_to_whole_shares(
                weights, total_nav=100_000.0, atr_map={}
            )

        symbols = [d.symbol for d in result.details]
        assert "SKIP" not in symbols
        assert "TINY" not in symbols
        assert "AAPL" in symbols
