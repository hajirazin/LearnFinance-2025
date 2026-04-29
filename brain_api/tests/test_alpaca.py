"""Tests for Alpaca paper trading endpoints.

This module tests:
- GET /alpaca/portfolio - fetch account, positions, open orders
- POST /alpaca/submit-orders - submit orders to Alpaca
- GET /alpaca/order-history - fetch order history
"""

from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from brain_api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_alpaca_credentials():
    """Mock Alpaca credentials for all accounts (sac, hrp, dhrp)."""
    with patch.dict(
        "os.environ",
        {
            "ALPACA_SAC_KEY": "test-sac-key",
            "ALPACA_SAC_SECRET": "test-sac-secret",
            "ALPACA_HRP_KEY": "test-hrp-key",
            "ALPACA_HRP_SECRET": "test-hrp-secret",
            "ALPACA_DHRP_KEY": "test-dhrp-key",
            "ALPACA_DHRP_SECRET": "test-dhrp-secret",
        },
    ):
        yield


# =============================================================================
# GET /alpaca/portfolio Tests
# =============================================================================


class TestGetPortfolio:
    """Tests for GET /alpaca/portfolio endpoint."""

    def test_get_portfolio_sac_success(self, client, mock_alpaca_credentials):
        """Test GET /alpaca/portfolio?account=sac returns normalized data."""
        mock_account = {"cash": "5000.00"}
        mock_positions = [{"symbol": "GOOGL", "qty": "3", "market_value": "4500.00"}]
        mock_orders = [{"id": "order-1"}]  # 1 open order

        with patch("brain_api.routes.alpaca.httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value.__enter__.return_value = mock_client
            mock_client.get.side_effect = [
                MagicMock(json=lambda: mock_account, raise_for_status=lambda: None),
                MagicMock(json=lambda: mock_positions, raise_for_status=lambda: None),
                MagicMock(json=lambda: mock_orders, raise_for_status=lambda: None),
            ]

            response = client.get("/alpaca/portfolio", params={"account": "sac"})

        assert response.status_code == 200
        data = response.json()
        assert data["cash"] == 5000.00
        assert len(data["positions"]) == 1
        assert data["open_orders_count"] == 1

    def test_get_portfolio_hrp_success(self, client, mock_alpaca_credentials):
        """Test GET /alpaca/portfolio?account=hrp returns normalized data."""
        mock_account = {"cash": "15000.00"}
        mock_positions = []  # Empty portfolio
        mock_orders = []

        with patch("brain_api.routes.alpaca.httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value.__enter__.return_value = mock_client
            mock_client.get.side_effect = [
                MagicMock(json=lambda: mock_account, raise_for_status=lambda: None),
                MagicMock(json=lambda: mock_positions, raise_for_status=lambda: None),
                MagicMock(json=lambda: mock_orders, raise_for_status=lambda: None),
            ]

            response = client.get("/alpaca/portfolio", params={"account": "hrp"})

        assert response.status_code == 200
        data = response.json()
        assert data["cash"] == 15000.00
        assert len(data["positions"]) == 0
        assert data["open_orders_count"] == 0

    def test_get_portfolio_invalid_account_returns_422(self, client):
        """Test GET /alpaca/portfolio?account=invalid returns 422."""
        response = client.get("/alpaca/portfolio", params={"account": "invalid"})
        assert response.status_code == 422

    def test_get_portfolio_missing_account_returns_422(self, client):
        """Test GET /alpaca/portfolio without account param returns 422."""
        response = client.get("/alpaca/portfolio")
        assert response.status_code == 422

    def test_get_portfolio_alpaca_timeout_returns_503(
        self, client, mock_alpaca_credentials
    ):
        """Test that Alpaca API timeout returns 503."""
        with patch("brain_api.routes.alpaca.httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value.__enter__.return_value = mock_client
            mock_client.get.side_effect = httpx.TimeoutException("Connection timeout")

            response = client.get("/alpaca/portfolio", params={"account": "sac"})

        assert response.status_code == 503
        assert "timeout" in response.json()["detail"].lower()

    def test_get_portfolio_alpaca_http_error_returns_503(
        self, client, mock_alpaca_credentials
    ):
        """Test that Alpaca API HTTP error returns 503."""
        with patch("brain_api.routes.alpaca.httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value.__enter__.return_value = mock_client

            mock_response = MagicMock()
            mock_response.status_code = 401
            mock_client.get.side_effect = httpx.HTTPStatusError(
                "Unauthorized", request=MagicMock(), response=mock_response
            )

            response = client.get("/alpaca/portfolio", params={"account": "sac"})

        assert response.status_code == 503


# =============================================================================
# POST /alpaca/submit-orders Tests
# =============================================================================


class TestSubmitOrders:
    """Tests for POST /alpaca/submit-orders endpoint."""

    def test_submit_orders_success(self, client, mock_alpaca_credentials):
        """Test POST /alpaca/submit-orders with valid orders returns success counts."""
        with patch("brain_api.routes.alpaca.httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value.__enter__.return_value = mock_client

            # Mock successful order response
            mock_client.post.return_value = MagicMock(
                json=lambda: {"id": "order-123", "status": "accepted"},
                raise_for_status=lambda: None,
            )

            response = client.post(
                "/alpaca/submit-orders",
                json={
                    "account": "sac",
                    "orders": [
                        {
                            "symbol": "AAPL",
                            "qty": 5,
                            "side": "buy",
                            "type": "limit",
                            "time_in_force": "day",
                            "limit_price": 175.50,
                            "client_order_id": "paper:2026-02-05:attempt-1:AAPL:BUY",
                        }
                    ],
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["account"] == "sac"
        assert data["orders_submitted"] == 1
        assert data["orders_failed"] == 0
        assert data["skipped"] is False
        assert len(data["results"]) == 1
        assert data["results"][0]["status"] == "accepted"

    def test_submit_orders_partial_failure(self, client, mock_alpaca_credentials):
        """Test POST /alpaca/submit-orders handles partial failures gracefully."""
        with patch("brain_api.routes.alpaca.httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value.__enter__.return_value = mock_client

            # First order succeeds, second fails
            mock_success = MagicMock(
                json=lambda: {"id": "order-1", "status": "accepted"},
                raise_for_status=lambda: None,
            )
            mock_fail_response = MagicMock()
            mock_fail_response.status_code = 422
            mock_fail_response.json.return_value = {"message": "Insufficient funds"}
            mock_fail = httpx.HTTPStatusError(
                "Error", request=MagicMock(), response=mock_fail_response
            )

            mock_client.post.side_effect = [mock_success, mock_fail]

            response = client.post(
                "/alpaca/submit-orders",
                json={
                    "account": "sac",
                    "orders": [
                        {
                            "symbol": "AAPL",
                            "qty": 5,
                            "side": "buy",
                            "type": "limit",
                            "time_in_force": "day",
                            "limit_price": 175.50,
                            "client_order_id": "order-1",
                        },
                        {
                            "symbol": "MSFT",
                            "qty": 100000,  # Too many shares
                            "side": "buy",
                            "type": "limit",
                            "time_in_force": "day",
                            "limit_price": 400.00,
                            "client_order_id": "order-2",
                        },
                    ],
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["orders_submitted"] == 1
        assert data["orders_failed"] == 1
        assert len(data["results"]) == 2
        assert data["results"][0]["status"] == "accepted"
        assert data["results"][1]["status"] == "rejected"
        assert "Insufficient funds" in data["results"][1]["error"]

    def test_submit_orders_empty_array(self, client, mock_alpaca_credentials):
        """Test POST /alpaca/submit-orders with empty orders returns 0 submitted."""
        response = client.post(
            "/alpaca/submit-orders",
            json={"account": "hrp", "orders": []},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["account"] == "hrp"
        assert data["orders_submitted"] == 0
        assert data["orders_failed"] == 0
        assert data["skipped"] is False
        assert len(data["results"]) == 0

    def test_submit_orders_invalid_account_returns_422(self, client):
        """Test POST /alpaca/submit-orders with invalid account returns 422."""
        response = client.post(
            "/alpaca/submit-orders",
            json={"account": "invalid", "orders": []},
        )
        assert response.status_code == 422

    def test_submit_orders_missing_required_fields_returns_422(self, client):
        """Test POST /alpaca/submit-orders with missing fields returns 422."""
        response = client.post(
            "/alpaca/submit-orders",
            json={
                "account": "sac",
                "orders": [
                    {
                        "symbol": "AAPL",
                        # Missing qty, side, limit_price, client_order_id
                    }
                ],
            },
        )
        assert response.status_code == 422


# =============================================================================
# GET /alpaca/order-history Tests
# =============================================================================


class TestOrderHistory:
    """Tests for GET /alpaca/order-history endpoint."""

    def test_order_history_success(self, client, mock_alpaca_credentials):
        """Test GET /alpaca/order-history returns list with order details."""
        mock_orders = [
            {
                "id": "order-1",
                "client_order_id": "paper:2026-02-05:attempt-1:AAPL:BUY",
                "symbol": "AAPL",
                "side": "buy",
                "status": "filled",
                "filled_qty": "10",
                "filled_avg_price": "175.25",
            },
            {
                "id": "order-2",
                "client_order_id": "paper:2026-02-05:attempt-1:MSFT:SELL",
                "symbol": "MSFT",
                "side": "sell",
                "status": "canceled",
                "filled_qty": "0",
                "filled_avg_price": None,
            },
        ]

        with patch("brain_api.routes.alpaca.httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value.__enter__.return_value = mock_client
            mock_client.get.return_value = MagicMock(
                json=lambda: mock_orders, raise_for_status=lambda: None
            )

            response = client.get(
                "/alpaca/order-history",
                params={"account": "sac", "after": "2026-02-01"},
            )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["id"] == "order-1"
        assert data[0]["client_order_id"] == "paper:2026-02-05:attempt-1:AAPL:BUY"
        assert data[0]["status"] == "filled"
        assert data[0]["filled_qty"] == "10"
        assert data[0]["filled_avg_price"] == "175.25"
        assert data[1]["status"] == "canceled"

    def test_order_history_empty(self, client, mock_alpaca_credentials):
        """Test GET /alpaca/order-history returns empty list when no orders."""
        with patch("brain_api.routes.alpaca.httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value.__enter__.return_value = mock_client
            mock_client.get.return_value = MagicMock(
                json=list, raise_for_status=lambda: None
            )

            response = client.get(
                "/alpaca/order-history",
                params={"account": "sac", "after": "2026-02-01"},
            )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 0

    def test_order_history_invalid_account_returns_422(self, client):
        """Test GET /alpaca/order-history?account=invalid returns 422."""
        response = client.get(
            "/alpaca/order-history",
            params={"account": "invalid", "after": "2026-02-01"},
        )
        assert response.status_code == 422

    def test_order_history_missing_after_returns_422(self, client):
        """Test GET /alpaca/order-history without after param returns 422."""
        response = client.get(
            "/alpaca/order-history",
            params={"account": "sac"},
        )
        assert response.status_code == 422

    def test_order_history_alpaca_timeout_returns_503(
        self, client, mock_alpaca_credentials
    ):
        """Test that Alpaca API timeout returns 503."""
        with patch("brain_api.routes.alpaca.httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value.__enter__.return_value = mock_client
            mock_client.get.side_effect = httpx.TimeoutException("Connection timeout")

            response = client.get(
                "/alpaca/order-history",
                params={"account": "hrp", "after": "2026-02-01"},
            )

        assert response.status_code == 503
        assert "timeout" in response.json()["detail"].lower()

    def test_order_history_alpaca_http_error_returns_503(
        self, client, mock_alpaca_credentials
    ):
        """Test that Alpaca API HTTP error returns 503."""
        with patch("brain_api.routes.alpaca.httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value.__enter__.return_value = mock_client

            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_client.get.side_effect = httpx.HTTPStatusError(
                "Server error", request=MagicMock(), response=mock_response
            )

            response = client.get(
                "/alpaca/order-history",
                params={"account": "sac", "after": "2026-02-01"},
            )

        assert response.status_code == 503


# =============================================================================
# Tests for the new dhrp account
# =============================================================================


class TestDHRPAccount:
    """Cover the third Alpaca account (Double HRP, halal_new universe)."""

    def test_get_portfolio_dhrp_uses_dhrp_credentials(
        self, client, mock_alpaca_credentials
    ):
        """GET /alpaca/portfolio?account=dhrp should send ALPACA_DHRP_* headers."""
        mock_account = {"cash": "8000.00"}
        mock_positions = [{"symbol": "AAPL", "qty": "2", "market_value": "350.00"}]
        mock_orders = []

        with patch("brain_api.routes.alpaca.httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value.__enter__.return_value = mock_client
            mock_client.get.side_effect = [
                MagicMock(json=lambda: mock_account, raise_for_status=lambda: None),
                MagicMock(json=lambda: mock_positions, raise_for_status=lambda: None),
                MagicMock(json=lambda: mock_orders, raise_for_status=lambda: None),
            ]

            response = client.get("/alpaca/portfolio", params={"account": "dhrp"})

        assert response.status_code == 200
        data = response.json()
        assert data["cash"] == 8000.00
        assert data["open_orders_count"] == 0

        # Confirm the dhrp credentials made it into the httpx.Client kwargs.
        kwargs = mock_client_class.call_args.kwargs
        assert kwargs["headers"]["APCA-API-KEY-ID"] == "test-dhrp-key"
        assert kwargs["headers"]["APCA-API-SECRET-KEY"] == "test-dhrp-secret"

    def test_get_portfolio_dhrp_missing_credentials_returns_500(self, client):
        """Without ALPACA_DHRP_KEY/SECRET, the endpoint must 500 explicitly."""
        with patch.dict(
            "os.environ",
            {"ALPACA_DHRP_KEY": "", "ALPACA_DHRP_SECRET": ""},
        ):
            response = client.get("/alpaca/portfolio", params={"account": "dhrp"})
        assert response.status_code == 500
        assert "ALPACA_DHRP_KEY" in response.json()["detail"]

    def test_submit_orders_dhrp_uses_dhrp_credentials(
        self, client, mock_alpaca_credentials
    ):
        """POST /alpaca/submit-orders with account=dhrp uses dhrp creds."""
        with patch("brain_api.routes.alpaca.httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value.__enter__.return_value = mock_client
            mock_client.post.return_value = MagicMock(
                json=lambda: {"id": "order-dhrp-1", "status": "accepted"},
                raise_for_status=lambda: None,
            )

            response = client.post(
                "/alpaca/submit-orders",
                json={
                    "account": "dhrp",
                    "orders": [
                        {
                            "symbol": "NVDA",
                            "qty": 1,
                            "side": "buy",
                            "type": "limit",
                            "time_in_force": "day",
                            "limit_price": 700.0,
                            "client_order_id": "paper:2026-02-23:attempt-1:NVDA:BUY",
                        }
                    ],
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["account"] == "dhrp"
        assert data["orders_submitted"] == 1

        kwargs = mock_client_class.call_args.kwargs
        assert kwargs["headers"]["APCA-API-KEY-ID"] == "test-dhrp-key"

    def test_order_history_dhrp_success(self, client, mock_alpaca_credentials):
        """GET /alpaca/order-history?account=dhrp returns dhrp order list."""
        mock_orders = [
            {
                "id": "ord-dhrp-1",
                "client_order_id": "paper:2026-02-23:attempt-1:AAPL:BUY",
                "symbol": "AAPL",
                "side": "buy",
                "status": "filled",
                "filled_qty": "1",
                "filled_avg_price": "175.0",
            }
        ]
        with patch("brain_api.routes.alpaca.httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value.__enter__.return_value = mock_client
            mock_client.get.return_value = MagicMock(
                json=lambda: mock_orders, raise_for_status=lambda: None
            )

            response = client.get(
                "/alpaca/order-history",
                params={"account": "dhrp", "after": "2026-02-23"},
            )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["id"] == "ord-dhrp-1"

        kwargs = mock_client_class.call_args.kwargs
        assert kwargs["headers"]["APCA-API-KEY-ID"] == "test-dhrp-key"
