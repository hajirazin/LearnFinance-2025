"""Tests for the REST-based IBKR Client Portal API client."""

from unittest.mock import MagicMock, patch

import pytest

from brain_api.core.ibkr_client import (
    IBKROrderSpec,
    get_connection_config,
    get_portfolio,
    get_session_status,
    list_open_order_refs,
    submit_order,
)


@pytest.fixture
def ibkr_env(monkeypatch):
    """Set the minimal IBKR_SAC_HALAL_* env block for routing."""
    monkeypatch.setenv("IBKR_SAC_HALAL_HOST", "localhost")
    monkeypatch.setenv("IBKR_SAC_HALAL_PORT", "5000")
    monkeypatch.setenv("IBKR_SAC_HALAL_CLIENT_ID", "11")
    monkeypatch.setenv("IBKR_SAC_HALAL_ACCOUNT_CODE", "DU1234567")


@pytest.fixture
def config(ibkr_env):
    return get_connection_config("sac_halal")


class TestIBKRClientStatus:
    @patch("httpx.Client.post")
    def test_get_session_status_authenticated(self, mock_post, config):
        # Mock auth status response
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "authenticated": True,
            "connected": True,
            "competing": False,
        }

        status = get_session_status(config)
        assert status is True

        # Verify tickle was also called
        assert mock_post.call_count == 2
        calls = mock_post.call_args_list
        assert "/v1/api/iserver/auth/status" in calls[0][0][0]
        assert "/v1/api/tickle" in calls[1][0][0]

    @patch("httpx.Client.post")
    def test_get_session_status_not_authenticated(self, mock_post, config):
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "authenticated": False,
            "connected": True,
            "competing": False,
        }

        status = get_session_status(config)
        assert status is False


class TestIBKRClientPortfolio:
    @patch("httpx.Client.get")
    def test_get_portfolio_happy_path(self, mock_get, config):
        def fake_get(*args, **kwargs):
            url = args[0] if isinstance(args[0], str) else args[1]
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            if "/iserver/accounts" in url:
                mock_resp.json.return_value = {"accounts": [config.account_code]}
            elif "/portfolio/accounts" in url:
                mock_resp.json.return_value = [
                    {"id": config.account_code, "accountId": config.account_code}
                ]
            elif "/summary" in url:
                mock_resp.json.return_value = {
                    "NetLiquidation": {"amount": 10000.0, "currency": "USD"}
                }
            elif "/positions/0" in url:
                mock_resp.json.return_value = [
                    {
                        "conid": 123,
                        "contractDesc": "AAPL",
                        "position": 10.0,
                        "mktValue": 1500.0,
                    }
                ]
            elif "/ledger" in url:
                mock_resp.json.return_value = {
                    "USD": {"cashbalance": 8500.0},
                    "BASE": {"cashbalance": 8500.0},
                }
            elif "/orders" in url:
                mock_resp.json.return_value = {"orders": []}
            return mock_resp

        mock_get.side_effect = fake_get

        portfolio = get_portfolio(config)

        assert portfolio.cash == 8500.0
        assert portfolio.cash_balances["USD"] == 8500.0
        assert portfolio.open_orders_count == 0
        assert len(portfolio.positions) == 1
        assert portfolio.positions[0].symbol == "AAPL"
        assert portfolio.positions[0].qty == 10.0


class TestIBKRClientOrders:
    @patch("httpx.Client.post")
    @patch("httpx.Client.get")
    def test_submit_order_direct_success(self, mock_get, mock_post, config):
        def fake_post(*args, **kwargs):
            url = args[0] if isinstance(args[0], str) else args[1]
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            if "/secdef/search" in url:
                mock_resp.json.return_value = [{"conid": 265598, "symbol": "AAPL"}]
            elif "/orders" in url:
                # Direct success, no precaution
                mock_resp.json.return_value = [
                    {"order_id": "999", "order_status": "Submitted"}
                ]
            return mock_resp

        mock_post.side_effect = fake_post

        spec = IBKROrderSpec(
            symbol="AAPL",
            qty=10.0,
            side="buy",
            order_type="limit",
            time_in_force="day",
            limit_price=150.0,
            client_order_id="test_id",
            currency="USD",
        )

        result = submit_order(config, spec)
        assert result.client_order_id == "test_id"
        assert result.perm_id == 999
        assert result.status == "Submitted"

    @patch("httpx.Client.post")
    def test_submit_order_with_reply_loop(self, mock_post, config):
        call_count = [0]

        def fake_post(*args, **kwargs):
            url = args[0] if isinstance(args[0], str) else args[1]
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            if "/secdef/search" in url:
                mock_resp.json.return_value = [{"conid": 265598, "symbol": "AAPL"}]
            elif "/orders" in url:
                # First submission returns a precaution
                mock_resp.json.return_value = [
                    {"id": "reply_123", "message": ["Are you sure?"]}
                ]
            elif "/reply/" in url:
                call_count[0] += 1
                if call_count[0] == 1:
                    # First reply returns ANOTHER precaution
                    mock_resp.json.return_value = [
                        {"id": "reply_456", "message": ["Really sure?"]}
                    ]
                else:
                    # Second reply returns success
                    mock_resp.json.return_value = [
                        {"order_id": "888", "order_status": "Submitted"}
                    ]
            return mock_resp

        mock_post.side_effect = fake_post

        spec = IBKROrderSpec(
            symbol="AAPL",
            qty=10.0,
            side="buy",
            order_type="limit",
            time_in_force="day",
            limit_price=150.0,
            client_order_id="test_id",
            currency="USD",
        )

        result = submit_order(config, spec)
        assert result.client_order_id == "test_id"
        assert result.perm_id == 888
        assert result.status == "Submitted"
        assert call_count[0] == 2

    @patch("httpx.Client.get")
    def test_list_open_order_refs(self, mock_get, config):
        def fake_get(*args, **kwargs):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "orders": [
                    {"status": "Submitted", "cOID": "ref1"},
                    {"status": "PreSubmitted", "cOID": "ref2"},
                    {
                        "status": "Filled",
                        "cOID": "ref3",
                    },  # Filled should not be included
                ]
            }
            return mock_resp

        mock_get.side_effect = fake_get

        refs = list_open_order_refs(config)
        assert refs == {"ref1", "ref2"}
