"""Tests for ``GET /alpaca/clock``.

The clock route proxies Alpaca's ``/v2/clock`` so the Temporal
``sell_wait_buy`` helper can sleep until the next NYSE open before it
starts the 1-min status-poll cadence.

Authentication contract (asserted explicitly here):
- Uses the generic ``ALPACA_API_KEY`` / ``ALPACA_API_SECRET`` env pair
  -- the same pair the news backfill and universe scraper already
  consume -- NOT the per-account ``ALPACA_{SAC,HRP,DHRP}_*`` trading
  credentials.
- Always hits the paper host (``https://paper-api.alpaca.markets``).
  The clock payload is identical paper vs live, so coupling it to a
  per-account live override would add complexity without changing
  semantics.

Per the repo testing rules (AGENTS.md), these are API-level tests
that exercise the route via the FastAPI ``TestClient``; we do not
add schema-only tests.
"""

from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from brain_api.main import app


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def mock_generic_alpaca_credentials():
    """Provision the generic non-account-scoped Alpaca creds.

    Distinct from the per-account ``mock_alpaca_credentials`` fixture
    in ``test_alpaca.py`` -- the clock route deliberately does NOT
    touch the per-account creds.
    """
    with patch.dict(
        "os.environ",
        {
            "ALPACA_API_KEY": "test-generic-key",
            "ALPACA_API_SECRET": "test-generic-secret",
        },
    ):
        yield


_CLOSED_PAYLOAD = {
    "timestamp": "2026-05-11T12:00:00.000000-04:00",
    "is_open": False,
    "next_open": "2026-05-11T09:30:00-04:00",
    "next_close": "2026-05-11T16:00:00-04:00",
}

_OPEN_PAYLOAD = {
    "timestamp": "2026-05-11T14:00:00.000000-04:00",
    "is_open": True,
    "next_open": "2026-05-12T09:30:00-04:00",
    "next_close": "2026-05-11T16:00:00-04:00",
}


class TestGetClock:
    """API-level tests for ``GET /alpaca/clock``."""

    def test_get_clock_market_closed_success(
        self, client, mock_generic_alpaca_credentials
    ):
        """A successful upstream call passes the clock payload through."""
        with patch("brain_api.routes.alpaca.httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value.__enter__.return_value = mock_client
            mock_client.get.return_value = MagicMock(
                json=lambda: _CLOSED_PAYLOAD,
                raise_for_status=lambda: None,
            )

            response = client.get("/alpaca/clock")

        assert response.status_code == 200
        data = response.json()
        assert data["is_open"] is False
        assert data["next_open"] == _CLOSED_PAYLOAD["next_open"]
        assert data["next_close"] == _CLOSED_PAYLOAD["next_close"]
        assert data["timestamp"] == _CLOSED_PAYLOAD["timestamp"]

    def test_get_clock_market_open_passthrough(
        self, client, mock_generic_alpaca_credentials
    ):
        """``is_open=True`` and timestamps are forwarded unchanged."""
        with patch("brain_api.routes.alpaca.httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value.__enter__.return_value = mock_client
            mock_client.get.return_value = MagicMock(
                json=lambda: _OPEN_PAYLOAD,
                raise_for_status=lambda: None,
            )

            response = client.get("/alpaca/clock")

        assert response.status_code == 200
        data = response.json()
        assert data["is_open"] is True
        assert data["next_open"] == _OPEN_PAYLOAD["next_open"]

    def test_get_clock_uses_paper_host_and_generic_creds(
        self, client, mock_generic_alpaca_credentials
    ):
        """Route hits the paper host with the generic creds, not per-account."""
        with patch("brain_api.routes.alpaca.httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value.__enter__.return_value = mock_client
            mock_client.get.return_value = MagicMock(
                json=lambda: _OPEN_PAYLOAD,
                raise_for_status=lambda: None,
            )

            response = client.get("/alpaca/clock")

        assert response.status_code == 200
        # Always paper host -- clock payload is identical paper vs live.
        kwargs = mock_client_class.call_args.kwargs
        assert kwargs["base_url"] == "https://paper-api.alpaca.markets"
        # Generic creds, not per-account.
        headers = kwargs["headers"]
        assert headers["APCA-API-KEY-ID"] == "test-generic-key"
        assert headers["APCA-API-SECRET-KEY"] == "test-generic-secret"

    def test_get_clock_missing_generic_credentials_returns_500(self, client):
        """Without the generic creds the route fails loud (no fallback)."""
        # Ensure the env vars are absent for this test specifically.
        with patch.dict("os.environ", {}, clear=False):
            # Explicitly remove if leaked from another test.
            import os

            os.environ.pop("ALPACA_API_KEY", None)
            os.environ.pop("ALPACA_API_SECRET", None)

            response = client.get("/alpaca/clock")

        assert response.status_code == 500
        assert "ALPACA_API_KEY" in response.json()["detail"]

    def test_get_clock_alpaca_5xx_returns_503(
        self, client, mock_generic_alpaca_credentials
    ):
        """An upstream HTTP error surfaces as a 503 with no payload leak."""
        with patch("brain_api.routes.alpaca.httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value.__enter__.return_value = mock_client

            mock_response = MagicMock()
            mock_response.status_code = 502
            mock_client.get.side_effect = httpx.HTTPStatusError(
                "Bad Gateway", request=MagicMock(), response=mock_response
            )

            response = client.get("/alpaca/clock")

        assert response.status_code == 503
        assert "502" in response.json()["detail"]

    def test_get_clock_timeout_returns_503(
        self, client, mock_generic_alpaca_credentials
    ):
        """An upstream timeout surfaces as a 503."""
        with patch("brain_api.routes.alpaca.httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value.__enter__.return_value = mock_client
            mock_client.get.side_effect = httpx.TimeoutException("Connection timeout")

            response = client.get("/alpaca/clock")

        assert response.status_code == 503
        assert "timeout" in response.json()["detail"].lower()
