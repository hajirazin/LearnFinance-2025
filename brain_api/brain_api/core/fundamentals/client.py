"""Alpha Vantage API client."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from brain_api.core.fundamentals.index import FundamentalsIndex


class AlphaVantageProviderError(RuntimeError):
    """Raised when Alpha Vantage cannot provide a checked statement response."""


class AlphaVantageClient(Protocol):
    """Protocol for Alpha Vantage API client."""

    def fetch_income_statement(self, symbol: str) -> dict[str, Any]:
        """Fetch income statement data for a symbol."""
        ...

    def fetch_balance_sheet(self, symbol: str) -> dict[str, Any]:
        """Fetch balance sheet data for a symbol."""
        ...


class RealAlphaVantageClient:
    """Real Alpha Vantage API client with rate limiting."""

    def __init__(
        self,
        api_key: str,
        index: FundamentalsIndex,
        daily_limit: int = 25,
        request_delay: float = 12.0,  # ~5 requests/minute for free tier
    ):
        """Initialize the client.

        Args:
            api_key: Alpha Vantage API key
            index: FundamentalsIndex for optional call counting
            daily_limit: Informational only (no local pre-gate)
            request_delay: Seconds to wait between requests
        """
        self.api_key = api_key
        self.index = index
        self.daily_limit = daily_limit
        self.request_delay = request_delay
        self._last_request_time: float = 0

    def _rate_limit(self) -> None:
        """Enforce inter-request spacing (not a daily quota gate)."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self._last_request_time = time.time()

    def _fetch_endpoint(self, function: str, symbol: str) -> dict[str, Any]:
        """Fetch data from Alpha Vantage API.

        Args:
            function: API function name (INCOME_STATEMENT, BALANCE_SHEET)
            symbol: Stock ticker

        Returns:
            Provider-checked API response.
        """
        import requests

        # No local daily_limit pre-gate: call AV; fail when AV rejects.
        self._rate_limit()

        url = "https://www.alphavantage.co/query"
        params = {
            "function": function,
            "symbol": symbol,
            "apikey": self.api_key,
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            raise AlphaVantageProviderError(
                f"Alpha Vantage request failed for {function} {symbol}: {exc}"
            ) from exc
        except ValueError as exc:
            raise AlphaVantageProviderError(
                f"Alpha Vantage returned invalid JSON for {function} {symbol}"
            ) from exc

        if not isinstance(data, dict):
            raise AlphaVantageProviderError(
                f"Alpha Vantage returned a non-object payload for {function} {symbol}"
            )
        provider_error = next(
            (
                str(data[key])
                for key in ("Error Message", "Note", "Information")
                if data.get(key)
            ),
            None,
        )
        if provider_error is not None:
            raise AlphaVantageProviderError(
                f"Alpha Vantage rejected {function} for {symbol}: {provider_error}"
            )

        self.index.increment_api_calls()
        return data

    def fetch_income_statement(self, symbol: str) -> dict[str, Any]:
        """Fetch income statement data for a symbol."""
        return self._fetch_endpoint("INCOME_STATEMENT", symbol)

    def fetch_balance_sheet(self, symbol: str) -> dict[str, Any]:
        """Fetch balance sheet data for a symbol."""
        return self._fetch_endpoint("BALANCE_SHEET", symbol)
