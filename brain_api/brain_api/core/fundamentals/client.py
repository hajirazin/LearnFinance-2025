"""Alpha Vantage API client."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from brain_api.core.fundamentals.index import FundamentalsIndex


class AlphaVantageClient(Protocol):
    """Protocol for Alpha Vantage API client."""

    def fetch_income_statement(self, symbol: str) -> dict[str, Any] | None:
        """Fetch income statement data for a symbol."""
        ...

    def fetch_balance_sheet(self, symbol: str) -> dict[str, Any] | None:
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
            index: FundamentalsIndex for rate limit tracking
            daily_limit: Maximum API calls per day
            request_delay: Seconds to wait between requests
        """
        self.api_key = api_key
        self.index = index
        self.daily_limit = daily_limit
        self.request_delay = request_delay
        self._last_request_time: float = 0

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self._last_request_time = time.time()

    def _check_daily_limit(self) -> bool:
        """Check if daily API limit has been reached.

        Returns:
            True if we can make more calls, False if limit reached
        """
        calls_today = self.index.get_api_calls_today()
        return calls_today < self.daily_limit

    def _fetch_endpoint(self, function: str, symbol: str) -> dict[str, Any] | None:
        """Fetch data from Alpha Vantage API.

        Args:
            function: API function name (INCOME_STATEMENT, BALANCE_SHEET)
            symbol: Stock ticker

        Returns:
            API response as dict, or None if rate limited/error
        """
        import requests

        if not self._check_daily_limit():
            return None

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

            # Check for error responses
            if "Error Message" in data or "Note" in data:
                # "Note" typically means rate limit hit
                return None

            self.index.increment_api_calls()
            return data

        except Exception:
            return None

    def fetch_income_statement(self, symbol: str) -> dict[str, Any] | None:
        """Fetch income statement data for a symbol."""
        return self._fetch_endpoint("INCOME_STATEMENT", symbol)

    def fetch_balance_sheet(self, symbol: str) -> dict[str, Any] | None:
        """Fetch balance sheet data for a symbol."""
        return self._fetch_endpoint("BALANCE_SHEET", symbol)
