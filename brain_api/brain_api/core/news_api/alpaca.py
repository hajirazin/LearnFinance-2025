"""Alpaca News API client.

Fetches financial news from Alpaca's Benzinga-powered news API.
Historical data available from 2015 onwards.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime

import requests

from brain_api.core.config import get_alpaca_api_key, get_alpaca_api_secret

logger = logging.getLogger(__name__)

# Alpaca News API earliest data date
ALPACA_EARLIEST_DATE = datetime(2015, 1, 1).date()


class AlpacaNewsProviderError(RuntimeError):
    """Raised when Alpaca cannot confirm a usable news response."""


@dataclass
class AlpacaNewsArticle:
    """A news article from Alpaca."""

    id: str
    headline: str
    summary: str
    author: str | None
    created_at: datetime
    updated_at: datetime
    url: str
    symbols: list[str]
    source: str

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "headline": self.headline,
            "summary": self.summary,
            "author": self.author,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "url": self.url,
            "symbols": self.symbols,
            "source": self.source,
        }


class AlpacaNewsClient:
    """Client for Alpaca News API.

    Rate limits:
    - Free tier: 200 calls/minute
    - Unlimited tier: 10,000 calls/minute

    Historical data available from 2015 onwards.
    """

    BASE_URL = "https://data.alpaca.markets/v1beta1/news"

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        rate_limit_delay: float = 0.3,  # ~200 calls/min
    ):
        """Initialize the client.

        Args:
            api_key: Alpaca API key (defaults to env var)
            api_secret: Alpaca API secret (defaults to env var)
            rate_limit_delay: Delay between requests in seconds
        """
        self.api_key = api_key or get_alpaca_api_key()
        self.api_secret = api_secret or get_alpaca_api_secret()
        self.rate_limit_delay = rate_limit_delay
        self._last_request_time: float = 0
        self._call_count: int = 0

    def _get_headers(self) -> dict[str, str]:
        """Get request headers with authentication."""
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
        }

    def _require_credentials(self) -> None:
        if not self.api_key or not self.api_secret:
            raise AlpacaNewsProviderError("Alpaca API credentials not configured")

    @staticmethod
    def _parse_response(response: requests.Response) -> tuple[list[dict], str | None]:
        try:
            data = response.json()
        except ValueError as exc:
            raise AlpacaNewsProviderError("Alpaca returned invalid JSON") from exc

        if not isinstance(data, dict):
            raise AlpacaNewsProviderError("Alpaca returned a non-object response")
        if "news" not in data or not isinstance(data["news"], list):
            raise AlpacaNewsProviderError(
                "Alpaca response did not contain a usable news collection"
            )
        if not all(isinstance(item, dict) for item in data["news"]):
            raise AlpacaNewsProviderError(
                "Alpaca news collection contained an unusable article"
            )
        return data["news"], data.get("next_page_token")

    @staticmethod
    def _parse_articles(news_items: list[dict]) -> list[AlpacaNewsArticle]:
        articles = []
        try:
            for item in news_items:
                articles.append(
                    AlpacaNewsArticle(
                        id=item.get("id", ""),
                        headline=item.get("headline", ""),
                        summary=item.get("summary", ""),
                        author=item.get("author"),
                        created_at=datetime.fromisoformat(
                            item["created_at"].replace("Z", "+00:00")
                        ),
                        updated_at=datetime.fromisoformat(
                            item["updated_at"].replace("Z", "+00:00")
                        ),
                        url=item.get("url", ""),
                        symbols=item.get("symbols", []),
                        source=item.get("source", ""),
                    )
                )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise AlpacaNewsProviderError(
                "Alpaca returned an unusable news article"
            ) from exc
        return articles

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    def fetch_news(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        limit: int = 50,
    ) -> list[AlpacaNewsArticle]:
        """Fetch news for symbols in date range.

        Args:
            symbols: List of stock symbols
            start: Start datetime (inclusive)
            end: End datetime (inclusive)
            limit: Maximum articles to return per call

        Returns:
            List of AlpacaNewsArticle objects
        """
        self._require_credentials()

        self._rate_limit()
        self._call_count += 1

        logger.info(
            f"Alpaca API call #{self._call_count}: "
            f"symbols={symbols[:3]}{'...' if len(symbols) > 3 else ''}, "
            f"date={start.date()}"
        )

        params = {
            "symbols": ",".join(symbols),
            "start": start.isoformat() + "Z"
            if start.tzinfo is None
            else start.isoformat(),
            "end": end.isoformat() + "Z" if end.tzinfo is None else end.isoformat(),
            "limit": limit,
            "sort": "desc",  # Most recent first
        }

        try:
            response = requests.get(
                self.BASE_URL,
                headers=self._get_headers(),
                params=params,
                timeout=30,
            )
            response.raise_for_status()
            news_items, _ = self._parse_response(response)
        except requests.RequestException as e:
            logger.error(f"Alpaca API call #{self._call_count} FAILED: {e}")
            raise AlpacaNewsProviderError("Alpaca news request failed") from e

        logger.info(
            "Alpaca API call #%d: returned %d articles",
            self._call_count,
            len(news_items),
        )
        return self._parse_articles(news_items)

    def fetch_news_for_date(
        self,
        symbols: list[str],
        target_date: datetime,
        limit: int = 50,
    ) -> list[AlpacaNewsArticle]:
        """Fetch news for a specific date.

        Args:
            symbols: List of stock symbols
            target_date: The date to fetch news for
            limit: Maximum articles to return

        Returns:
            List of AlpacaNewsArticle objects
        """
        # Create datetime range for the full day
        if isinstance(target_date, datetime):
            day_start = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            day_start = datetime.combine(target_date, datetime.min.time())

        day_end = day_start.replace(hour=23, minute=59, second=59, microsecond=999999)

        return self.fetch_news(symbols, day_start, day_end, limit)

    def fetch_news_batch(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        max_articles: int = 1000,
    ) -> list[AlpacaNewsArticle]:
        """Fetch news with pagination for larger date ranges.

        Args:
            symbols: List of stock symbols
            start: Start datetime
            end: End datetime
            max_articles: Maximum total articles to fetch

        Returns:
            List of AlpacaNewsArticle objects
        """
        all_articles = []
        page_token = None
        limit_per_call = min(50, max_articles)
        self._require_credentials()

        while len(all_articles) < max_articles:
            self._rate_limit()
            self._call_count += 1

            logger.info(
                f"Alpaca API batch call #{self._call_count}: "
                f"symbols={symbols[:3]}{'...' if len(symbols) > 3 else ''}, "
                f"fetched_so_far={len(all_articles)}"
            )

            params = {
                "symbols": ",".join(symbols),
                "start": start.isoformat() + "Z"
                if start.tzinfo is None
                else start.isoformat(),
                "end": end.isoformat() + "Z" if end.tzinfo is None else end.isoformat(),
                "limit": limit_per_call,
                "sort": "desc",
            }
            if page_token:
                params["page_token"] = page_token

            try:
                response = requests.get(
                    self.BASE_URL,
                    headers=self._get_headers(),
                    params=params,
                    timeout=30,
                )
                response.raise_for_status()
                news_items, next_page_token = self._parse_response(response)
            except requests.RequestException as e:
                logger.error(f"Alpaca API batch call #{self._call_count} FAILED: {e}")
                raise AlpacaNewsProviderError("Alpaca batch news request failed") from e

            logger.info(
                "Alpaca API batch call #%d: returned %d articles",
                self._call_count,
                len(news_items),
            )
            if not news_items:
                break

            all_articles.extend(self._parse_articles(news_items))

            # Check for next page
            page_token = next_page_token
            if not page_token:
                break

        return all_articles[:max_articles]

    @property
    def call_count(self) -> int:
        """Number of API calls made."""
        return self._call_count

    def reset_call_count(self) -> None:
        """Reset the API call counter."""
        self._call_count = 0
