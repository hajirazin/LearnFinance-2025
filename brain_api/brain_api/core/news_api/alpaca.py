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
        if not self.api_key or not self.api_secret:
            logger.warning("Alpaca API credentials not configured")
            return []

        self._rate_limit()
        self._call_count += 1

        params = {
            "symbols": ",".join(symbols),
            "start": start.isoformat() + "Z" if start.tzinfo is None else start.isoformat(),
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
            data = response.json()
        except requests.RequestException as e:
            logger.error(f"Alpaca API request failed: {e}")
            return []

        articles = []
        for item in data.get("news", []):
            try:
                article = AlpacaNewsArticle(
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
                articles.append(article)
            except (KeyError, ValueError) as e:
                logger.warning(f"Failed to parse article: {e}")
                continue

        return articles

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

        while len(all_articles) < max_articles:
            if not self.api_key or not self.api_secret:
                break

            self._rate_limit()
            self._call_count += 1

            params = {
                "symbols": ",".join(symbols),
                "start": start.isoformat() + "Z" if start.tzinfo is None else start.isoformat(),
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
                data = response.json()
            except requests.RequestException as e:
                logger.error(f"Alpaca API request failed: {e}")
                break

            news_items = data.get("news", [])
            if not news_items:
                break

            for item in news_items:
                try:
                    article = AlpacaNewsArticle(
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
                    all_articles.append(article)
                except (KeyError, ValueError) as e:
                    logger.warning(f"Failed to parse article: {e}")
                    continue

            # Check for next page
            page_token = data.get("next_page_token")
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

