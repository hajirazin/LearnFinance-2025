"""Alpaca Benzinga-backed News API provider."""

from __future__ import annotations

import logging
import time
from datetime import datetime

import requests

from brain_api.core.config import get_alpaca_api_key, get_alpaca_api_secret
from brain_api.news.errors import (
    NewsCapExceeded,
    NewsProviderError,
    RepeatedPageTokenError,
)
from brain_api.news.models import (
    MAX_ARTICLES_PER_SYMBOL_WINDOW,
    NewsWindow,
    ProviderArticle,
)

logger = logging.getLogger(__name__)

ALPACA_NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
PAGE_LIMIT = 50
MAX_RETRIES = 3


class AlpacaNewsProvider:
    """Single production news provider. Inclusive fetch, local created_at filter."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_secret: str | None = None,
        rate_limit_delay: float = 0.3,
        session: requests.Session | None = None,
    ) -> None:
        self.api_key = api_key or get_alpaca_api_key()
        self.api_secret = api_secret or get_alpaca_api_secret()
        self.rate_limit_delay = rate_limit_delay
        self._session = session or requests.Session()
        self._last_request_time = 0.0

    def _headers(self) -> dict[str, str]:
        if not self.api_key or not self.api_secret:
            raise NewsProviderError("Alpaca API credentials not configured")
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
        }

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    @staticmethod
    def _in_created_window(created_at: datetime, window: NewsWindow) -> bool:
        if created_at.tzinfo is None:
            raise NewsProviderError("Alpaca article timestamp is timezone-naive")
        created = created_at.astimezone(window.end_inclusive.tzinfo)
        start = window.start_exclusive.astimezone(window.end_inclusive.tzinfo)
        end = window.end_inclusive
        return start < created <= end

    def _get_page(
        self,
        *,
        symbol: str,
        window: NewsWindow,
        page_token: str | None,
    ) -> tuple[list[dict], str | None]:
        params: dict[str, object] = {
            "symbols": symbol,
            "start": window.start_exclusive.isoformat(),
            "end": window.end_inclusive.isoformat(),
            "limit": PAGE_LIMIT,
            "sort": "desc",
        }
        if page_token:
            params["page_token"] = page_token
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES + 1):
            self._rate_limit()
            try:
                response = self._session.get(
                    ALPACA_NEWS_URL,
                    headers=self._headers(),
                    params=params,
                    timeout=30,
                )
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= MAX_RETRIES:
                    raise NewsProviderError("Alpaca news request failed") from exc
                backoff = 0.5 * (2**attempt)
                logger.warning(
                    "Alpaca request error symbol=%s attempt=%s backoff=%.2fs err=%s",
                    symbol,
                    attempt + 1,
                    backoff,
                    exc,
                )
                time.sleep(backoff)
                continue
            if response.status_code == 429 or 500 <= response.status_code < 600:
                last_error = NewsProviderError(
                    f"Alpaca news HTTP {response.status_code}"
                )
                if attempt >= MAX_RETRIES:
                    raise last_error
                backoff = 0.5 * (2**attempt)
                logger.warning(
                    "Alpaca HTTP %s symbol=%s attempt=%s backoff=%.2fs",
                    response.status_code,
                    symbol,
                    attempt + 1,
                    backoff,
                )
                time.sleep(backoff)
                continue
            try:
                response.raise_for_status()
                payload = response.json()
            except (ValueError, requests.RequestException) as exc:
                raise NewsProviderError(
                    "Alpaca returned an unusable news response"
                ) from exc
            if not isinstance(payload, dict) or not isinstance(
                payload.get("news"), list
            ):
                raise NewsProviderError(
                    "Alpaca response did not contain a news collection"
                )
            return payload["news"], payload.get("next_page_token")
        raise NewsProviderError("Alpaca news request failed") from last_error

    @staticmethod
    def _parse_article(item: dict, symbol: str) -> ProviderArticle:
        try:
            created_at = datetime.fromisoformat(
                str(item["created_at"]).replace("Z", "+00:00")
            )
            updated_at = datetime.fromisoformat(
                str(item["updated_at"]).replace("Z", "+00:00")
            )
            if created_at.tzinfo is None or updated_at.tzinfo is None:
                raise NewsProviderError("Alpaca article timestamp is timezone-naive")
            article_id = str(item.get("id") or "")
            if not article_id:
                raise KeyError("id")
            return ProviderArticle(
                provider_article_id=article_id,
                symbol=symbol,
                created_at=created_at,
                updated_at=updated_at,
                source=str(item.get("source") or ""),
                headline=str(item.get("headline") or ""),
                summary=str(item.get("summary") or ""),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise NewsProviderError("Alpaca returned an unusable news article") from exc

    def fetch_window(
        self, symbol: str, window: NewsWindow
    ) -> tuple[list[ProviderArticle], int]:
        articles: list[ProviderArticle] = []
        page_token: str | None = None
        seen_tokens: set[str] = set()
        seen_ids: set[str] = set()
        page_count = 0
        while True:
            items, next_token = self._get_page(
                symbol=symbol, window=window, page_token=page_token
            )
            page_count += 1
            is_last = not next_token
            if page_count == 1 or is_last or page_count % 10 == 0:
                logger.info(
                    "Alpaca page symbol=%s page=%s page_token_present=%s articles_on_page=%s",
                    symbol,
                    page_count,
                    bool(next_token),
                    len(items),
                )
            for item in items:
                if not isinstance(item, dict):
                    raise NewsProviderError(
                        "Alpaca news collection contained an unusable article"
                    )
                article = self._parse_article(item, symbol)
                seen_ids.add(article.provider_article_id)
                if len(seen_ids) > MAX_ARTICLES_PER_SYMBOL_WINDOW:
                    raise NewsCapExceeded(
                        f"{symbol} has {len(seen_ids)} unique articles in window "
                        f"(cap {MAX_ARTICLES_PER_SYMBOL_WINDOW})"
                    )
                if self._in_created_window(article.created_at, window):
                    articles.append(article)
            if not next_token:
                break
            if next_token in seen_tokens:
                raise RepeatedPageTokenError(
                    f"repeated Alpaca page token for {symbol}: {next_token}"
                )
            seen_tokens.add(next_token)
            page_token = next_token
        return articles, page_count
