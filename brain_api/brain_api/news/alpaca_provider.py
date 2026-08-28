"""Alpaca Benzinga-backed News API provider."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
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


@dataclass(frozen=True)
class WindowBatchFetch:
    articles_by_symbol: dict[str, list[ProviderArticle]]
    page_count: int
    empties_are_proven: bool


def _article_targets(item: dict, requested: set[str]) -> list[str]:
    raw = item.get("symbols")
    names: list[str] = []
    if isinstance(raw, str):
        names = [part.strip() for part in raw.split(",") if part.strip()]
    elif isinstance(raw, list):
        names = [str(part).strip() for part in raw if str(part).strip()]
    return [name for name in names if name in requested]


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
        symbols: str,
        window: NewsWindow,
        page_token: str | None,
    ) -> tuple[list[dict], str | None]:
        params: dict[str, object] = {
            "symbols": symbols,
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
                    "Alpaca request error symbols=%s attempt=%s backoff=%.2fs err=%s",
                    symbols,
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
                    "Alpaca HTTP %s symbols=%s attempt=%s backoff=%.2fs",
                    response.status_code,
                    symbols,
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
                symbols=symbol, window=window, page_token=page_token
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

    def fetch_window_batch(
        self, symbols: Sequence[str], window: NewsWindow
    ) -> WindowBatchFetch:
        requested = tuple(dict.fromkeys(symbols))
        if not requested:
            return WindowBatchFetch({}, 0, True)
        requested_set = set(requested)
        articles_by_symbol: dict[str, list[ProviderArticle]] = {
            symbol: [] for symbol in requested
        }
        unique_ids: dict[str, set[str]] = {symbol: set() for symbol in requested}
        page_token: str | None = None
        seen_tokens: set[str] = set()
        page_count = 0
        last_page_len = 0
        truncated = False
        leftover_token: str | None = None
        while True:
            items, next_token = self._get_page(
                symbols=",".join(requested), window=window, page_token=page_token
            )
            page_count += 1
            last_page_len = len(items)
            leftover_token = next_token
            logger.info(
                "Alpaca batch page symbol_count=%s page=%s page_token_present=%s "
                "articles_on_page=%s",
                len(requested),
                page_count,
                bool(next_token),
                len(items),
            )
            capped = False
            for item in items:
                if not isinstance(item, dict):
                    raise NewsProviderError(
                        "Alpaca news collection contained an unusable article"
                    )
                for target in _article_targets(item, requested_set):
                    article = self._parse_article(item, target)
                    if article.provider_article_id in unique_ids[target]:
                        continue
                    unique_ids[target].add(article.provider_article_id)
                    if len(unique_ids[target]) > MAX_ARTICLES_PER_SYMBOL_WINDOW:
                        unique_ids[target].remove(article.provider_article_id)
                        capped = True
                        continue
                    if self._in_created_window(article.created_at, window):
                        articles_by_symbol[target].append(article)
            if capped:
                truncated = True
                break
            if not next_token:
                leftover_token = None
                break
            if next_token in seen_tokens:
                raise RepeatedPageTokenError(
                    f"repeated Alpaca page token for {','.join(requested)}: {next_token}"
                )
            seen_tokens.add(next_token)
            page_token = next_token
        last_page_short = last_page_len < PAGE_LIMIT
        empties_are_proven = (
            (not truncated) and leftover_token is None and last_page_short
        )
        return WindowBatchFetch(
            articles_by_symbol=articles_by_symbol,
            page_count=page_count,
            empties_are_proven=empties_are_proven,
        )
