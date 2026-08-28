"""Materialize and query news windows. No RL imports."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence

from brain_api.news.alpaca_provider import (
    AlpacaNewsProvider,
    WindowBatchFetch,
)
from brain_api.news.coalescing import COORDINATOR, _InFlight, materialization_key
from brain_api.news.errors import (
    NewsCapExceeded,
    NewsCoverageMissing,
    NewsError,
    NewsProviderError,
    NewsWindowNotClosed,
    SentimentScoringError,
)
from brain_api.news.hashing import request_manifest_hash
from brain_api.news.models import (
    MAX_ARTICLES_PER_REQUEST,
    MAX_ARTICLES_PER_SYMBOL_WINDOW,
    NEWS_PROVIDER,
    NEWS_SCHEMA_VERSION,
    NEWS_SENTIMENT_MODEL,
    NEWS_SENTIMENT_REVISION,
    SCORING_SCHEMA_VERSION,
    NewsCoverage,
    NewsEvent,
    NewsWindow,
    ProviderArticle,
)
from brain_api.news.provider import NewsProvider
from brain_api.news.sentiment import (
    StrictFinBERTScorer,
    scored_text_sha256,
    try_assemble_scored_text,
)
from brain_api.news.store import NewsStore, utcnow

logger = logging.getLogger(__name__)


class NewsService:
    """Fetch, score, persist, and query exact coverage windows."""

    def __init__(
        self,
        store: NewsStore,
        *,
        provider: NewsProvider | None = None,
        scorer: StrictFinBERTScorer | None = None,
    ) -> None:
        self.store = store
        self.provider = provider or AlpacaNewsProvider()
        self.scorer = scorer or StrictFinBERTScorer()

    def query(
        self, symbols: Sequence[str], window: NewsWindow
    ) -> tuple[list[NewsCoverage], list[NewsEvent]]:
        logger.info(
            "news query start symbols=%s start=%s end=%s",
            len(symbols),
            window.start_exclusive.isoformat(),
            window.end_inclusive.isoformat(),
        )
        coverage = self.store.require_coverage(symbols, window)
        events = self.store.query_events(symbols, window)
        logger.info(
            "news query end symbols=%s events=%s",
            len(symbols),
            len(events),
        )
        return coverage, events

    def materialize(
        self, symbols: Sequence[str], window: NewsWindow
    ) -> tuple[list[NewsCoverage], list[NewsEvent]]:
        now = utcnow()
        if window.end_inclusive.astimezone(now.tzinfo) > now:
            raise NewsWindowNotClosed(
                "cannot materialize a news window whose end is still in the future: "
                f"end_inclusive={window.end_inclusive.isoformat()} now={now.isoformat()}"
            )
        unique_symbols = list(dict.fromkeys(symbols))
        key = materialization_key(window)
        covered = self.store.covered_symbols(unique_symbols, window)
        pending = [symbol for symbol in unique_symbols if symbol not in covered]
        coalesce = "hit" if not pending else "miss"
        logger.info(
            "news materialize start key=%s symbol_count=%s pending=%s coalesce=%s start=%s end=%s",
            f"{key.start_exclusive}|{key.end_inclusive}",
            len(unique_symbols),
            len(pending),
            coalesce,
            window.start_exclusive.isoformat(),
            window.end_inclusive.isoformat(),
        )
        started = time.perf_counter()
        if pending:

            def _worker(todo: list[str], job: _InFlight) -> None:
                self._materialize_symbols(todo, window, job)

            COORDINATOR.run(key, pending, _worker)
        coverage, events = self.query(unique_symbols, window)
        excluded = sum(row.future_revision_excluded_count for row in coverage)
        logger.info(
            "news materialize end symbols=%s events=%s excluded=%s elapsed_s=%.1f",
            len(unique_symbols),
            len(events),
            excluded,
            time.perf_counter() - started,
        )
        return coverage, events

    def _materialize_symbols(
        self, symbols: Sequence[str], window: NewsWindow, job: _InFlight | None
    ) -> None:
        total = len(symbols)
        unique_request_ids: set[str] = set()
        covered = self.store.covered_symbols(symbols, window)
        pending: list[str] = []
        for symbol in symbols:
            if symbol in covered:
                if job is not None:
                    with job.lock:
                        job.done.add(symbol)
                continue
            pending.append(symbol)
        batch_fetch = getattr(self.provider, "fetch_window_batch", None)
        if callable(batch_fetch):
            if pending:
                self._materialize_batch(
                    batch_fetch,
                    pending,
                    window,
                    job,
                    unique_request_ids,
                    total=total,
                    index_base=0,
                )
            return
        for index, symbol in enumerate(pending, start=1):
            self._materialize_one(
                symbol,
                window,
                job,
                unique_request_ids,
                index=index,
                total=total,
            )

    def _materialize_batch(
        self,
        batch_fetch,
        symbols: Sequence[str],
        window: NewsWindow,
        job: _InFlight | None,
        unique_request_ids: set[str],
        *,
        total: int,
        index_base: int,
    ) -> None:
        result: WindowBatchFetch = batch_fetch(symbols, window)
        ready: list[tuple[str, Sequence[ProviderArticle]]] = []
        misses: list[str] = []
        for symbol in symbols:
            articles = result.articles_by_symbol.get(symbol, [])
            if articles or result.empties_are_proven:
                ready.append((symbol, articles))
            else:
                misses.append(symbol)
        self._persist_ready_batch(
            ready,
            result.page_count,
            window,
            job,
            unique_request_ids,
            empties_are_proven=result.empties_are_proven,
        )
        for offset, symbol in enumerate(misses):
            self._materialize_one(
                symbol,
                window,
                job,
                unique_request_ids,
                index=index_base + offset + 1,
                total=total,
            )

    def _persist_ready_batch(
        self,
        ready: Sequence[tuple[str, Sequence[ProviderArticle]]],
        page_count: int,
        window: NewsWindow,
        job: _InFlight | None,
        unique_request_ids: set[str],
        *,
        empties_are_proven: bool,
    ) -> None:
        if not ready:
            return
        started = time.perf_counter()
        usable_by_symbol: dict[str, list[ProviderArticle]] = {}
        excluded_by_symbol: dict[str, int] = {}
        flat_usable: list[ProviderArticle] = []
        for symbol, articles in ready:
            unique_ids = {article.provider_article_id for article in articles}
            if len(unique_ids) > MAX_ARTICLES_PER_SYMBOL_WINDOW:
                raise NewsCapExceeded(
                    f"{symbol} has {len(unique_ids)} unique articles in window "
                    f"(cap {MAX_ARTICLES_PER_SYMBOL_WINDOW})"
                )
            unique_request_ids.update(unique_ids)
            if len(unique_request_ids) > MAX_ARTICLES_PER_REQUEST:
                raise NewsCapExceeded(
                    f"request has {len(unique_request_ids)} unique articles "
                    f"(cap {MAX_ARTICLES_PER_REQUEST})"
                )
            usable, excluded = self._partition_revisions(articles, window)
            usable_by_symbol[symbol] = usable
            excluded_by_symbol[symbol] = excluded
            flat_usable.extend(usable)
        all_events, cache_rows = self._score_articles(flat_usable)
        events_by_symbol: dict[str, list[NewsEvent]] = {
            symbol: [] for symbol, _ in ready
        }
        for event in all_events:
            events_by_symbol[event.symbol].append(event)
        fetched_at = utcnow()
        items: list[tuple[list[NewsEvent], NewsCoverage, list[tuple]]] = []
        complete = 0
        for symbol, _articles in ready:
            events = events_by_symbol[symbol]
            status = "verified_empty" if not events else "complete"
            if status == "complete":
                complete += 1
            items.append(
                (
                    events,
                    NewsCoverage(
                        provider=NEWS_PROVIDER,
                        symbol=symbol,
                        window_start_exclusive=window.start_exclusive,
                        window_end_inclusive=window.end_inclusive,
                        schema_version=NEWS_SCHEMA_VERSION,
                        sentiment_model=NEWS_SENTIMENT_MODEL,
                        sentiment_model_revision=NEWS_SENTIMENT_REVISION,
                        status=status,
                        page_count=page_count,
                        event_count=len(events),
                        future_revision_excluded_count=excluded_by_symbol[symbol],
                        fetched_at=fetched_at,
                        request_manifest_hash=request_manifest_hash([symbol], window),
                    ),
                    cache_rows if symbol == ready[0][0] else [],
                )
            )
        self.store.commit_windows(items)
        if job is not None:
            with job.lock:
                job.done.update(symbol for symbol, _articles in ready)
        logger.info(
            "news batch persist symbols=%s complete=%s verified_empty=%s "
            "events=%s pages=%s empties_proven=%s elapsed_ms=%.0f",
            len(ready),
            complete,
            len(ready) - complete,
            len(all_events),
            page_count,
            empties_are_proven,
            (time.perf_counter() - started) * 1000,
        )

    def _materialize_one(
        self,
        symbol: str,
        window: NewsWindow,
        job: _InFlight | None,
        unique_request_ids: set[str],
        *,
        index: int,
        total: int,
    ) -> None:
        if self.store.get_coverage(symbol, window) is not None:
            if job is not None:
                with job.lock:
                    job.done.add(symbol)
            return
        try:
            articles, page_count = self.provider.fetch_window(symbol, window)
        except Exception as exc:
            logger.error(
                "news materialize failed symbol=%s window_end=%s err=%s",
                symbol,
                window.end_inclusive.isoformat(),
                exc,
            )
            raise
        self._persist_fetched(
            symbol,
            articles,
            page_count,
            window,
            job,
            unique_request_ids,
            index=index,
            total=total,
        )

    def _persist_fetched(
        self,
        symbol: str,
        articles: Sequence[ProviderArticle],
        page_count: int,
        window: NewsWindow,
        job: _InFlight | None,
        unique_request_ids: set[str],
        *,
        index: int,
        total: int,
    ) -> None:
        symbol_started = time.perf_counter()
        unique_ids = {article.provider_article_id for article in articles}
        if len(unique_ids) > MAX_ARTICLES_PER_SYMBOL_WINDOW:
            raise NewsCapExceeded(
                f"{symbol} has {len(unique_ids)} unique articles in window "
                f"(cap {MAX_ARTICLES_PER_SYMBOL_WINDOW})"
            )
        unique_request_ids.update(unique_ids)
        if len(unique_request_ids) > MAX_ARTICLES_PER_REQUEST:
            raise NewsCapExceeded(
                f"request has {len(unique_request_ids)} unique articles "
                f"(cap {MAX_ARTICLES_PER_REQUEST})"
            )
        usable, excluded = self._partition_revisions(articles, window)
        events, cache_rows = self._score_articles(usable)
        status = "verified_empty" if not events else "complete"
        coverage = NewsCoverage(
            provider=NEWS_PROVIDER,
            symbol=symbol,
            window_start_exclusive=window.start_exclusive,
            window_end_inclusive=window.end_inclusive,
            schema_version=NEWS_SCHEMA_VERSION,
            sentiment_model=NEWS_SENTIMENT_MODEL,
            sentiment_model_revision=NEWS_SENTIMENT_REVISION,
            status=status,
            page_count=page_count,
            event_count=len(events),
            future_revision_excluded_count=excluded,
            fetched_at=utcnow(),
            request_manifest_hash=request_manifest_hash([symbol], window),
        )
        self.store.commit_window(
            events=events, coverage=coverage, cache_rows=cache_rows
        )
        if job is not None:
            with job.lock:
                job.done.add(symbol)
        logger.info(
            "news symbol=%s index=%s/%s pages=%s event_count=%s status=%s elapsed_ms=%.0f",
            symbol,
            index,
            total,
            page_count,
            len(events),
            status,
            (time.perf_counter() - symbol_started) * 1000,
        )

    @staticmethod
    def _partition_revisions(
        articles: Sequence[ProviderArticle], window: NewsWindow
    ) -> tuple[list[ProviderArticle], int]:
        latest: dict[str, ProviderArticle] = {}
        for article in articles:
            current = latest.get(article.provider_article_id)
            if current is None or article.updated_at > current.updated_at:
                latest[article.provider_article_id] = article
        usable: list[ProviderArticle] = []
        excluded = 0
        end = window.end_inclusive
        for article in latest.values():
            updated = article.updated_at.astimezone(end.tzinfo)
            if updated > end:
                excluded += 1
                continue
            usable.append(article)
        return usable, excluded

    def _score_articles(
        self, articles: Sequence[ProviderArticle]
    ) -> tuple[list[NewsEvent], list[tuple]]:
        if not articles:
            return [], []
        events: list[NewsEvent] = []
        cache_rows: list[tuple] = []
        prepared: list[tuple[ProviderArticle, str, str]] = []
        for article in articles:
            text = try_assemble_scored_text(article.headline, article.summary)
            if text is None:
                logger.warning(
                    "news skip unscorable article id=%s symbol=%s",
                    article.provider_article_id,
                    article.symbol,
                )
                continue
            prepared.append((article, text, scored_text_sha256(text)))
        cached = self.store.cache_get_many(
            [digest for _article, _text, digest in prepared]
        )
        to_score: list[tuple[ProviderArticle, str, str]] = []
        cache_hits = 0
        for article, text, digest in prepared:
            hit = cached.get(digest)
            if hit is not None:
                cache_hits += 1
                score, p_pos, p_neg, p_neu, conf = hit
                events.append(
                    self._event_from_score(
                        article, digest, score, p_pos, p_neg, p_neu, conf
                    )
                )
                continue
            to_score.append((article, text, digest))
        scored_started = time.perf_counter()
        if to_score:
            scored = self.scorer.score_texts([item[1] for item in to_score])
            seen_digests: set[str] = set()
            for (article, _text, digest), item in zip(to_score, scored, strict=True):
                events.append(
                    self._event_from_score(
                        article,
                        item.scored_text_sha256,
                        item.sentiment_score,
                        item.p_positive,
                        item.p_negative,
                        item.p_neutral,
                        item.confidence,
                    )
                )
                if digest in seen_digests:
                    continue
                seen_digests.add(digest)
                cache_rows.append(
                    (
                        digest,
                        NEWS_SENTIMENT_MODEL,
                        NEWS_SENTIMENT_REVISION,
                        SCORING_SCHEMA_VERSION,
                        item.sentiment_score,
                        item.p_positive,
                        item.p_negative,
                        item.p_neutral,
                        item.confidence,
                    )
                )
        logger.info(
            "FinBERT scored=%s cache_hits=%s elapsed_ms=%.0f",
            len(to_score),
            cache_hits,
            (time.perf_counter() - scored_started) * 1000 if to_score else 0,
        )
        return events, cache_rows

    @staticmethod
    def _event_from_score(
        article: ProviderArticle,
        digest: str,
        score: float,
        p_pos: float,
        p_neg: float,
        p_neu: float,
        confidence: float,
    ) -> NewsEvent:
        return NewsEvent(
            provider=NEWS_PROVIDER,
            provider_article_id=article.provider_article_id,
            symbol=article.symbol,
            created_at=article.created_at,
            updated_at=article.updated_at,
            source=article.source,
            sentiment_score=score,
            p_positive=p_pos,
            p_negative=p_neg,
            p_neutral=p_neu,
            confidence=confidence,
            scored_text_sha256=digest,
            sentiment_model=NEWS_SENTIMENT_MODEL,
            sentiment_model_revision=NEWS_SENTIMENT_REVISION,
            schema_version=NEWS_SCHEMA_VERSION,
            ingested_at=utcnow(),
        )


def raise_http_status(exc: BaseException) -> int:
    """Map domain errors to the locked HTTP statuses."""
    if isinstance(exc, NewsCapExceeded | NewsCoverageMissing | NewsWindowNotClosed):
        return 422
    if isinstance(exc, NewsProviderError | SentimentScoringError):
        return 503
    if isinstance(exc, NewsError):
        return 503
    return 503
