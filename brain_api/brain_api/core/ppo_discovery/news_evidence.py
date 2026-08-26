"""Mandatory Alpaca/Benzinga news evidence for ppo_discovery.

Live ``POST /signals/news`` (yfinance) is never called. Incomplete
pagination, HTTP errors, missing ``created_at``, FinBERT exceptions, and
article-cap hits abort the week. Verified empty queries are valid zeros.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from brain_api.core.finbert import FINBERT_MODEL, FinBERTScorer, SentimentScore
from brain_api.core.news_api.alpaca import (
    AlpacaNewsArticle,
    AlpacaNewsClient,
    AlpacaNewsProviderError,
)
from brain_api.core.ppo_discovery.config import (
    ARTICLE_PAGE_CAP,
    NEWS_RECENCY_TAU_HOURS,
)
from brain_api.core.ppo_discovery.schemas import (
    PPODiscoveryError,
    SymbolNewsFeatures,
    sha256_digest,
)
from brain_api.storage.base import DEFAULT_DATA_PATH

PROVIDER_NAME = "alpaca_benzinga"
FINBERT_REVISION = FINBERT_MODEL


class NewsEvidenceError(PPODiscoveryError):
    """Raised when a news query is incomplete or unscorable."""


@dataclass(frozen=True)
class NewsWindow:
    """Half-open window ``(start, end]`` in UTC."""

    start: datetime
    end: datetime


def news_window_for_cutoff(
    cutoff: datetime, previous_cutoff: datetime | None
) -> NewsWindow:
    """Build the news availability window for one decision cutoff."""
    end = cutoff.astimezone(UTC)
    if previous_cutoff is None:
        start = end - timedelta(days=7)
    else:
        start = previous_cutoff.astimezone(UTC)
    if start >= end:
        raise NewsEvidenceError("news window start must be before cutoff")
    return NewsWindow(start=start, end=end)


def article_dedupe_key(article: AlpacaNewsArticle) -> str:
    if article.id:
        return f"id:{article.id}"
    payload = f"{article.url}|{article.created_at.isoformat()}|{article.headline}"
    return "hash:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _in_window(created_at: datetime, window: NewsWindow) -> bool:
    created = created_at.astimezone(UTC)
    return window.start < created <= window.end


def fetch_news_exhaustive(
    client: AlpacaNewsClient,
    symbols: Sequence[str],
    window: NewsWindow,
    *,
    article_cap: int = ARTICLE_PAGE_CAP,
    fetch_page: Callable[..., tuple[list[AlpacaNewsArticle], str | None]] | None = None,
) -> tuple[list[AlpacaNewsArticle], dict[str, Any]]:
    """Paginate until ``next_page_token`` is absent. Cap hit is a failure."""
    if not symbols:
        raise NewsEvidenceError("news fetch requires at least one symbol")
    pages = 0
    token: str | None = None
    collected: list[AlpacaNewsArticle] = []
    page_tokens: list[str] = []
    while True:
        pages += 1
        try:
            if fetch_page is not None:
                articles, token = fetch_page(
                    symbols=list(symbols),
                    start=window.start,
                    end=window.end,
                    page_token=token,
                )
            else:
                articles, token = _client_page(client, list(symbols), window, token)
        except AlpacaNewsProviderError as exc:
            raise NewsEvidenceError(f"Alpaca news provider failure: {exc}") from exc
        except Exception as exc:
            raise NewsEvidenceError(f"Alpaca news request failed: {exc}") from exc
        collected.extend(articles)
        if token:
            page_tokens.append(token)
        if len(collected) > article_cap or (token and len(collected) >= article_cap):
            raise NewsEvidenceError(
                f"news article cap {article_cap} hit with remaining page token"
            )
        if not token:
            break
    manifest = {
        "provider": PROVIDER_NAME,
        "symbol_count": len(symbols),
        "pages": pages,
        "raw_article_count": len(collected),
        "page_token_hashes": [
            hashlib.sha256(value.encode()).hexdigest()[:16] for value in page_tokens
        ],
        "window_start": window.start.isoformat(),
        "window_end": window.end.isoformat(),
    }
    return collected, manifest


def _client_page(
    client: AlpacaNewsClient,
    symbols: list[str],
    window: NewsWindow,
    page_token: str | None,
) -> tuple[list[AlpacaNewsArticle], str | None]:
    """One authenticated Alpaca page. Does not truncate remaining pages."""
    client._require_credentials()
    client._rate_limit()
    client._call_count += 1
    params: dict[str, Any] = {
        "symbols": ",".join(symbols),
        "start": window.start.isoformat(),
        "end": window.end.isoformat(),
        "limit": 50,
        "sort": "desc",
    }
    if page_token:
        params["page_token"] = page_token
    import requests

    try:
        response = requests.get(
            client.BASE_URL,
            headers=client._get_headers(),
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        news_items, next_token = client._parse_response(response)
    except requests.RequestException as exc:
        raise AlpacaNewsProviderError("Alpaca news request failed") from exc
    return client._parse_articles(news_items), next_token


def score_texts_or_abort(
    scorer: FinBERTScorer, texts: Sequence[str]
) -> list[SentimentScore]:
    """Score texts with FinBERT; abort instead of using the silent-neutral fallback."""
    if not texts:
        return []
    try:
        with scorer._inference_lock:
            scorer._ensure_loaded()
            raw = scorer._pipeline(list(texts))
    except Exception as exc:
        raise NewsEvidenceError(f"FinBERT scoring failed: {exc}") from exc
    scores: list[SentimentScore] = []
    for item in raw:
        p_pos = p_neg = p_neu = 0.0
        for row in item:
            label = str(row["label"]).lower()
            prob = float(row["score"])
            if label == "positive":
                p_pos = prob
            elif label == "negative":
                p_neg = prob
            elif label == "neutral":
                p_neu = prob
        if p_pos >= p_neg and p_pos >= p_neu:
            label = "positive"
        elif p_neg >= p_pos and p_neg >= p_neu:
            label = "negative"
        else:
            label = "neutral"
        scores.append(
            SentimentScore(
                label=label,
                p_pos=round(p_pos, 4),
                p_neg=round(p_neg, 4),
                p_neu=round(p_neu, 4),
                score=round(p_pos - p_neg, 4),
                confidence=round(max(p_pos, p_neg, p_neu), 4),
            )
        )
    return scores


def aggregate_symbol_news(
    symbol: str,
    articles: Sequence[AlpacaNewsArticle],
    scores: Sequence[SentimentScore],
    cutoff: datetime,
    *,
    query_complete: bool,
    request_manifest_sha256: str,
) -> SymbolNewsFeatures:
    """Build compact + audit news features for one symbol."""
    if not query_complete:
        raise NewsEvidenceError(f"incomplete news query for {symbol}")
    if len(articles) != len(scores):
        raise NewsEvidenceError("article/score length mismatch")
    cutoff_utc = cutoff.astimezone(UTC)
    if not articles:
        return SymbolNewsFeatures(
            symbol=symbol,
            raw_sentiment=0.0,
            article_count=0,
            average_confidence=0.0,
            sentiment_dispersion=0.0,
            hours_since_latest=0.0,
            unique_source_count=0,
            has_news=0,
            query_complete=True,
            news_recency=0.0,
            log1p_article_count=0.0,
            article_ids_sha256=sha256_digest([]),
            request_manifest_sha256=request_manifest_sha256,
        )
    signed = np.asarray([score.score for score in scores], dtype=np.float64)
    confidences = np.asarray([score.confidence for score in scores], dtype=np.float64)
    weights = confidences
    weight_sum = float(weights.sum())
    raw = 0.0 if weight_sum <= 0 else float(np.dot(signed, weights) / weight_sum)
    latest = max(article.created_at.astimezone(UTC) for article in articles)
    hours = max((cutoff_utc - latest).total_seconds() / 3600.0, 0.0)
    recency = math.exp(-hours / NEWS_RECENCY_TAU_HOURS)
    dispersion = float(np.std(signed, ddof=0)) if len(signed) >= 2 else 0.0
    sources = {article.source.strip().lower() for article in articles if article.source}
    ids = [article_dedupe_key(article) for article in articles]
    return SymbolNewsFeatures(
        symbol=symbol,
        raw_sentiment=raw,
        article_count=len(articles),
        average_confidence=float(confidences.mean()),
        sentiment_dispersion=dispersion,
        hours_since_latest=hours,
        unique_source_count=len(sources),
        has_news=1,
        query_complete=True,
        news_recency=recency,
        log1p_article_count=float(np.log1p(len(articles))),
        article_ids_sha256=sha256_digest(ids),
        request_manifest_sha256=request_manifest_sha256,
    )


def assign_articles_to_symbols(
    articles: Sequence[AlpacaNewsArticle], requested: Sequence[str]
) -> dict[str, list[AlpacaNewsArticle]]:
    """Deduplicate globally then attach each article to requested symbols it tags."""
    seen: set[str] = set()
    unique: list[AlpacaNewsArticle] = []
    for article in articles:
        key = article_dedupe_key(article)
        if key in seen:
            continue
        seen.add(key)
        unique.append(article)
    wanted = {symbol: [] for symbol in requested}
    requested_set = set(requested)
    for article in unique:
        for symbol in article.symbols:
            if symbol in requested_set:
                wanted[symbol].append(article)
    return wanted


def materialize_news_evidence(
    symbols: Sequence[str],
    cutoff: datetime,
    *,
    previous_cutoff: datetime | None = None,
    client: AlpacaNewsClient | None = None,
    scorer: FinBERTScorer | None = None,
    fetch_page: Callable[..., tuple[list[AlpacaNewsArticle], str | None]] | None = None,
    score_fn: Callable[[Sequence[str]], list[SentimentScore]] | None = None,
) -> dict[str, SymbolNewsFeatures]:
    """Fetch, score, and aggregate news for every requested symbol."""
    window = news_window_for_cutoff(cutoff, previous_cutoff)
    client = client or AlpacaNewsClient()
    collected, manifest = fetch_news_exhaustive(
        client, symbols, window, fetch_page=fetch_page
    )
    for article in collected:
        if article.created_at is None:
            raise NewsEvidenceError("article missing created_at")
        try:
            article.created_at.astimezone(UTC)
        except Exception as exc:
            raise NewsEvidenceError("unparseable created_at") from exc
    in_window = [
        article for article in collected if _in_window(article.created_at, window)
    ]
    by_symbol = assign_articles_to_symbols(in_window, symbols)
    scorer = scorer or FinBERTScorer()
    scorer_fn = score_fn or (lambda texts: score_texts_or_abort(scorer, texts))
    request_hash = sha256_digest(manifest)
    features: dict[str, SymbolNewsFeatures] = {}
    for symbol in symbols:
        symbol_articles = by_symbol[symbol]
        texts = [
            f"{article.headline} {article.summary or ''}".strip()
            for article in symbol_articles
        ]
        scores = scorer_fn(texts) if texts else []
        features[symbol] = aggregate_symbol_news(
            symbol,
            symbol_articles,
            scores,
            cutoff,
            query_complete=True,
            request_manifest_sha256=request_hash,
        )
    return features


def persist_weekly_news_features(
    cutoff: datetime,
    features: dict[str, SymbolNewsFeatures],
    *,
    window: NewsWindow,
    base_path: Path | str | None = None,
) -> Path:
    """Append audit rows to the PPO-specific weekly parquet."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = (
        Path(base_path or DEFAULT_DATA_PATH)
        / "ppo_discovery"
        / "news"
        / "weekly_features.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for symbol, row in sorted(features.items()):
        rows.append(
            {
                "decision_cutoff": cutoff.astimezone(UTC).isoformat(),
                "symbol": symbol,
                "window_start": window.start.isoformat(),
                "window_end": window.end.isoformat(),
                "raw_sentiment": row.raw_sentiment,
                "article_count": row.article_count,
                "average_confidence": row.average_confidence,
                "sentiment_dispersion": row.sentiment_dispersion,
                "hours_since_latest": row.hours_since_latest,
                "unique_source_count": row.unique_source_count,
                "has_news": row.has_news,
                "query_complete": int(row.query_complete),
                "provider": PROVIDER_NAME,
                "finbert_revision": FINBERT_REVISION,
                "article_ids_sha256": row.article_ids_sha256,
                "request_manifest_sha256": row.request_manifest_sha256,
            }
        )
    table = pa.Table.from_pylist(rows)
    if path.exists():
        existing = pq.read_table(path)
        table = pa.concat_tables([existing, table])
    pq.write_table(table, path)
    return path


__all__ = [
    "NewsEvidenceError",
    "NewsWindow",
    "aggregate_symbol_news",
    "article_dedupe_key",
    "assign_articles_to_symbols",
    "fetch_news_exhaustive",
    "materialize_news_evidence",
    "news_window_for_cutoff",
    "persist_weekly_news_features",
    "score_texts_or_abort",
]
