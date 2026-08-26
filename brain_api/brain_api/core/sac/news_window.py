"""Convert a NewsWindowResult DTO into SAC adapter inputs."""

from __future__ import annotations

from datetime import datetime

from brain_api.core.sac.news_adapter import build_sac_news_features
from brain_api.news.models import NewsEvent
from brain_api.routes.inference.models import SACNewsAudit, SACNewsSymbolAudit
from brain_api.routes.news.models import NewsWindowResult


def events_by_symbol_from_window(
    news_window: NewsWindowResult, symbols: list[str]
) -> dict[str, list[NewsEvent]]:
    grouped: dict[str, list[NewsEvent]] = {symbol: [] for symbol in symbols}
    for dto in news_window.events:
        if dto.symbol not in grouped:
            continue
        grouped[dto.symbol].append(
            NewsEvent(
                provider=dto.provider,
                provider_article_id=dto.provider_article_id,
                symbol=dto.symbol,
                created_at=dto.created_at,
                updated_at=dto.updated_at,
                source=dto.source,
                sentiment_score=dto.sentiment_score,
                p_positive=dto.p_positive,
                p_negative=dto.p_negative,
                p_neutral=dto.p_neutral,
                confidence=dto.confidence,
                scored_text_sha256=dto.scored_text_sha256,
                sentiment_model=dto.sentiment_model,
                sentiment_model_revision=dto.sentiment_model_revision,
                schema_version=dto.schema_version,
                ingested_at=dto.updated_at,
            )
        )
    return grouped


def sac_news_from_window(
    news_window: NewsWindowResult,
    *,
    symbols: list[str],
    cutoff: datetime,
) -> tuple[dict[str, float], dict[str, int], SACNewsAudit]:
    coverage = {item.symbol: item for item in news_window.coverage}
    missing = [symbol for symbol in symbols if symbol not in coverage]
    if missing:
        raise ValueError(f"news_window missing coverage for {missing}")
    events_by_symbol = events_by_symbol_from_window(news_window, symbols)
    status = {symbol: coverage[symbol].status for symbol in symbols}
    sentiment = build_sac_news_features(
        events_by_symbol, cutoff=cutoff, coverage_status=status
    )
    counts = {symbol: len(events_by_symbol[symbol]) for symbol in symbols}
    audit = SACNewsAudit(
        as_of=cutoff,
        start_exclusive=news_window.start_exclusive,
        end_inclusive=news_window.end_inclusive,
        per_symbol=[
            SACNewsSymbolAudit(
                symbol=symbol,
                sentiment_score=sentiment[symbol],
                article_count=counts[symbol],
                coverage_status=coverage[symbol].status,
            )
            for symbol in symbols
        ],
    )
    return sentiment, counts, audit
