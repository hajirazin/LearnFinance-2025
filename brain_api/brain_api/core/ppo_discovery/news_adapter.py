"""PPO news adapter: 4 asset-level news inputs from shared NewsEvent facts."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from brain_api.core.ppo_discovery.schemas import SymbolNewsFeatures
from brain_api.core.weekly_decision import (
    monday_cutoff_for_actor_friday,
    monday_window_bounds,
)
from brain_api.news.aggregation import (
    confidence_recency_weighted_mean,
    population_std,
)
from brain_api.news.models import CONFIDENCE_RECENCY_TAU_HOURS, NewsEvent, NewsWindow
from brain_api.news.store import NewsStore


@dataclass(frozen=True)
class PPOSymbolNewsFeatures:
    raw_sentiment: float
    article_count: int
    log1p_article_count: float
    recency: float
    sentiment_dispersion: float


def build_ppo_news_features(
    events_by_symbol: Mapping[str, Sequence[NewsEvent]],
    *,
    cutoff: datetime,
) -> dict[str, PPOSymbolNewsFeatures]:
    features: dict[str, PPOSymbolNewsFeatures] = {}
    for symbol, events in events_by_symbol.items():
        unique: dict[str, NewsEvent] = {}
        for event in events:
            unique[event.provider_article_id] = event
        rows = list(unique.values())
        count = len(rows)
        if count == 0:
            features[symbol] = PPOSymbolNewsFeatures(
                raw_sentiment=0.0,
                article_count=0,
                log1p_article_count=0.0,
                recency=0.0,
                sentiment_dispersion=0.0,
            )
            continue
        raw = confidence_recency_weighted_mean(
            [event.sentiment_score for event in rows],
            [event.confidence for event in rows],
            [event.created_at for event in rows],
            cutoff,
            tau=CONFIDENCE_RECENCY_TAU_HOURS,
        )
        latest = max(event.created_at for event in rows)
        hours = (cutoff - latest.astimezone(cutoff.tzinfo)).total_seconds() / 3600.0
        recency = math.exp(-hours / CONFIDENCE_RECENCY_TAU_HOURS)
        features[symbol] = PPOSymbolNewsFeatures(
            raw_sentiment=raw,
            article_count=count,
            log1p_article_count=math.log1p(count),
            recency=recency,
            sentiment_dispersion=population_std(
                [event.sentiment_score for event in rows]
            ),
        )
    return features


def features_to_schema(
    symbol: str,
    features: PPOSymbolNewsFeatures,
    events: Sequence[NewsEvent],
    *,
    cutoff: datetime,
) -> SymbolNewsFeatures:
    """Map adapter outputs onto the packed-state news schema."""
    hours = 0.0
    if events:
        latest = max(event.created_at for event in events)
        hours = (cutoff - latest.astimezone(cutoff.tzinfo)).total_seconds() / 3600.0
    confidences = [event.confidence for event in events]
    sources = {event.source for event in events}
    ids = sorted({event.provider_article_id for event in events})
    return SymbolNewsFeatures(
        symbol=symbol,
        raw_sentiment=features.raw_sentiment,
        article_count=features.article_count,
        average_confidence=(
            float(sum(confidences) / len(confidences)) if confidences else 0.0
        ),
        sentiment_dispersion=features.sentiment_dispersion,
        hours_since_latest=hours,
        unique_source_count=len(sources),
        has_news=1 if features.article_count > 0 else 0,
        query_complete=True,
        news_recency=features.recency,
        log1p_article_count=features.log1p_article_count,
        article_ids_sha256=(
            hashlib.sha256("|".join(ids).encode("utf-8")).hexdigest() if ids else ""
        ),
        request_manifest_sha256="",
    )


def load_weekly_ppo_news_features(
    cutoff: datetime,
    symbols: Sequence[str],
    *,
    store: NewsStore | None = None,
) -> dict[str, SymbolNewsFeatures]:
    """DuckDB coverage + the same weekly adapter used at live inference.

    Training cutoffs are Friday actor dates. Live inference uses the
    Monday 09:00 New York window *after* that Friday, so this loader
    maps Friday -> following Monday before querying DuckDB and before
    recency decay.
    """
    news_store = store if store is not None else NewsStore()
    decision_cutoff = monday_cutoff_for_actor_friday(cutoff.date())
    start_exclusive, end_inclusive = monday_window_bounds(decision_cutoff.date())
    window = NewsWindow(start_exclusive=start_exclusive, end_inclusive=end_inclusive)
    coverage = news_store.require_coverage(list(symbols), window)
    events = news_store.query_events(list(symbols), window)
    events_by_symbol: dict[str, list[NewsEvent]] = {symbol: [] for symbol in symbols}
    for event in events:
        if event.symbol in events_by_symbol:
            events_by_symbol[event.symbol].append(event)
    del coverage
    packed = build_ppo_news_features(events_by_symbol, cutoff=decision_cutoff)
    return {
        symbol: features_to_schema(
            symbol,
            packed[symbol],
            events_by_symbol[symbol],
            cutoff=decision_cutoff,
        )
        for symbol in symbols
    }


def news_adapter_revision() -> str:
    """Sha256 of this adapter module. Pinned on ppo_discovery artifacts."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
