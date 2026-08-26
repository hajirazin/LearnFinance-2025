"""Materialize one complete news partition per weekly actor cutoff."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

from brain_api.core.finbert import SentimentScore
from brain_api.core.news_api.alpaca import AlpacaNewsArticle
from brain_api.core.ppo_discovery.news_evidence import (
    materialize_news_evidence,
    news_window_for_cutoff,
)
from brain_api.core.ppo_discovery.news_store import (
    partition_exists,
    persist_weekly_news_features,
)
from brain_api.core.ppo_discovery.weeks import (
    actor_cutoff_datetimes,
    weekly_trade_clock,
)


def materialize_weekly_news_history(
    symbols: Sequence[str],
    start_date: date,
    end_date: date,
    *,
    force: bool = False,
    base_path: Path | str | None = None,
    fetch_page: Callable[..., tuple[list[AlpacaNewsArticle], str | None]] | None = None,
    score_fn: Callable[[Sequence[str]], list[SentimentScore]] | None = None,
) -> dict[str, Any]:
    """One Alpaca window per Friday cutoff. Existing partitions are skipped."""
    clock = weekly_trade_clock(start_date, end_date)
    cutoffs = actor_cutoff_datetimes(clock)
    previous: datetime | None = None
    written = 0
    skipped = 0
    article_counts: dict[str, int] = {}
    last_path = None
    for cutoff in cutoffs:
        if partition_exists(cutoff, base_path=base_path) and not force:
            skipped += 1
            previous = cutoff
            continue
        features = materialize_news_evidence(
            symbols,
            cutoff,
            previous_cutoff=previous,
            fetch_page=fetch_page,
            score_fn=score_fn,
        )
        window = news_window_for_cutoff(cutoff, previous)
        last_path = persist_weekly_news_features(
            cutoff, features, window=window, base_path=base_path, force=force
        )
        written += 1
        for symbol, row in features.items():
            article_counts[symbol] = article_counts.get(symbol, 0) + row.article_count
        previous = cutoff
    return {
        "cutoffs": len(cutoffs),
        "written": written,
        "skipped": skipped,
        "article_counts": article_counts,
        "parquet_path": str(last_path) if last_path is not None else None,
        "force": force,
    }


__all__ = ["materialize_weekly_news_history"]
