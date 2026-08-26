"""Detect missing canonical Monday coverage and backfill those windows."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from brain_api.core.weekly_decision import canonical_monday_windows_contained_in
from brain_api.news.models import NewsWindow
from brain_api.news.service import NewsService
from brain_api.news.store import NewsStore
from brain_api.news_etl.backfill import run_backfill


def missing_windows(
    store: NewsStore,
    *,
    symbols: Sequence[str],
    start: datetime,
    end: datetime,
) -> list[tuple[str, NewsWindow]]:
    gaps: list[tuple[str, NewsWindow]] = []
    windows = [
        NewsWindow(start_exclusive=start_exclusive, end_inclusive=end_inclusive)
        for start_exclusive, end_inclusive in canonical_monday_windows_contained_in(
            start, end
        )
    ]
    for symbol in sorted(set(symbols)):
        for window in windows:
            if store.get_coverage(symbol, window) is None:
                gaps.append((symbol, window))
    return gaps


def run_gap_fill(
    *,
    symbols: Sequence[str],
    start: datetime,
    end: datetime,
    store: NewsStore,
    service: NewsService | None = None,
) -> str:
    """Same worker as backfill: skip existing coverage, fill the rest."""
    return run_backfill(
        symbols=symbols, start=start, end=end, store=store, service=service
    )
