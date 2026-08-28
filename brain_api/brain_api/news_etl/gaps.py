"""Detect missing canonical Monday coverage and backfill those windows."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from brain_api.core.weekly_decision import canonical_monday_windows_contained_in
from brain_api.news.models import NewsWindow
from brain_api.news.service import NewsService
from brain_api.news.store import NewsStore, coverage_key
from brain_api.news_etl.backfill import run_backfill


def missing_windows(
    store: NewsStore,
    *,
    symbols: Sequence[str],
    start: datetime,
    end: datetime,
) -> list[tuple[str, NewsWindow]]:
    ordered_symbols = sorted(set(symbols))
    windows = [
        NewsWindow(start_exclusive=start_exclusive, end_inclusive=end_inclusive)
        for start_exclusive, end_inclusive in canonical_monday_windows_contained_in(
            start, end
        )
    ]
    existing = store.coverage_keys(ordered_symbols)
    gaps: list[tuple[str, NewsWindow]] = []
    for window in windows:
        for symbol in ordered_symbols:
            if (
                coverage_key(symbol, window.start_exclusive, window.end_inclusive)
                not in existing
            ):
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
