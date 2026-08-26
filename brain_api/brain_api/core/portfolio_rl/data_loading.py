"""Data loading utilities for portfolio RL training.

Load weekly news scores from the DuckDB news store and align them with
price-momentum features for SAC training.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from brain_api.core.sac.momentum_signals import (
    MomentumSignalError,
    compute_momentum_1w,
    compute_momentum_4w,
    compute_momentum_12_1,
    compute_realized_vol_20d,
)
from brain_api.core.sac.news_adapter import build_sac_news_features
from brain_api.core.weekly_decision import (
    monday_cutoff_for_actor_friday,
    monday_window_bounds,
)
from brain_api.news.errors import NewsCoverageMissing
from brain_api.news.models import NewsWindow
from brain_api.news.store import NewsStore
from brain_api.storage.base import DEFAULT_DATA_PATH

NewsObservationError = NewsCoverageMissing


def _store(store: NewsStore | None) -> NewsStore:
    return store if store is not None else NewsStore(DEFAULT_DATA_PATH)


def news_backfill_bounds(
    weekly_cutoffs: pd.DatetimeIndex,
) -> tuple[str, str]:
    """Inclusive ISO bounds that fully contain Monday windows for actor Fridays."""
    first = monday_cutoff_for_actor_friday(weekly_cutoffs[0].date())
    last = monday_cutoff_for_actor_friday(weekly_cutoffs[-1].date())
    start_exclusive, _ = monday_window_bounds(first.date())
    _, end_inclusive = monday_window_bounds(last.date())
    return start_exclusive.isoformat(), end_inclusive.isoformat()


def require_weekly_news_coverage(
    symbols: list[str],
    weekly_cutoffs: pd.DatetimeIndex,
    *,
    store: NewsStore | None = None,
) -> None:
    """Raise if any (symbol, Monday window) lacks exact coverage."""
    news_store = _store(store)
    for timestamp in weekly_cutoffs:
        cutoff = monday_cutoff_for_actor_friday(timestamp.date())
        start_exclusive, end_inclusive = monday_window_bounds(cutoff.date())
        window = NewsWindow(
            start_exclusive=start_exclusive, end_inclusive=end_inclusive
        )
        news_store.require_coverage(symbols, window)


def load_weekly_news_scores(
    symbols: list[str],
    weekly_cutoffs: pd.DatetimeIndex,
    *,
    store: NewsStore | None = None,
) -> dict[str, np.ndarray]:
    """One adapter scalar per symbol per SAC Friday cutoff (Monday 09:00 window)."""
    news_store = _store(store)
    scores: dict[str, list[float]] = {symbol: [] for symbol in symbols}
    for timestamp in weekly_cutoffs:
        cutoff = monday_cutoff_for_actor_friday(timestamp.date())
        start_exclusive, end_inclusive = monday_window_bounds(cutoff.date())
        window = NewsWindow(
            start_exclusive=start_exclusive, end_inclusive=end_inclusive
        )
        coverage = news_store.require_coverage(symbols, window)
        events = news_store.query_events(symbols, window)
        events_by_symbol: dict[str, list] = {symbol: [] for symbol in symbols}
        for event in events:
            if event.symbol in events_by_symbol:
                events_by_symbol[event.symbol].append(event)
        status = {row.symbol: row.status for row in coverage}
        week_scores = build_sac_news_features(
            events_by_symbol, cutoff=cutoff, coverage_status=status
        )
        for symbol in symbols:
            scores[symbol].append(week_scores[symbol])
    return {
        symbol: np.asarray(values, dtype=float) for symbol, values in scores.items()
    }


def align_signals_to_weekly(
    prices_dict: dict[str, pd.DataFrame],
    symbols: list[str],
    weekly_cutoffs: pd.DatetimeIndex | None = None,
    *,
    store: NewsStore | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Align weekly news scores and price-momentum signals.

    Args:
        prices_dict: Dict of symbol -> OHLCV DataFrame with DatetimeIndex
        symbols: Ordered list of symbols
        weekly_cutoffs: SAC Friday actor cutoffs

    Returns:
        Dict of symbol -> dict of signal_name -> weekly numpy array
    """
    signals: dict[str, dict[str, np.ndarray]] = {}
    news_scores: dict[str, np.ndarray] | None = None

    for symbol in symbols:
        if symbol not in prices_dict:
            continue

        price_df = prices_dict[symbol]
        if price_df is None or len(price_df) == 0:
            continue

        weekly_index = (
            pd.DatetimeIndex(weekly_cutoffs)
            if weekly_cutoffs is not None
            else price_df["close"].resample("W-FRI").last().dropna().index
        )
        if weekly_index.tz is not None:
            weekly_index = weekly_index.tz_localize(None)
        n_weeks = len(weekly_index)

        if n_weeks < 2:
            continue

        if news_scores is None:
            news_scores = load_weekly_news_scores(symbols, weekly_index, store=store)

        symbol_signals: dict[str, np.ndarray] = {
            "news_sentiment": news_scores[symbol],
        }

        close_series = price_df["close"]
        if close_series.index.tz is not None:
            close_series = close_series.tz_localize(None)
        close_dates = close_series.index.values
        close_values = close_series.to_numpy(dtype=float)
        close_positions = (
            np.searchsorted(close_dates, weekly_index.values, side="right") - 1
        )
        if np.any(close_positions < 0):
            raise ValueError(
                f"No daily close available on/before a training week for {symbol}"
            )

        momentum_1w = np.empty(n_weeks)
        momentum_4w = np.empty(n_weeks)
        momentum_12_1 = np.empty(n_weeks)
        realized_vol_20d = np.empty(n_weeks)
        for week_idx, close_position in enumerate(close_positions):
            as_of_index = int(close_position)
            try:
                momentum_1w[week_idx] = compute_momentum_1w(
                    close_values, as_of_index=as_of_index
                )
                momentum_4w[week_idx] = compute_momentum_4w(
                    close_values, as_of_index=as_of_index
                )
                momentum_12_1[week_idx] = compute_momentum_12_1(
                    close_values, as_of_index=as_of_index
                )
                realized_vol_20d[week_idx] = compute_realized_vol_20d(
                    close_values, as_of_index=as_of_index
                )
            except MomentumSignalError:
                momentum_1w[week_idx] = np.nan
                momentum_4w[week_idx] = np.nan
                momentum_12_1[week_idx] = np.nan
                realized_vol_20d[week_idx] = np.nan

        symbol_signals["momentum_1w"] = momentum_1w
        symbol_signals["momentum_4w"] = momentum_4w
        symbol_signals["momentum_12_1"] = momentum_12_1
        symbol_signals["realized_vol_20d"] = realized_vol_20d

        signals[symbol] = symbol_signals

    return signals


def build_rl_training_signals(
    prices_dict: dict[str, pd.DataFrame],
    symbols: list[str],
    start_date: date,
    end_date: date,
    weekly_cutoffs: pd.DatetimeIndex | None = None,
    *,
    store: NewsStore | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Build complete signals dict for RL training."""
    del start_date, end_date
    print(f"[PortfolioRL] Loading historical signals for {len(symbols)} symbols...")
    signals = align_signals_to_weekly(
        prices_dict,
        symbols,
        weekly_cutoffs=weekly_cutoffs,
        store=store,
    )
    print(f"[PortfolioRL] Aligned signals for {len(signals)} symbols")
    return signals
