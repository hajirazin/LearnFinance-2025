"""PPO preflight price evidence: session counts, hashes, and XNYS gaps."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any

import pandas as pd

from brain_api.core.ppo_discovery.config import HISTORY_BARS, MIN_ELIGIBLE_ASSETS
from brain_api.core.ppo_discovery.dataset_identity import frame_session_hash
from brain_api.core.prices import load_prices_yfinance
from brain_api.core.sac.market_sessions import xnys_session_dates

INDEX_SYMBOLS = ("SPY", "^VIX")


def assess_price_readiness(
    symbols: Sequence[str],
    *,
    end_date: date,
    start_date: date | None = None,
    prices: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Return ``ready`` plus per-symbol hashes/counts. Never hits the network
    when ``prices`` is injected (tests). Production callers omit ``prices``.
    """
    price_start = start_date or date(end_date.year - 7, 1, 1)
    loaded = prices
    if loaded is None:
        loaded = load_prices_yfinance([*symbols, *INDEX_SYMBOLS], price_start, end_date)
    issues: list[str] = []
    session_hashes: dict[str, str] = {}
    session_counts: dict[str, int] = {}
    for name in INDEX_SYMBOLS:
        issue = _inspect_frame(
            loaded.get(name),
            name,
            session_hashes,
            session_counts,
            require_history=True,
        )
        if issue:
            issues.append(issue)
    eligible = 0
    for symbol in symbols:
        issue = _inspect_frame(
            loaded.get(symbol),
            symbol,
            session_hashes,
            session_counts,
            require_history=False,
        )
        if issue:
            issues.append(issue)
            continue
        if session_counts.get(symbol, 0) >= HISTORY_BARS:
            eligible += 1
    if eligible < MIN_ELIGIBLE_ASSETS:
        issues.append(
            f"only {eligible} symbols have {HISTORY_BARS} sessions "
            f"(need {MIN_ELIGIBLE_ASSETS})"
        )
    return {
        "ready": not issues,
        "issues": issues,
        "session_hashes": session_hashes,
        "session_counts": session_counts,
        "eligible_symbol_count": eligible,
        "price_start": price_start.isoformat(),
        "end_date": end_date.isoformat(),
    }


def _inspect_frame(
    frame: pd.DataFrame | None,
    symbol: str,
    session_hashes: dict[str, str],
    session_counts: dict[str, int],
    *,
    require_history: bool,
) -> str | None:
    if frame is None or frame.empty:
        return f"missing price frame for {symbol}"
    if not isinstance(frame.index, pd.DatetimeIndex):
        return f"{symbol} price frame must use a DatetimeIndex"
    index = frame.index.tz_localize(None) if frame.index.tz is not None else frame.index
    dates = [pd.Timestamp(ts).date() for ts in pd.DatetimeIndex(index).normalize()]
    session_counts[symbol] = len(dates)
    session_hashes[symbol] = frame_session_hash(frame)
    if require_history and len(dates) < HISTORY_BARS:
        return f"{symbol} has {len(dates)} sessions; need {HISTORY_BARS}"
    if len(dates) < 2:
        return f"{symbol} has fewer than two sessions"
    expected = xnys_session_dates(dates[0], dates[-1])
    if dates != expected:
        missing = [value.isoformat() for value in expected if value not in dates]
        extra = [value.isoformat() for value in dates if value not in expected]
        return (
            f"{symbol} is discontinuous versus XNYS "
            f"(missing={missing[:5]}, extra={extra[:5]})"
        )
    return None


__all__ = ["INDEX_SYMBOLS", "assess_price_readiness"]
