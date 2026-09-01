"""PPO preflight price evidence: session counts, hashes, and XNYS gaps."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any

import pandas as pd

from brain_api.core.ppo_discovery.config import HISTORY_BARS, MIN_ELIGIBLE_ASSETS
from brain_api.core.ppo_discovery.dataset_identity import frame_session_hash
from brain_api.core.prices import load_prices_yfinance
from brain_api.core.sac.market_sessions import (
    align_to_xnys_sessions,
    xnys_session_dates,
)
from brain_api.core.vix_fallback import (
    VixFallbackAudit,
    VixFallbackError,
    apply_cboe_vix_fallback,
)

INDEX_SYMBOLS = ("SPY", "^VIX")


def assess_price_readiness(
    symbols: Sequence[str],
    *,
    end_date: date,
    start_date: date | None = None,
    index_end_date: date | None = None,
    prices: dict[str, pd.DataFrame] | None = None,
    cboe_history: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Return readiness plus exact-session hashes, counts, and VIX provenance."""
    price_start = start_date or date(end_date.year - 7, 1, 1)
    loaded = prices
    if loaded is None:
        loaded = load_prices_yfinance([*symbols, *INDEX_SYMBOLS], price_start, end_date)
    issues: list[str] = []
    index_end = index_end_date or end_date
    vix_audit = VixFallbackAudit()
    try:
        vix_result = apply_cboe_vix_fallback(
            loaded,
            required_dates=xnys_session_dates(price_start, index_end),
            cboe_history=cboe_history,
        )
        loaded = vix_result.prices
        vix_audit = vix_result.audit
    except VixFallbackError as exc:
        issues.append(str(exc))
    exclusions: list[str] = []
    session_hashes: dict[str, str] = {}
    session_counts: dict[str, int] = {}
    for name in INDEX_SYMBOLS:
        issue = _inspect_frame(
            loaded.get(name),
            name,
            session_hashes,
            session_counts,
            require_history=True,
            expected_start=price_start,
            expected_end=index_end,
            align_to_xnys=True,
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
            expected_start=None,
            expected_end=end_date,
        )
        if issue:
            exclusions.append(issue)
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
        "exclusions": exclusions,
        "session_hashes": session_hashes,
        "session_counts": session_counts,
        "eligible_symbol_count": eligible,
        "price_start": price_start.isoformat(),
        "end_date": end_date.isoformat(),
        "index_end_date": index_end.isoformat(),
        "vix_provenance": vix_audit.to_dict(),
    }


def _inspect_frame(
    frame: pd.DataFrame | None,
    symbol: str,
    session_hashes: dict[str, str],
    session_counts: dict[str, int],
    *,
    require_history: bool,
    expected_start: date | None,
    expected_end: date,
    align_to_xnys: bool = False,
) -> str | None:
    if frame is None or frame.empty:
        return f"missing price frame for {symbol}"
    if not isinstance(frame.index, pd.DatetimeIndex):
        return f"{symbol} price frame must use a DatetimeIndex"
    index = frame.index.tz_localize(None) if frame.index.tz is not None else frame.index
    dates = [pd.Timestamp(ts).date() for ts in pd.DatetimeIndex(index).normalize()]
    inspected_frame = frame
    if align_to_xnys:
        calendar_start = expected_start or dates[0]
        calendar_dates = xnys_session_dates(calendar_start, expected_end)
        inspected_frame, dates = align_to_xnys_sessions(frame, calendar_dates)
    required_start = expected_start or dates[0]
    expected = xnys_session_dates(required_start, expected_end)
    session_counts[symbol] = len(dates)
    session_hashes[symbol] = frame_session_hash(inspected_frame)
    if require_history and len(dates) < HISTORY_BARS:
        return f"{symbol} has {len(dates)} sessions; need {HISTORY_BARS}"
    if len(dates) < 2:
        return f"{symbol} has fewer than two sessions"
    if dates != expected:
        missing = [value.isoformat() for value in expected if value not in dates]
        extra = [value.isoformat() for value in dates if value not in expected]
        return (
            f"{symbol} is discontinuous versus XNYS "
            f"(missing={missing[:5]}, extra={extra[:5]})"
        )
    return None


__all__ = ["INDEX_SYMBOLS", "assess_price_readiness"]
