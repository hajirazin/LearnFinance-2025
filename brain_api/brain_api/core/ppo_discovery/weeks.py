"""Weekly decision cutoffs and point-in-time price slices for ppo_discovery."""

from __future__ import annotations

from datetime import UTC, date, datetime, time

import pandas as pd

from brain_api.core.ppo_discovery.schemas import PPODiscoveryError
from brain_api.core.sac.trade_clock import (
    SACWeeklyTradeClock,
    build_sac_weekly_trade_clock,
    extract_session_open_prices,
)
from brain_api.core.weekly_decision import (
    monday_cutoff_for_actor_friday,
    monday_window_bounds,
)
from brain_api.news.models import NEWS_ARCHIVE_START_ISO

DECISION_CLOCK_TIME = time(20, 0)


def weekly_trade_clock(start_date: date, end_date: date) -> SACWeeklyTradeClock:
    """Reuse the SAC XNYS Monday-open / Friday-cutoff clock."""
    return build_sac_weekly_trade_clock(start_date, end_date)


def cutoff_datetime(session_date: date) -> datetime:
    """Friday actor cutoff as an aware UTC datetime after the US cash close."""
    return datetime.combine(session_date, DECISION_CLOCK_TIME, tzinfo=UTC)


def actor_cutoff_datetimes(clock: SACWeeklyTradeClock) -> list[datetime]:
    return [cutoff_datetime(timestamp.date()) for timestamp in clock.actor_cutoffs]


def news_window_starts_at_or_after_archive(cutoff: datetime) -> bool:
    """True when the Monday news window for this Friday cutoff is in coverage."""
    archive = datetime.fromisoformat(NEWS_ARCHIVE_START_ISO)
    monday = monday_cutoff_for_actor_friday(cutoff.date())
    start_exclusive, _end = monday_window_bounds(monday.date())
    return start_exclusive >= archive


def prices_as_of(frame: pd.DataFrame, cutoff: date) -> pd.DataFrame:
    """Rows whose session date is on or before ``cutoff``. No fill."""
    if frame is None or frame.empty:
        raise PPODiscoveryError("price frame is empty")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise PPODiscoveryError("price frame must use a DatetimeIndex")
    index = frame.index.tz_localize(None) if frame.index.tz is not None else frame.index
    normalized = pd.DatetimeIndex(index).normalize()
    mask = normalized.date <= cutoff
    sliced = frame.loc[mask].copy()
    sliced.index = normalized[mask]
    if sliced.empty:
        raise PPODiscoveryError(f"no sessions on or before {cutoff.isoformat()}")
    return sliced


def open_to_open_return(
    frame: pd.DataFrame,
    start_session: pd.Timestamp,
    end_session: pd.Timestamp,
    *,
    symbol: str,
) -> tuple[float, float]:
    """Return ``(start_open, simple_return)`` between two XNYS sessions."""
    opens = extract_session_open_prices(
        frame,
        pd.DatetimeIndex([start_session, end_session]),
        symbol=symbol,
    )
    return float(opens[0]), float(opens[1] / opens[0] - 1.0)


__all__ = [
    "actor_cutoff_datetimes",
    "cutoff_datetime",
    "news_window_starts_at_or_after_archive",
    "open_to_open_return",
    "prices_as_of",
    "weekly_trade_clock",
]
