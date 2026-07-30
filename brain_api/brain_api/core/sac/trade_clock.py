"""Holiday-aware XNYS trade-clock primitives for weekly SAC transitions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import exchange_calendars as xcals
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SACWeeklyTradeClock:
    """Weekly rebalance opens and the preceding Friday actor cutoffs."""

    rebalance_sessions: pd.DatetimeIndex
    actor_cutoffs: pd.DatetimeIndex

    @property
    def transition_count(self) -> int:
        """Number of open-to-next-open reward transitions."""
        return len(self.rebalance_sessions) - 1

    @property
    def transition_actor_cutoffs(self) -> pd.DatetimeIndex:
        """Actor cutoffs aligned with each reward transition."""
        return self.actor_cutoffs[:-1]

    @property
    def transition_start_sessions(self) -> pd.DatetimeIndex:
        """Trade sessions at which each transition begins."""
        return self.rebalance_sessions[:-1]


def build_sac_weekly_trade_clock(
    start_date: date,
    end_date: date,
) -> SACWeeklyTradeClock:
    """Return the first XNYS session of every ISO week in a date window."""
    if start_date > end_date:
        raise ValueError("start_date must not be after end_date")
    calendar = xcals.get_calendar("XNYS")
    # Include the preceding calendar week while identifying weekly first
    # sessions. Otherwise a mid-week window boundary (for example January 1
    # on a Friday) would be mistaken for that ISO week's rebalance session.
    sessions = calendar.sessions_in_range(
        pd.Timestamp(start_date - timedelta(days=7)),
        pd.Timestamp(end_date),
    )
    first_by_week: dict[tuple[int, int], pd.Timestamp] = {}
    for session in sessions:
        session_date = session.date()
        iso = session_date.isocalendar()
        first_by_week.setdefault((iso.year, iso.week), session)
    rebalance_sessions = pd.DatetimeIndex(
        session
        for session in first_by_week.values()
        if start_date <= session.date() <= end_date
    )
    if len(rebalance_sessions) < 2:
        raise ValueError("SAC requires at least two weekly XNYS rebalance sessions")

    actor_cutoffs = pd.DatetimeIndex(
        [
            pd.Timestamp(
                session.date()
                - timedelta(days=session.date().weekday())
                - timedelta(days=3)
            )
            for session in rebalance_sessions
        ]
    )
    return SACWeeklyTradeClock(
        rebalance_sessions=rebalance_sessions,
        actor_cutoffs=actor_cutoffs,
    )


def extract_session_open_prices(
    price_frame: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    *,
    symbol: str,
) -> np.ndarray:
    """Extract a finite positive daily open for every requested XNYS session."""
    if "open" not in price_frame.columns:
        raise ValueError(f"Price data for {symbol} has no open column")
    frame = price_frame.copy()
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"Price data for {symbol} must use a DatetimeIndex")
    index = frame.index
    if index.tz is not None:
        index = index.tz_localize(None)
    frame.index = index.normalize()
    if frame.index.has_duplicates:
        raise ValueError(f"Price data for {symbol} has duplicate session dates")

    requested = pd.DatetimeIndex(sessions)
    if requested.tz is not None:
        requested = requested.tz_localize(None)
    requested = requested.normalize()
    aligned = frame["open"].reindex(requested)
    if aligned.isna().any():
        missing = [
            timestamp.date().isoformat() for timestamp in aligned.index[aligned.isna()]
        ]
        raise ValueError(
            f"Missing XNYS rebalance-session open prices for {symbol}: {missing}"
        )
    values = aligned.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError(
            f"XNYS rebalance-session open prices for {symbol} must be finite and positive"
        )
    return values


def experience_open_transition(
    price_frame: pd.DataFrame,
    week_start: date,
    *,
    symbol: str,
) -> tuple[float, float]:
    """Return start-open price and simple return to next week's first open."""
    calendar_monday = week_start - timedelta(days=week_start.weekday())
    clock = build_sac_weekly_trade_clock(
        calendar_monday,
        calendar_monday + timedelta(days=11),
    )
    opens = extract_session_open_prices(
        price_frame,
        clock.rebalance_sessions[:2],
        symbol=symbol,
    )
    return float(opens[0]), float(opens[1] / opens[0] - 1.0)
