"""Shared inference utilities for ML models.

This module contains common functions used by multiple model inference pipelines.
"""

from dataclasses import dataclass
from datetime import date, timedelta

import exchange_calendars as xcals
import pandas as pd

# exchange_calendars has no XNSE; XBOM (Bombay) is the India equity calendar
# used for NSE/BSE holiday alignment in this repo.
DEFAULT_US_EXCHANGE = "XNYS"
DEFAULT_INDIA_EXCHANGE = "XBOM"


@dataclass
class WeekBoundaries:
    """Trading week boundaries for inference.

    Represents the target week for prediction, computed with holiday awareness.
    """

    target_week_start: date  # First trading day of the week (Mon or later if holiday)
    target_week_end: date  # Last trading day of the week (Fri or earlier if holiday)
    calendar_monday: date  # Calendar Monday of the ISO week
    calendar_friday: date  # Calendar Friday of the ISO week


def _sessions_in_range(calendar, monday_ts: pd.Timestamp, friday_ts: pd.Timestamp):
    """Return sessions in range, or empty if outside the calendar's known span.

    Some exchange_calendars (notably XBOM) have a finite last_session; calling
    sessions_in_range past that raises DateOutOfBounds. Fall back to empty so
    callers use calendar Mon-Fri dates rather than crashing inference.
    """
    try:
        return calendar.sessions_in_range(monday_ts, friday_ts)
    except Exception as exc:
        # Loud: holiday-aware boundaries unavailable; do not invent sessions.
        import logging

        logging.getLogger(__name__).warning(
            "exchange calendar %s sessions_in_range(%s, %s) failed (%s); "
            "falling back to calendar Mon-Fri",
            getattr(calendar, "name", type(calendar).__name__),
            monday_ts.date(),
            friday_ts.date(),
            exc,
        )
        return []


def compute_week_boundaries(
    as_of_date: date, exchange: str = DEFAULT_US_EXCHANGE
) -> WeekBoundaries:
    """Compute holiday-aware week boundaries for the week containing as_of_date.

    Uses the given exchange calendar to determine actual trading days.
    The target week is the ISO week that contains as_of_date.

    Args:
        as_of_date: Reference date (typically the Monday when inference runs)
        exchange: exchange_calendars name (default XNYS; India use XBOM)

    Returns:
        WeekBoundaries with actual trading day start/end for the week
    """
    calendar = xcals.get_calendar(exchange)

    days_since_monday = as_of_date.weekday()
    calendar_monday = as_of_date - timedelta(days=days_since_monday)
    calendar_friday = calendar_monday + timedelta(days=4)

    monday_ts = pd.Timestamp(calendar_monday)
    friday_ts = pd.Timestamp(calendar_friday)

    schedule = _sessions_in_range(calendar, monday_ts, friday_ts)

    if len(schedule) == 0:
        return WeekBoundaries(
            target_week_start=calendar_monday,
            target_week_end=calendar_friday,
            calendar_monday=calendar_monday,
            calendar_friday=calendar_friday,
        )

    target_week_start = schedule[0].date()
    target_week_end = schedule[-1].date()

    return WeekBoundaries(
        target_week_start=target_week_start,
        target_week_end=target_week_end,
        calendar_monday=calendar_monday,
        calendar_friday=calendar_friday,
    )


def compute_week_from_cutoff(
    cutoff_friday: date, exchange: str = DEFAULT_US_EXCHANGE
) -> WeekBoundaries:
    """Compute target week boundaries from a Friday cutoff date.

    Given a Friday cutoff date, returns the NEXT week (Mon-Fri after the cutoff).
    Uses the given exchange calendar for holiday awareness.

    Args:
        cutoff_friday: Must be a Friday. Data is available up to this date.
        exchange: exchange_calendars name (default XNYS; India use XBOM)

    Returns:
        WeekBoundaries for the week AFTER cutoff_friday.

    Raises:
        ValueError: If cutoff_friday is not a Friday.
    """
    if cutoff_friday.weekday() != 4:
        raise ValueError(
            f"cutoff_friday must be a Friday, got {cutoff_friday} "
            f"({cutoff_friday.strftime('%A')})"
        )

    calendar_monday = cutoff_friday + timedelta(days=3)
    calendar_friday = cutoff_friday + timedelta(days=7)

    calendar = xcals.get_calendar(exchange)
    monday_ts = pd.Timestamp(calendar_monday)
    friday_ts = pd.Timestamp(calendar_friday)

    schedule = _sessions_in_range(calendar, monday_ts, friday_ts)

    if len(schedule) == 0:
        return WeekBoundaries(
            target_week_start=calendar_monday,
            target_week_end=calendar_friday,
            calendar_monday=calendar_monday,
            calendar_friday=calendar_friday,
        )

    return WeekBoundaries(
        target_week_start=schedule[0].date(),
        target_week_end=schedule[-1].date(),
        calendar_monday=calendar_monday,
        calendar_friday=calendar_friday,
    )


def extract_trading_weeks(df: pd.DataFrame, min_days: int = 3) -> list[pd.DataFrame]:
    """Extract trading weeks from a price DataFrame.

    Groups data by ISO week and filters out weeks with too few trading days.

    Args:
        df: DataFrame with DatetimeIndex containing OHLCV data
        min_days: Minimum trading days required for a valid week

    Returns:
        List of DataFrames, one per valid trading week
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        return []

    df = df.copy()
    df["_year_week"] = df.index.to_period("W")

    weeks = []
    for _, week_df in df.groupby("_year_week"):
        if len(week_df) >= min_days:
            weeks.append(week_df.drop(columns=["_year_week"]))

    return weeks
