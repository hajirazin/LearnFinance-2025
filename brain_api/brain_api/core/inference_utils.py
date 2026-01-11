"""Shared inference utilities for ML models.

This module contains common functions used by multiple model inference pipelines.
"""

from dataclasses import dataclass
from datetime import date, timedelta

import exchange_calendars as xcals
import pandas as pd


@dataclass
class WeekBoundaries:
    """Trading week boundaries for inference.

    Represents the target week for prediction, computed with holiday awareness.
    """

    target_week_start: date  # First trading day of the week (Mon or later if holiday)
    target_week_end: date  # Last trading day of the week (Fri or earlier if holiday)
    calendar_monday: date  # Calendar Monday of the ISO week
    calendar_friday: date  # Calendar Friday of the ISO week


def compute_week_boundaries(as_of_date: date) -> WeekBoundaries:
    """Compute holiday-aware week boundaries for the week containing as_of_date.

    Uses the NYSE calendar (XNYS) to determine actual trading days.
    The target week is the ISO week that contains as_of_date.

    Args:
        as_of_date: Reference date (typically the Monday when inference runs)

    Returns:
        WeekBoundaries with actual trading day start/end for the week
    """
    # Get NYSE calendar
    nyse = xcals.get_calendar("XNYS")

    # Find the Monday of the ISO week containing as_of_date
    # weekday(): Monday=0, Tuesday=1, ..., Sunday=6
    days_since_monday = as_of_date.weekday()
    calendar_monday = as_of_date - timedelta(days=days_since_monday)
    calendar_friday = calendar_monday + timedelta(days=4)

    # Convert to pandas Timestamp for exchange_calendars
    monday_ts = pd.Timestamp(calendar_monday)
    friday_ts = pd.Timestamp(calendar_friday)

    # Find trading days in the week
    schedule = nyse.sessions_in_range(monday_ts, friday_ts)

    if len(schedule) == 0:
        # Entire week is holiday - rare but possible (e.g., week between Christmas and New Year)
        # Fall back to calendar dates; inference will note this in quality
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


def compute_week_from_cutoff(cutoff_friday: date) -> WeekBoundaries:
    """Compute target week boundaries from a Friday cutoff date.

    Given a Friday cutoff date, returns the NEXT week (Mon-Fri after the cutoff).
    Uses NYSE calendar for holiday awareness.

    Args:
        cutoff_friday: Must be a Friday. Data is available up to this date.

    Returns:
        WeekBoundaries for the week AFTER cutoff_friday.

    Raises:
        ValueError: If cutoff_friday is not a Friday.

    Example:
        cutoff_friday = Jan 9, 2026 (Friday)
        Returns: target_week_start = Jan 12 (Mon), target_week_end = Jan 16 (Fri)
    """
    if cutoff_friday.weekday() != 4:
        raise ValueError(
            f"cutoff_friday must be a Friday, got {cutoff_friday} "
            f"({cutoff_friday.strftime('%A')})"
        )

    # Next week's Monday is 3 days after Friday
    calendar_monday = cutoff_friday + timedelta(days=3)
    calendar_friday = cutoff_friday + timedelta(days=7)

    # Get NYSE calendar for holiday-aware boundaries
    nyse = xcals.get_calendar("XNYS")
    monday_ts = pd.Timestamp(calendar_monday)
    friday_ts = pd.Timestamp(calendar_friday)

    schedule = nyse.sessions_in_range(monday_ts, friday_ts)

    if len(schedule) == 0:
        # Entire week is holiday - fall back to calendar dates
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
    # Ensure we have a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        return []

    # Group by ISO week (year + week number)
    df = df.copy()
    df["_year_week"] = df.index.to_period("W")

    weeks = []
    for _, week_df in df.groupby("_year_week"):
        if len(week_df) >= min_days:
            weeks.append(week_df.drop(columns=["_year_week"]))

    return weeks
