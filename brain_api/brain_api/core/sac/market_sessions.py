"""Exact XNYS session ranges for SAC market evidence."""

from datetime import date, timedelta

import exchange_calendars as xcals
import pandas as pd


def xnys_session_dates(start_date: date, end_date: date) -> list[date]:
    """Return every XNYS session in an inclusive calendar-date range."""
    if start_date > end_date:
        return []
    calendar = xcals.get_calendar("XNYS")
    sessions = calendar.sessions_in_range(
        pd.Timestamp(start_date), pd.Timestamp(end_date)
    )
    return [session.date() for session in sessions]


def completed_xnys_session_dates(start_date: date, decision_date: date) -> list[date]:
    """Return evidence sessions completed before the pre-open decision date.

    SAC decisions are scheduled for Monday 09:00 America/New_York,
    before XNYS opens. The decision-date session is therefore never complete.
    """
    return xnys_session_dates(start_date, decision_date - timedelta(days=1))


def require_exact_session_dates(
    actual_dates: list[date], expected_dates: list[date], *, context: str
) -> None:
    """Reject missing, extra, duplicated, unordered, or early-ending evidence."""
    if actual_dates != expected_dates:
        missing = [
            value.isoformat() for value in expected_dates if value not in actual_dates
        ]
        extra = [
            value.isoformat() for value in actual_dates if value not in expected_dates
        ]
        raise ValueError(
            f"{context} must contain the exact completed XNYS session range; "
            f"missing={missing}, extra={extra}"
        )
