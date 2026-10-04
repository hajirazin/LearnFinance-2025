"""Price/session integrity for US PatchTST inputs (no imputation)."""

from datetime import date, timedelta

import exchange_calendars as xcals
import pandas as pd


def session_dates(start: date, end: date) -> pd.DatetimeIndex:
    """Verified inclusive XNYS sessions; calendar failures propagate."""
    calendar = xcals.get_calendar("XNYS")
    # Requested boundaries may be holidays; the calendar must extend beyond
    # them rather than starting at its first *trading* session after start.
    if (
        pd.Timestamp(start) < calendar.first_session
        or pd.Timestamp(end) > calendar.last_session
    ):
        calendar = xcals.get_calendar(
            "XNYS",
            start=str(min(start - timedelta(days=7), calendar.first_session.date())),
            end=str(max(end + timedelta(days=7), calendar.last_session.date())),
        )
    sessions = calendar.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
    return sessions.tz_localize(None).normalize()


def normalize_price_dates(frame: pd.DataFrame) -> pd.DataFrame:
    """Preserve exchange-local dates and reject ambiguous daily evidence."""
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("PatchTST prices require a DatetimeIndex")
    result = frame.copy()
    result.index = result.index.tz_localize(None).normalize()
    if result.index.has_duplicates or not result.index.is_monotonic_increasing:
        raise ValueError("PatchTST daily prices must be unique and chronological")
    return result


def completed_context_dates(cutoff: date, context_length: int) -> pd.DatetimeIndex:
    """The context's returns require context_length + 1 completed closes."""
    sessions = session_dates(
        cutoff - timedelta(days=context_length * 3 + 30),
        cutoff - timedelta(days=1),
    )
    if len(sessions) < context_length + 1:
        raise ValueError("Insufficient exchange calendar coverage for PatchTST")
    return sessions[-context_length - 1 :]


def align_us_price_sessions(frame: pd.DataFrame) -> pd.DataFrame:
    """Missing US sessions stay NaN, so adjacent returns/windows are invalid."""
    result = normalize_price_dates(frame)
    if result.empty:
        return result
    return result.reindex(
        session_dates(result.index[0].date(), result.index[-1].date())
    )
