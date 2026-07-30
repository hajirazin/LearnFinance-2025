"""Deterministic run-date helpers shared by allocation workflows."""

from datetime import datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30), name="IST")


def in_ist(current_time: datetime) -> datetime:
    """Convert a workflow timestamp to India Standard Time."""
    return current_time.astimezone(IST)


def ist_calendar_date(current_time: datetime) -> str:
    """Return the calendar date in India Standard Time as ``YYYY-MM-DD``."""
    return in_ist(current_time).date().isoformat()
