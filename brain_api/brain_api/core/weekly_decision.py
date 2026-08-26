"""Monday 09:00 America/New_York cutoff — an RL decision rule, not news-domain logic."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

DECISION_TIMEZONE = ZoneInfo("America/New_York")
DECISION_HOUR = 9
DECISION_MINUTE = 0


class MondayCutoffError(ValueError):
    """Raised when ``as_of`` is not that ISO week's Monday 09:00 New York."""


def monday_of(week_date: date) -> date:
    """Return the Monday (weekday=0) of the ISO week containing ``week_date``."""
    return week_date - timedelta(days=week_date.weekday())


def monday_decision_cutoff(week_date: date) -> datetime:
    """That ISO week's calendar Monday 09:00 America/New_York (DST-aware)."""
    monday = monday_of(week_date)
    return datetime.combine(
        monday, time(DECISION_HOUR, DECISION_MINUTE), tzinfo=DECISION_TIMEZONE
    )


def previous_monday_decision_cutoff(week_date: date) -> datetime:
    """Monday 09:00 New York of the week before ``week_date``'s Monday."""
    this_monday = monday_of(week_date)
    return monday_decision_cutoff(this_monday - timedelta(days=7))


def monday_window_bounds(week_date: date) -> tuple[datetime, datetime]:
    """Return ``(previous Monday 09:00 NY, this Monday 09:00 NY]`` as a pair."""
    end_inclusive = monday_decision_cutoff(week_date)
    start_exclusive = previous_monday_decision_cutoff(week_date)
    return start_exclusive, end_inclusive


def monday_cutoff_for_actor_friday(friday: date) -> datetime:
    """Monday 09:00 NY after an SAC Friday actor cutoff."""
    return monday_decision_cutoff(friday + timedelta(days=3))


def require_monday_decision_cutoff(as_of: datetime) -> datetime:
    """Return the canonical cutoff, or raise if ``as_of`` is not that instant."""
    if as_of.tzinfo is None:
        raise MondayCutoffError("as_of must be timezone-aware")
    local = as_of.astimezone(DECISION_TIMEZONE)
    expected = monday_decision_cutoff(local.date())
    if local != expected:
        raise MondayCutoffError(
            f"as_of {as_of.isoformat()} is not Monday 09:00 America/New_York "
            f"(expected {expected.isoformat()})"
        )
    return expected


def canonical_monday_windows_contained_in(
    start: datetime, end: datetime
) -> list[tuple[datetime, datetime]]:
    """Fully contained ``(prev Monday 09:00, Monday 09:00]`` windows in ``[start, end]``.

    Partial edge weeks are omitted.
    """
    if start.tzinfo is None or end.tzinfo is None:
        raise MondayCutoffError("start and end must be timezone-aware")
    start_ny = start.astimezone(DECISION_TIMEZONE)
    end_ny = end.astimezone(DECISION_TIMEZONE)
    if start_ny >= end_ny:
        return []

    windows: list[tuple[datetime, datetime]] = []
    cursor = monday_of(start_ny.date()) + timedelta(days=7)
    last_monday = monday_of(end_ny.date())
    while cursor <= last_monday:
        end_inclusive = monday_decision_cutoff(cursor)
        start_exclusive = previous_monday_decision_cutoff(cursor)
        if start_exclusive >= start_ny and end_inclusive <= end_ny:
            windows.append((start_exclusive, end_inclusive))
        cursor += timedelta(days=7)
    return windows


__all__ = [
    "DECISION_HOUR",
    "DECISION_MINUTE",
    "DECISION_TIMEZONE",
    "MondayCutoffError",
    "canonical_monday_windows_contained_in",
    "monday_cutoff_for_actor_friday",
    "monday_decision_cutoff",
    "monday_of",
    "monday_window_bounds",
    "previous_monday_decision_cutoff",
    "require_monday_decision_cutoff",
]
