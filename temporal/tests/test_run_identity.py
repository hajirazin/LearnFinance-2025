"""Tests for the IST run-date invariant."""

from datetime import UTC, datetime

from workflows._run_identity import ist_calendar_date


def test_ist_calendar_date_advances_at_ist_midnight():
    just_before_midnight_ist = datetime(2026, 7, 26, 18, 29, 59, tzinfo=UTC)
    midnight_ist = datetime(2026, 7, 26, 18, 30, tzinfo=UTC)

    assert ist_calendar_date(just_before_midnight_ist) == "2026-07-26"
    assert ist_calendar_date(midnight_ist) == "2026-07-27"


def test_ist_calendar_date_does_not_depend_on_process_timezone():
    sunday_utc_monday_ist = datetime(2026, 7, 26, 20, 0, tzinfo=UTC)

    assert ist_calendar_date(sunday_utc_monday_ist) == "2026-07-27"
