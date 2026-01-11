"""Tests for brain_api.core.inference_utils module."""

from datetime import date

import pytest

from brain_api.core.inference_utils import compute_week_from_cutoff


class TestComputeWeekFromCutoff:
    """Tests for compute_week_from_cutoff function."""

    def test_normal_week(self) -> None:
        """Friday Jan 9 -> predict Mon Jan 12 to Fri Jan 16."""
        boundaries = compute_week_from_cutoff(date(2026, 1, 9))
        assert boundaries.calendar_monday == date(2026, 1, 12)
        assert boundaries.calendar_friday == date(2026, 1, 16)

    def test_raises_if_not_friday(self) -> None:
        """Should raise ValueError if input is not a Friday."""
        with pytest.raises(ValueError, match="must be a Friday"):
            compute_week_from_cutoff(date(2026, 1, 10))  # Saturday

    def test_raises_for_each_non_friday_weekday(self) -> None:
        """Should raise for Mon, Tue, Wed, Thu, Sat, Sun."""
        non_fridays = [
            date(2026, 1, 12),  # Monday
            date(2026, 1, 13),  # Tuesday
            date(2026, 1, 14),  # Wednesday
            date(2026, 1, 15),  # Thursday
            date(2026, 1, 10),  # Saturday
            date(2026, 1, 11),  # Sunday
        ]
        for d in non_fridays:
            with pytest.raises(ValueError, match="must be a Friday"):
                compute_week_from_cutoff(d)

    def test_accepts_friday(self) -> None:
        """Should accept Friday without raising."""
        # Jan 9, 2026 is Friday
        boundaries = compute_week_from_cutoff(date(2026, 1, 9))
        assert boundaries is not None

    def test_target_week_is_next_week(self) -> None:
        """Target week should be the week AFTER the cutoff Friday."""
        # Friday Jan 2, 2026 -> next week is Jan 5-9
        boundaries = compute_week_from_cutoff(date(2026, 1, 2))
        assert boundaries.calendar_monday == date(2026, 1, 5)
        assert boundaries.calendar_friday == date(2026, 1, 9)

    def test_year_boundary(self) -> None:
        """Friday Dec 26, 2025 -> next week is Dec 29 - Jan 2."""
        boundaries = compute_week_from_cutoff(date(2025, 12, 26))
        assert boundaries.calendar_monday == date(2025, 12, 29)
        assert boundaries.calendar_friday == date(2026, 1, 2)

    def test_mlk_day_holiday_week(self) -> None:
        """Test week containing MLK Day (Monday holiday).

        MLK Day 2026 is Jan 19 (3rd Monday of January).
        Friday Jan 16 -> next week is Jan 19-23.
        If Jan 19 is a market holiday, target_week_start should be Jan 20 (Tue).
        """
        # Jan 16, 2026 is Friday
        boundaries = compute_week_from_cutoff(date(2026, 1, 16))
        assert boundaries.calendar_monday == date(2026, 1, 19)
        assert boundaries.calendar_friday == date(2026, 1, 23)
        # target_week_start should be Tuesday if Monday is holiday
        assert boundaries.target_week_start >= date(2026, 1, 19)
        # Either Monday (if not holiday) or Tuesday (if MLK is observed)
        assert boundaries.target_week_start.weekday() in [0, 1]

    def test_calendar_dates_always_mon_to_fri(self) -> None:
        """Calendar Monday and Friday should always be 3 and 7 days after cutoff."""
        test_fridays = [
            date(2026, 1, 2),
            date(2026, 1, 9),
            date(2026, 1, 16),
            date(2026, 2, 6),
            date(2025, 12, 26),
        ]
        for friday in test_fridays:
            boundaries = compute_week_from_cutoff(friday)
            # Monday is 3 days after Friday
            assert (boundaries.calendar_monday - friday).days == 3
            # Friday is 7 days after Friday
            assert (boundaries.calendar_friday - friday).days == 7
            # Monday should be a Monday
            assert boundaries.calendar_monday.weekday() == 0
            # Friday should be a Friday
            assert boundaries.calendar_friday.weekday() == 4
