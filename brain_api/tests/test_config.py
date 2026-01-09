"""Tests for brain_api.core.config module."""

import os
from datetime import date
from unittest.mock import patch

from brain_api.core.config import resolve_training_window


class TestResolveTrainingWindow:
    """Tests for resolve_training_window function."""

    def test_default_lookback_anchored_to_year_start(self) -> None:
        """Start date should be Jan 1 of (current_year - 15)."""
        # Clear any env overrides
        with patch.dict(os.environ, {}, clear=True):
            start_date, end_date = resolve_training_window()

        today = date.today()
        expected_start = date(today.year - 15, 1, 1)

        assert end_date == today
        assert start_date == expected_start
        # Verify it's always Jan 1
        assert start_date.month == 1
        assert start_date.day == 1

    def test_custom_lookback_years(self) -> None:
        """Custom lookback years should still anchor to Jan 1."""
        with patch.dict(os.environ, {"LSTM_TRAIN_LOOKBACK_YEARS": "10"}, clear=True):
            start_date, end_date = resolve_training_window()

        today = date.today()
        expected_start = date(today.year - 10, 1, 1)

        assert end_date == today
        assert start_date == expected_start
        assert start_date.month == 1
        assert start_date.day == 1

    def test_custom_end_date(self) -> None:
        """Custom end date should compute start from that year."""
        with patch.dict(
            os.environ, {"LSTM_TRAIN_WINDOW_END_DATE": "2025-06-15"}, clear=True
        ):
            start_date, end_date = resolve_training_window()

        assert end_date == date(2025, 6, 15)
        # Start should be Jan 1 of 2025-15 = 2010
        assert start_date == date(2010, 1, 1)

    def test_both_overrides(self) -> None:
        """Both lookback and end date overrides should work together."""
        with patch.dict(
            os.environ,
            {
                "LSTM_TRAIN_LOOKBACK_YEARS": "5",
                "LSTM_TRAIN_WINDOW_END_DATE": "2024-03-20",
            },
            clear=True,
        ):
            start_date, end_date = resolve_training_window()

        assert end_date == date(2024, 3, 20)
        # Start should be Jan 1 of 2024-5 = 2019
        assert start_date == date(2019, 1, 1)

    def test_consistent_throughout_year(self) -> None:
        """Start date should be same regardless of when in the year we run."""
        # Simulate running on different days in same year
        test_dates = ["2026-01-10", "2026-06-15", "2026-12-31"]

        start_dates = []
        for test_date in test_dates:
            with patch.dict(
                os.environ, {"LSTM_TRAIN_WINDOW_END_DATE": test_date}, clear=True
            ):
                start_date, _ = resolve_training_window()
                start_dates.append(start_date)

        # All should have same start date (Jan 1, 2011 for 15 year lookback)
        assert all(s == date(2011, 1, 1) for s in start_dates)
