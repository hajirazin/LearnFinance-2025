"""Tests for brain_api.core.config module."""

import os
from datetime import date
from unittest.mock import patch

import pytest

from brain_api.core.config import (
    UniverseType,
    get_forecaster_train_universe,
    resolve_cutoff_date,
    resolve_training_window,
)


class TestResolveCutoffDate:
    """Tests for resolve_cutoff_date function (Friday anchoring)."""

    def test_monday_returns_previous_friday(self) -> None:
        """Monday Jan 12, 2026 -> Friday Jan 9, 2026."""
        assert resolve_cutoff_date(date(2026, 1, 12)) == date(2026, 1, 9)

    def test_tuesday_returns_previous_friday(self) -> None:
        """Tuesday Jan 13, 2026 -> Friday Jan 9, 2026."""
        assert resolve_cutoff_date(date(2026, 1, 13)) == date(2026, 1, 9)

    def test_wednesday_returns_previous_friday(self) -> None:
        """Wednesday Jan 14, 2026 -> Friday Jan 9, 2026."""
        assert resolve_cutoff_date(date(2026, 1, 14)) == date(2026, 1, 9)

    def test_thursday_returns_previous_friday(self) -> None:
        """Thursday Jan 15, 2026 -> Friday Jan 9, 2026."""
        assert resolve_cutoff_date(date(2026, 1, 15)) == date(2026, 1, 9)

    def test_friday_returns_previous_friday_not_same_day(self) -> None:
        """Friday Jan 9, 2026 -> Friday Jan 2, 2026 (NOT Jan 9!)."""
        assert resolve_cutoff_date(date(2026, 1, 9)) == date(2026, 1, 2)

    def test_saturday_returns_previous_friday(self) -> None:
        """Saturday Jan 10, 2026 -> Friday Jan 9, 2026."""
        assert resolve_cutoff_date(date(2026, 1, 10)) == date(2026, 1, 9)

    def test_sunday_returns_previous_friday(self) -> None:
        """Sunday Jan 11, 2026 -> Friday Jan 9, 2026."""
        assert resolve_cutoff_date(date(2026, 1, 11)) == date(2026, 1, 9)

    def test_year_boundary_monday(self) -> None:
        """Monday Jan 5, 2026 -> Friday Jan 2, 2026."""
        assert resolve_cutoff_date(date(2026, 1, 5)) == date(2026, 1, 2)

    def test_year_boundary_friday(self) -> None:
        """Friday Jan 2, 2026 -> Friday Dec 26, 2025."""
        assert resolve_cutoff_date(date(2026, 1, 2)) == date(2025, 12, 26)

    def test_result_is_always_friday(self) -> None:
        """Result should always be a Friday (weekday=4)."""
        test_dates = [
            date(2026, 1, 5),
            date(2026, 1, 6),
            date(2026, 1, 7),
            date(2026, 1, 8),
            date(2026, 1, 9),
            date(2026, 1, 10),
            date(2026, 1, 11),
        ]
        for test_date in test_dates:
            result = resolve_cutoff_date(test_date)
            assert result.weekday() == 4, (
                f"Expected Friday, got {result.strftime('%A')} for input {test_date}"
            )

    def test_reads_from_env_when_no_argument(self) -> None:
        """Should read CUTOFF_DATE env var when no argument provided."""
        with patch.dict(os.environ, {"CUTOFF_DATE": "2026-01-14"}, clear=True):
            result = resolve_cutoff_date()
        # Jan 14, 2026 is Wednesday -> should return Jan 9, 2026 (Friday)
        assert result == date(2026, 1, 9)


class TestResolveTrainingWindow:
    """Tests for resolve_training_window function."""

    def test_default_lookback_anchored_to_friday(self) -> None:
        """End date should be Friday-anchored."""
        with patch.dict(os.environ, {}, clear=True):
            start_date, end_date = resolve_training_window()

        # End date should be a Friday
        assert end_date.weekday() == 4, (
            f"Expected Friday, got {end_date.strftime('%A')}"
        )
        # Start date should be Jan 1 of (end_date.year - 15)
        expected_start = date(end_date.year - 15, 1, 1)
        assert start_date == expected_start

    def test_custom_lookback_years_friday_anchored(self) -> None:
        """Custom lookback years with Friday-anchored end date."""
        with patch.dict(os.environ, {"LSTM_TRAIN_LOOKBACK_YEARS": "10"}, clear=True):
            start_date, end_date = resolve_training_window()

        # End date should be a Friday
        assert end_date.weekday() == 4
        expected_start = date(end_date.year - 10, 1, 1)
        assert start_date == expected_start

    def test_custom_end_date_anchors_to_friday(self) -> None:
        """Custom end date (Sunday) should anchor to previous Friday."""
        # June 15, 2025 is a Sunday -> should anchor to June 13, 2025 (Friday)
        with patch.dict(
            os.environ, {"LSTM_TRAIN_WINDOW_END_DATE": "2025-06-15"}, clear=True
        ):
            start_date, end_date = resolve_training_window()

        assert end_date == date(2025, 6, 13)  # Friday before June 15
        # Start should be Jan 1 of 2025-15 = 2010
        assert start_date == date(2010, 1, 1)

    def test_custom_end_date_on_friday_goes_to_previous_friday(self) -> None:
        """Custom end date on Friday should go to PREVIOUS Friday."""
        # Jan 9, 2026 is a Friday -> should anchor to Jan 2, 2026 (prev Friday)
        with patch.dict(
            os.environ, {"LSTM_TRAIN_WINDOW_END_DATE": "2026-01-09"}, clear=True
        ):
            start_date, end_date = resolve_training_window()

        assert end_date == date(2026, 1, 2)  # Previous Friday
        assert start_date == date(2011, 1, 1)

    def test_both_overrides_friday_anchored(self) -> None:
        """Both lookback and end date overrides should work with Friday anchoring."""
        # March 20, 2024 is a Wednesday -> should anchor to March 15, 2024 (Friday)
        with patch.dict(
            os.environ,
            {
                "LSTM_TRAIN_LOOKBACK_YEARS": "5",
                "LSTM_TRAIN_WINDOW_END_DATE": "2024-03-20",
            },
            clear=True,
        ):
            start_date, end_date = resolve_training_window()

        assert end_date == date(2024, 3, 15)  # Friday before March 20
        # Start should be Jan 1 of 2024-5 = 2019
        assert start_date == date(2019, 1, 1)


class TestUniverseType:
    """Tests for UniverseType enum."""

    def test_universe_type_values(self) -> None:
        """UniverseType enum has expected values."""
        assert UniverseType.HALAL.value == "halal"
        assert UniverseType.SP500.value == "sp500"
        assert UniverseType.HALAL_NEW.value == "halal_new"

    def test_universe_type_is_string_compatible(self) -> None:
        """UniverseType can be compared with strings."""
        assert UniverseType.HALAL == "halal"
        assert UniverseType.SP500 == "sp500"
        assert UniverseType.HALAL_NEW == "halal_new"

    def test_universe_type_from_string(self) -> None:
        """UniverseType can be created from string."""
        assert UniverseType("halal") == UniverseType.HALAL
        assert UniverseType("sp500") == UniverseType.SP500
        assert UniverseType("halal_new") == UniverseType.HALAL_NEW

    def test_universe_type_invalid_raises(self) -> None:
        """Invalid UniverseType value raises ValueError."""
        with pytest.raises(ValueError):
            UniverseType("invalid")


class TestGetForecasterTrainUniverse:
    """Tests for get_forecaster_train_universe function."""

    def test_default_returns_halal(self) -> None:
        """Default should be HALAL when no env var set."""
        with patch.dict(os.environ, {}, clear=True):
            result = get_forecaster_train_universe()
        assert result == UniverseType.HALAL

    def test_halal_from_env(self) -> None:
        """FORECASTER_TRAIN_UNIVERSE=halal returns HALAL."""
        with patch.dict(os.environ, {"FORECASTER_TRAIN_UNIVERSE": "halal"}, clear=True):
            result = get_forecaster_train_universe()
        assert result == UniverseType.HALAL

    def test_sp500_from_env(self) -> None:
        """FORECASTER_TRAIN_UNIVERSE=sp500 returns SP500."""
        with patch.dict(os.environ, {"FORECASTER_TRAIN_UNIVERSE": "sp500"}, clear=True):
            result = get_forecaster_train_universe()
        assert result == UniverseType.SP500

    def test_halal_new_from_env(self) -> None:
        """FORECASTER_TRAIN_UNIVERSE=halal_new returns HALAL_NEW."""
        with patch.dict(
            os.environ, {"FORECASTER_TRAIN_UNIVERSE": "halal_new"}, clear=True
        ):
            result = get_forecaster_train_universe()
        assert result == UniverseType.HALAL_NEW

    def test_case_insensitive(self) -> None:
        """Environment variable is case-insensitive."""
        with patch.dict(os.environ, {"FORECASTER_TRAIN_UNIVERSE": "HALAL"}, clear=True):
            result = get_forecaster_train_universe()
        assert result == UniverseType.HALAL

        with patch.dict(os.environ, {"FORECASTER_TRAIN_UNIVERSE": "SP500"}, clear=True):
            result = get_forecaster_train_universe()
        assert result == UniverseType.SP500

        with patch.dict(
            os.environ, {"FORECASTER_TRAIN_UNIVERSE": "HALAL_NEW"}, clear=True
        ):
            result = get_forecaster_train_universe()
        assert result == UniverseType.HALAL_NEW

    def test_invalid_value_raises_error(self) -> None:
        """Invalid FORECASTER_TRAIN_UNIVERSE value raises ValueError."""
        with patch.dict(
            os.environ, {"FORECASTER_TRAIN_UNIVERSE": "invalid"}, clear=True
        ):
            with pytest.raises(ValueError) as exc_info:
                get_forecaster_train_universe()
            assert "invalid" in str(exc_info.value).lower()
            assert "halal" in str(exc_info.value).lower()
            assert "sp500" in str(exc_info.value).lower()
