"""Unit tests for LSTM domain service (pure functions)."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from brain_api.domain.entities.lstm import LSTMConfig
from brain_api.domain.services.lstm_computation import (
    build_inference_features_from_data,
    classify_direction,
    compute_version,
    compute_week_boundaries_simple,
    compute_weekly_return,
    extract_trading_weeks,
)


class TestComputeVersion:
    """Tests for compute_version."""

    def test_basic_version_format(self):
        """Version should have format v{date}-{hash}."""
        config = LSTMConfig()
        version = compute_version(
            start_date=date(2023, 1, 1),
            end_date=date(2024, 1, 1),
            symbols=["AAPL", "GOOGL"],
            config=config,
        )

        assert version.startswith("v2024-01-01-")
        assert len(version) == 24  # v + YYYY-MM-DD (10) + - + 12-char hash

    def test_deterministic(self):
        """Same inputs should produce same version."""
        config = LSTMConfig()
        version1 = compute_version(
            start_date=date(2023, 1, 1),
            end_date=date(2024, 1, 1),
            symbols=["AAPL", "GOOGL"],
            config=config,
        )
        version2 = compute_version(
            start_date=date(2023, 1, 1),
            end_date=date(2024, 1, 1),
            symbols=["AAPL", "GOOGL"],
            config=config,
        )

        assert version1 == version2

    def test_symbol_order_independent(self):
        """Symbol order should not affect version (sorted internally)."""
        config = LSTMConfig()
        version1 = compute_version(
            start_date=date(2023, 1, 1),
            end_date=date(2024, 1, 1),
            symbols=["AAPL", "GOOGL"],
            config=config,
        )
        version2 = compute_version(
            start_date=date(2023, 1, 1),
            end_date=date(2024, 1, 1),
            symbols=["GOOGL", "AAPL"],
            config=config,
        )

        assert version1 == version2

    def test_different_dates_different_version(self):
        """Different dates should produce different versions."""
        config = LSTMConfig()
        version1 = compute_version(
            start_date=date(2023, 1, 1),
            end_date=date(2024, 1, 1),
            symbols=["AAPL"],
            config=config,
        )
        version2 = compute_version(
            start_date=date(2023, 1, 1),
            end_date=date(2024, 1, 2),
            symbols=["AAPL"],
            config=config,
        )

        assert version1 != version2

    def test_different_config_different_version(self):
        """Different config should produce different versions."""
        config1 = LSTMConfig(hidden_size=64)
        config2 = LSTMConfig(hidden_size=128)

        version1 = compute_version(
            start_date=date(2023, 1, 1),
            end_date=date(2024, 1, 1),
            symbols=["AAPL"],
            config=config1,
        )
        version2 = compute_version(
            start_date=date(2023, 1, 1),
            end_date=date(2024, 1, 1),
            symbols=["AAPL"],
            config=config2,
        )

        assert version1 != version2


class TestComputeWeekBoundariesSimple:
    """Tests for compute_week_boundaries_simple."""

    def test_monday_input(self):
        """Monday should be week start."""
        bounds = compute_week_boundaries_simple(date(2024, 1, 8))  # Monday

        assert bounds.calendar_monday == date(2024, 1, 8)
        assert bounds.calendar_friday == date(2024, 1, 12)
        assert bounds.target_week_start == date(2024, 1, 8)
        assert bounds.target_week_end == date(2024, 1, 12)

    def test_wednesday_input(self):
        """Wednesday should map to correct Monday."""
        bounds = compute_week_boundaries_simple(date(2024, 1, 10))  # Wednesday

        assert bounds.calendar_monday == date(2024, 1, 8)
        assert bounds.calendar_friday == date(2024, 1, 12)

    def test_friday_input(self):
        """Friday should map to same week."""
        bounds = compute_week_boundaries_simple(date(2024, 1, 12))  # Friday

        assert bounds.calendar_monday == date(2024, 1, 8)
        assert bounds.calendar_friday == date(2024, 1, 12)


class TestExtractTradingWeeks:
    """Tests for extract_trading_weeks."""

    def test_empty_dataframe(self):
        """Empty DataFrame should return empty list."""
        df = pd.DataFrame()
        weeks = extract_trading_weeks(df)
        assert weeks == []

    def test_full_week(self):
        """A full week should be extracted."""
        dates = pd.date_range("2024-01-08", "2024-01-12", freq="D")
        df = pd.DataFrame(
            {"open": [100] * 5, "close": [101] * 5},
            index=dates,
        )

        weeks = extract_trading_weeks(df, min_days=3)
        assert len(weeks) == 1
        assert len(weeks[0]) == 5

    def test_short_week_excluded(self):
        """Weeks with fewer than min_days should be excluded."""
        dates = pd.date_range("2024-01-08", "2024-01-09", freq="D")
        df = pd.DataFrame(
            {"open": [100] * 2, "close": [101] * 2},
            index=dates,
        )

        weeks = extract_trading_weeks(df, min_days=3)
        assert len(weeks) == 0

    def test_multiple_weeks(self):
        """Multiple weeks should be extracted separately."""
        dates = pd.date_range("2024-01-08", "2024-01-19", freq="B")  # Business days
        df = pd.DataFrame(
            {"open": [100] * len(dates), "close": [101] * len(dates)},
            index=dates,
        )

        weeks = extract_trading_weeks(df, min_days=3)
        assert len(weeks) == 2


class TestComputeWeeklyReturn:
    """Tests for compute_weekly_return."""

    def test_basic_return(self):
        """Basic weekly return calculation."""
        dates = pd.date_range("2024-01-08", "2024-01-12", freq="D")
        df = pd.DataFrame(
            {
                "open": [100, 101, 102, 103, 104],
                "close": [101, 102, 103, 104, 105],
            },
            index=dates,
        )

        ret = compute_weekly_return(df)
        # Return = (105 - 100) / 100 = 0.05
        assert ret == pytest.approx(0.05)

    def test_negative_return(self):
        """Negative weekly return."""
        dates = pd.date_range("2024-01-08", "2024-01-12", freq="D")
        df = pd.DataFrame(
            {
                "open": [100, 99, 98, 97, 96],
                "close": [99, 98, 97, 96, 95],
            },
            index=dates,
        )

        ret = compute_weekly_return(df)
        # Return = (95 - 100) / 100 = -0.05
        assert ret == pytest.approx(-0.05)

    def test_empty_dataframe(self):
        """Empty DataFrame should return None."""
        df = pd.DataFrame()
        ret = compute_weekly_return(df)
        assert ret is None

    def test_missing_columns(self):
        """Missing required columns should return None."""
        dates = pd.date_range("2024-01-08", "2024-01-12", freq="D")
        df = pd.DataFrame(
            {"high": [100] * 5, "low": [99] * 5},
            index=dates,
        )

        ret = compute_weekly_return(df)
        assert ret is None


class TestBuildInferenceFeaturesFromData:
    """Tests for build_inference_features_from_data."""

    def test_empty_dataframe(self):
        """Empty DataFrame should return insufficient features."""
        df = pd.DataFrame()
        config = LSTMConfig(sequence_length=5)

        features = build_inference_features_from_data("AAPL", df, config, date(2024, 1, 1))

        assert features.symbol == "AAPL"
        assert features.features is None
        assert features.has_enough_history is False

    def test_insufficient_history(self):
        """Insufficient history should return flag."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        df = pd.DataFrame(
            {
                "open": np.random.uniform(100, 110, 10),
                "high": np.random.uniform(110, 120, 10),
                "low": np.random.uniform(90, 100, 10),
                "close": np.random.uniform(100, 110, 10),
                "volume": np.random.uniform(1000000, 2000000, 10),
            },
            index=dates,
        )
        config = LSTMConfig(sequence_length=60)  # Need 60 days, only have 10

        features = build_inference_features_from_data("AAPL", df, config, date(2024, 2, 1))

        assert features.has_enough_history is False
        assert features.features is None

    def test_sufficient_history(self):
        """Sufficient history should produce features."""
        dates = pd.date_range("2024-01-01", periods=70, freq="D")
        df = pd.DataFrame(
            {
                "open": np.random.uniform(100, 110, 70),
                "high": np.random.uniform(110, 120, 70),
                "low": np.random.uniform(90, 100, 70),
                "close": np.random.uniform(100, 110, 70),
                "volume": np.random.uniform(1000000, 2000000, 70),
            },
            index=dates,
        )
        config = LSTMConfig(sequence_length=60)

        features = build_inference_features_from_data("AAPL", df, config, date(2024, 4, 1))

        assert features.has_enough_history is True
        assert features.features is not None
        assert features.features.shape == (60, 5)  # seq_len x n_features

    def test_cutoff_date_filtering(self):
        """Data after cutoff should be excluded."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        df = pd.DataFrame(
            {
                "open": np.random.uniform(100, 110, 100),
                "high": np.random.uniform(110, 120, 100),
                "low": np.random.uniform(90, 100, 100),
                "close": np.random.uniform(100, 110, 100),
                "volume": np.random.uniform(1000000, 2000000, 100),
            },
            index=dates,
        )
        config = LSTMConfig(sequence_length=30)

        # Set cutoff to only include first 40 days
        cutoff = date(2024, 2, 10)
        features = build_inference_features_from_data("AAPL", df, config, cutoff)

        assert features.has_enough_history is True
        assert features.data_end_date < cutoff


class TestClassifyDirection:
    """Tests for classify_direction."""

    def test_up(self):
        """Returns > 0.5% should be UP."""
        assert classify_direction(1.0) == "UP"
        assert classify_direction(0.6) == "UP"

    def test_down(self):
        """Returns < -0.5% should be DOWN."""
        assert classify_direction(-1.0) == "DOWN"
        assert classify_direction(-0.6) == "DOWN"

    def test_flat(self):
        """Returns between -0.5% and 0.5% should be FLAT."""
        assert classify_direction(0.0) == "FLAT"
        assert classify_direction(0.3) == "FLAT"
        assert classify_direction(-0.3) == "FLAT"

    def test_none(self):
        """None should be FLAT."""
        assert classify_direction(None) == "FLAT"

