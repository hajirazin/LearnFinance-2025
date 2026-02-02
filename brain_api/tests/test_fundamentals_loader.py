"""Tests for the shared fundamentals loader.

Tests for load_historical_fundamentals_from_cache() which is used by:
- POST /signals/fundamentals/historical endpoint
- PatchTST training
- PPO training/finetune
- SAC training/finetune
"""

import tempfile
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from brain_api.core.fundamentals.loader import load_historical_fundamentals_from_cache

# ============================================================================
# Sample data for tests
# ============================================================================

SAMPLE_INCOME_STATEMENT = {
    "response": {
        "symbol": "TEST",
        "quarterlyReports": [
            {
                "fiscalDateEnding": "2024-09-30",
                "reportedCurrency": "USD",
                "grossProfit": "9591000000",
                "totalRevenue": "16331000000",
                "operatingIncome": "2660000000",
                "netIncome": "1744000000",
            },
            {
                "fiscalDateEnding": "2024-06-30",
                "reportedCurrency": "USD",
                "grossProfit": "8500000000",
                "totalRevenue": "15000000000",
                "operatingIncome": "2200000000",
                "netIncome": "1500000000",
            },
            {
                "fiscalDateEnding": "2024-03-31",
                "reportedCurrency": "USD",
                "grossProfit": "8000000000",
                "totalRevenue": "14500000000",
                "operatingIncome": "2100000000",
                "netIncome": "1400000000",
            },
        ],
    }
}

SAMPLE_BALANCE_SHEET = {
    "response": {
        "symbol": "TEST",
        "quarterlyReports": [
            {
                "fiscalDateEnding": "2024-09-30",
                "reportedCurrency": "USD",
                "totalCurrentAssets": "32740000000",
                "totalCurrentLiabilities": "35142000000",
                "shortLongTermDebtTotal": "66569000000",
                "totalShareholderEquity": "27905000000",
            },
            {
                "fiscalDateEnding": "2024-06-30",
                "reportedCurrency": "USD",
                "totalCurrentAssets": "31000000000",
                "totalCurrentLiabilities": "32000000000",
                "shortLongTermDebtTotal": "60000000000",
                "totalShareholderEquity": "28000000000",
            },
        ],
    }
}


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_data_path():
    """Create a temporary data directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def _write_cache_file(base_path: Path, symbol: str, endpoint: str, data: dict) -> None:
    """Helper to write a cached JSON file in the expected structure."""
    import json

    cache_dir = base_path / "raw" / "fundamentals" / symbol
    cache_dir.mkdir(parents=True, exist_ok=True)
    file_path = cache_dir / f"{endpoint}.json"
    with open(file_path, "w") as f:
        json.dump(data, f)


# ============================================================================
# Tests
# ============================================================================


class TestLoadHistoricalFundamentalsFromCache:
    """Tests for load_historical_fundamentals_from_cache function."""

    def test_returns_empty_dict_when_no_cache(self, temp_data_path: Path) -> None:
        """Returns empty dict when no cached data exists."""
        result = load_historical_fundamentals_from_cache(
            symbols=["AAPL", "MSFT"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            base_path=temp_data_path,
        )

        assert result == {}

    def test_loads_single_symbol_from_cache(self, temp_data_path: Path) -> None:
        """Loads fundamentals for a single symbol from cache."""
        # Write cache files
        _write_cache_file(
            temp_data_path, "AAPL", "income_statement", SAMPLE_INCOME_STATEMENT
        )
        _write_cache_file(temp_data_path, "AAPL", "balance_sheet", SAMPLE_BALANCE_SHEET)

        result = load_historical_fundamentals_from_cache(
            symbols=["AAPL"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            base_path=temp_data_path,
        )

        assert "AAPL" in result
        assert isinstance(result["AAPL"], pd.DataFrame)
        assert len(result["AAPL"]) > 0

        # Check expected columns
        expected_cols = [
            "gross_margin",
            "operating_margin",
            "net_margin",
            "current_ratio",
            "debt_to_equity",
        ]
        for col in expected_cols:
            assert col in result["AAPL"].columns

    def test_loads_multiple_symbols(self, temp_data_path: Path) -> None:
        """Loads fundamentals for multiple symbols from cache."""
        # Write cache files for two symbols
        for symbol in ["AAPL", "MSFT"]:
            _write_cache_file(
                temp_data_path, symbol, "income_statement", SAMPLE_INCOME_STATEMENT
            )
            _write_cache_file(
                temp_data_path, symbol, "balance_sheet", SAMPLE_BALANCE_SHEET
            )

        result = load_historical_fundamentals_from_cache(
            symbols=["AAPL", "MSFT", "GOOGL"],  # GOOGL not in cache
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            base_path=temp_data_path,
        )

        assert "AAPL" in result
        assert "MSFT" in result
        assert "GOOGL" not in result  # Not in cache

    def test_filters_by_date_range(self, temp_data_path: Path) -> None:
        """Only returns data within the specified date range."""
        _write_cache_file(
            temp_data_path, "AAPL", "income_statement", SAMPLE_INCOME_STATEMENT
        )
        _write_cache_file(temp_data_path, "AAPL", "balance_sheet", SAMPLE_BALANCE_SHEET)

        # Request only Q2 2024 data
        result = load_historical_fundamentals_from_cache(
            symbols=["AAPL"],
            start_date=date(2024, 4, 1),
            end_date=date(2024, 7, 31),
            base_path=temp_data_path,
        )

        assert "AAPL" in result
        df = result["AAPL"]

        # Should only have Q2 (June 30) data
        for idx in df.index:
            assert idx >= pd.Timestamp("2024-04-01")
            assert idx <= pd.Timestamp("2024-07-31")

    def test_skips_symbols_not_in_cache(self, temp_data_path: Path) -> None:
        """Symbols not in cache are silently skipped."""
        # Only cache AAPL
        _write_cache_file(
            temp_data_path, "AAPL", "income_statement", SAMPLE_INCOME_STATEMENT
        )
        _write_cache_file(temp_data_path, "AAPL", "balance_sheet", SAMPLE_BALANCE_SHEET)

        result = load_historical_fundamentals_from_cache(
            symbols=["AAPL", "NOTCACHED"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            base_path=temp_data_path,
        )

        assert "AAPL" in result
        assert "NOTCACHED" not in result
        assert len(result) == 1

    def test_handles_partial_data(self, temp_data_path: Path) -> None:
        """Handles symbols with only income statement or only balance sheet."""
        # Only write income statement
        _write_cache_file(
            temp_data_path, "AAPL", "income_statement", SAMPLE_INCOME_STATEMENT
        )

        result = load_historical_fundamentals_from_cache(
            symbols=["AAPL"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            base_path=temp_data_path,
        )

        # Should still return data (with some ratios as None)
        assert "AAPL" in result
        assert len(result["AAPL"]) > 0

    def test_empty_date_range_returns_empty(self, temp_data_path: Path) -> None:
        """Returns empty dict for date range outside cached data."""
        _write_cache_file(
            temp_data_path, "AAPL", "income_statement", SAMPLE_INCOME_STATEMENT
        )
        _write_cache_file(temp_data_path, "AAPL", "balance_sheet", SAMPLE_BALANCE_SHEET)

        # Request data from 2020 (before our sample data)
        result = load_historical_fundamentals_from_cache(
            symbols=["AAPL"],
            start_date=date(2020, 1, 1),
            end_date=date(2020, 12, 31),
            base_path=temp_data_path,
        )

        # AAPL should not be in result (no data in range)
        assert "AAPL" not in result
