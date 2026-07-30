"""Tests for the shared fundamentals loader.

Tests for load_historical_fundamentals_from_cache() which is used by:
- POST /signals/fundamentals/historical endpoint
- PatchTST training
- SAC training/finetune
"""

import tempfile
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from brain_api.core.fundamentals.loader import (
    FundamentalsCacheError,
    load_historical_fundamentals_from_cache,
)

# ============================================================================
# Sample data for tests
# ============================================================================

SAMPLE_INCOME_STATEMENT = {
    "response": {
        "symbol": "TEST",
        "quarterlyReports": [
            {
                "fiscalDateEnding": "2024-09-30",
                "filingDate": "2024-11-01",
                "accessionNumber": "0001-24-000003",
                "filingForm": "10-Q",
                "filingSource": "https://www.sec.gov/test/q3",
                "reportedCurrency": "USD",
                "grossProfit": "9591000000",
                "totalRevenue": "16331000000",
                "operatingIncome": "2660000000",
                "netIncome": "1744000000",
            },
            {
                "fiscalDateEnding": "2024-06-30",
                "filingDate": "2024-08-01",
                "accessionNumber": "0001-24-000002",
                "filingForm": "10-Q",
                "filingSource": "https://www.sec.gov/test/q2",
                "reportedCurrency": "USD",
                "grossProfit": "8500000000",
                "totalRevenue": "15000000000",
                "operatingIncome": "2200000000",
                "netIncome": "1500000000",
            },
            {
                "fiscalDateEnding": "2024-03-31",
                "filingDate": "2024-05-01",
                "accessionNumber": "0001-24-000001",
                "filingForm": "10-Q",
                "filingSource": "https://www.sec.gov/test/q1",
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
                "filingDate": "2024-11-01",
                "accessionNumber": "0001-24-000003",
                "filingForm": "10-Q",
                "filingSource": "https://www.sec.gov/test/q3",
                "reportedCurrency": "USD",
                "totalCurrentAssets": "32740000000",
                "totalCurrentLiabilities": "35142000000",
                "shortLongTermDebtTotal": "66569000000",
                "totalShareholderEquity": "27905000000",
            },
            {
                "fiscalDateEnding": "2024-06-30",
                "filingDate": "2024-08-01",
                "accessionNumber": "0001-24-000002",
                "filingForm": "10-Q",
                "filingSource": "https://www.sec.gov/test/q2",
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

        # Request only the window containing Q2's public filing date.
        result = load_historical_fundamentals_from_cache(
            symbols=["AAPL"],
            start_date=date(2024, 4, 1),
            end_date=date(2024, 8, 31),
            base_path=temp_data_path,
        )

        assert "AAPL" in result
        df = result["AAPL"]

        # Index is availability date, not the earlier fiscal period end.
        for idx in df.index:
            assert idx >= pd.Timestamp("2024-04-01")
            assert idx <= pd.Timestamp("2024-08-31")

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

    def test_excludes_partial_periods(self, temp_data_path: Path) -> None:
        """A period missing required statement ratios is not SAC-ready."""
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

        assert "AAPL" not in result

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

    def test_corrupt_cache_raises_diagnostic_error(self, temp_data_path: Path) -> None:
        """Malformed cached evidence must not look like absent fundamentals."""
        cache_dir = temp_data_path / "raw" / "fundamentals" / "AAPL"
        cache_dir.mkdir(parents=True)
        (cache_dir / "income_statement.json").write_text("{invalid json")

        with pytest.raises(
            FundamentalsCacheError,
            match="Malformed fundamentals cache for AAPL",
        ):
            load_historical_fundamentals_from_cache(
                symbols=["AAPL"],
                start_date=date(2024, 1, 1),
                end_date=date(2024, 12, 31),
                base_path=temp_data_path,
            )
