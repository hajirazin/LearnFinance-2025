"""API-level tests for fundamentals signals endpoints.

Tests for:
- POST /signals/fundamentals - Current fundamentals (inference, yfinance)
- POST /signals/fundamentals/historical - Historical fundamentals (training, Alpha Vantage)
"""

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from brain_api.main import app
from brain_api.routes.signals import (
    get_data_base_path,
)

# ============================================================================
# Sample API responses (based on real Alpha Vantage output for historical)
# ============================================================================

SAMPLE_INCOME_STATEMENT = {
    "symbol": "TEST",
    "annualReports": [
        {
            "fiscalDateEnding": "2024-12-31",
            "reportedCurrency": "USD",
            "grossProfit": "35551000000",
            "totalRevenue": "62753000000",
            "operatingIncome": "10074000000",
            "netIncome": "6023000000",
            "interestExpense": "1712000000",
        }
    ],
    "quarterlyReports": [
        {
            "fiscalDateEnding": "2024-09-30",
            "reportedCurrency": "USD",
            "grossProfit": "9591000000",
            "totalRevenue": "16331000000",
            "operatingIncome": "2660000000",
            "netIncome": "1744000000",
            "interestExpense": "492000000",
        },
        {
            "fiscalDateEnding": "2024-06-30",
            "reportedCurrency": "USD",
            "grossProfit": "8500000000",
            "totalRevenue": "15000000000",
            "operatingIncome": "2200000000",
            "netIncome": "1500000000",
            "interestExpense": "400000000",
        },
        {
            "fiscalDateEnding": "2024-03-31",
            "reportedCurrency": "USD",
            "grossProfit": "8000000000",
            "totalRevenue": "14500000000",
            "operatingIncome": "2100000000",
            "netIncome": "1400000000",
            "interestExpense": "380000000",
        },
    ],
}

SAMPLE_BALANCE_SHEET = {
    "symbol": "TEST",
    "annualReports": [
        {
            "fiscalDateEnding": "2024-12-31",
            "reportedCurrency": "USD",
            "totalAssets": "137175000000",
            "totalCurrentAssets": "34482000000",
            "totalCurrentLiabilities": "33142000000",
            "shortLongTermDebtTotal": "58396000000",
            "totalShareholderEquity": "27307000000",
        }
    ],
    "quarterlyReports": [
        {
            "fiscalDateEnding": "2024-09-30",
            "reportedCurrency": "USD",
            "totalAssets": "146312000000",
            "totalCurrentAssets": "32740000000",
            "totalCurrentLiabilities": "35142000000",
            "shortLongTermDebtTotal": "66569000000",
            "totalShareholderEquity": "27905000000",
        },
        {
            "fiscalDateEnding": "2024-06-30",
            "reportedCurrency": "USD",
            "totalAssets": "140000000000",
            "totalCurrentAssets": "31000000000",
            "totalCurrentLiabilities": "32000000000",
            "shortLongTermDebtTotal": "60000000000",
            "totalShareholderEquity": "28000000000",
        },
    ],
}

# Sample yfinance info response (real fields from yfinance)
SAMPLE_YFINANCE_INFO = {
    "grossMargins": 0.45,
    "operatingMargins": 0.30,
    "profitMargins": 0.25,
    "currentRatio": 1.5,
    "debtToEquity": 150,  # yfinance returns as percentage (150 = 1.5x)
    "ebitda": 100000000,
    # Note: interestExpense NOT available in yfinance ticker.info
}


# ============================================================================
# Mock implementations for historical endpoint (Alpha Vantage)
# ============================================================================


class MockAlphaVantageClient:
    """Mock Alpha Vantage client that returns sample data."""

    def __init__(self, data_by_symbol: dict[str, dict[str, Any]] | None = None):
        self.data_by_symbol = data_by_symbol or {}
        self.call_count = 0

    def fetch_income_statement(self, symbol: str) -> dict[str, Any] | None:
        self.call_count += 1
        key = f"{symbol}:income_statement"
        if key in self.data_by_symbol:
            return self.data_by_symbol[key]
        data = SAMPLE_INCOME_STATEMENT.copy()
        data["symbol"] = symbol
        return data

    def fetch_balance_sheet(self, symbol: str) -> dict[str, Any] | None:
        self.call_count += 1
        key = f"{symbol}:balance_sheet"
        if key in self.data_by_symbol:
            return self.data_by_symbol[key]
        data = SAMPLE_BALANCE_SHEET.copy()
        data["symbol"] = symbol
        return data


class MockFundamentalsFetcher:
    """Mock FundamentalsFetcher for testing historical endpoint."""

    def __init__(
        self,
        base_path: Path,
        api_key: str = "test_key",
        return_error_for: set[str] | None = None,
    ):
        from brain_api.core.fundamentals import FundamentalsIndex

        self.base_path = base_path
        self.api_key = api_key
        self.return_error_for = return_error_for or set()

        self.index = FundamentalsIndex(base_path / "cache")
        self.client = MockAlphaVantageClient()
        self.daily_limit = 25

    def fetch_symbol(self, symbol: str):
        """Fetch symbol data (no force_refresh - use PUT endpoint for refresh)."""
        from brain_api.core.fundamentals import (
            FundamentalsResult,
            parse_quarterly_statements,
            save_raw_response,
        )

        if symbol in self.return_error_for:
            raise ValueError(f"API error for {symbol}")

        income_record = self.index.get_fetch_record(symbol, "income_statement")
        balance_record = self.index.get_fetch_record(symbol, "balance_sheet")

        from_cache = income_record is not None and balance_record is not None
        api_calls = 0

        if income_record is None:
            income_data = self.client.fetch_income_statement(symbol)
            if income_data:
                file_path = save_raw_response(
                    self.base_path, symbol, "income_statement", income_data
                )
                quarterly = income_data.get("quarterlyReports", [])
                annual = income_data.get("annualReports", [])
                latest_q = quarterly[0].get("fiscalDateEnding") if quarterly else None
                latest_a = annual[0].get("fiscalDateEnding") if annual else None
                self.index.record_fetch(
                    symbol, "income_statement", str(file_path), latest_a, latest_q
                )
                api_calls += 1
                from_cache = False

        if balance_record is None:
            balance_data = self.client.fetch_balance_sheet(symbol)
            if balance_data:
                file_path = save_raw_response(
                    self.base_path, symbol, "balance_sheet", balance_data
                )
                quarterly = balance_data.get("quarterlyReports", [])
                annual = balance_data.get("annualReports", [])
                latest_q = quarterly[0].get("fiscalDateEnding") if quarterly else None
                latest_a = annual[0].get("fiscalDateEnding") if annual else None
                self.index.record_fetch(
                    symbol, "balance_sheet", str(file_path), latest_a, latest_q
                )
                api_calls += 1
                from_cache = False

        from brain_api.core.fundamentals import load_raw_response

        income_data = load_raw_response(self.base_path, symbol, "income_statement")
        balance_data = load_raw_response(self.base_path, symbol, "balance_sheet")

        income_statements = []
        balance_sheets = []

        if income_data:
            income_statements = parse_quarterly_statements(
                symbol, "income_statement", income_data
            )
        if balance_data:
            balance_sheets = parse_quarterly_statements(
                symbol, "balance_sheet", balance_data
            )

        return FundamentalsResult(
            symbol=symbol,
            income_statements=income_statements,
            balance_sheets=balance_sheets,
            from_cache=from_cache,
            api_calls_made=api_calls,
            api_calls_remaining=self.daily_limit - self.index.get_api_calls_today(),
        )

    def get_api_status(self) -> dict[str, Any]:
        calls_today = self.index.get_api_calls_today()
        return {
            "calls_today": calls_today,
            "daily_limit": self.daily_limit,
            "remaining": max(0, self.daily_limit - calls_today),
        }

    def close(self) -> None:
        self.index.close()


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_data_path():
    """Create a temporary data directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_yfinance_ticker():
    """Mock yfinance.Ticker for current fundamentals tests."""
    with patch("yfinance.Ticker") as mock_ticker_class:
        mock_ticker = MagicMock()
        mock_ticker.info = SAMPLE_YFINANCE_INFO.copy()
        mock_ticker_class.return_value = mock_ticker
        yield mock_ticker_class


@pytest.fixture
def client_with_yfinance_mock(mock_yfinance_ticker):
    """Create test client with mocked yfinance."""
    client = TestClient(app)
    yield client


def _write_cache_file(base_path: Path, symbol: str, endpoint: str, data: dict) -> None:
    """Helper to write a cached JSON file in the expected structure."""
    import json

    cache_dir = base_path / "raw" / "fundamentals" / symbol
    cache_dir.mkdir(parents=True, exist_ok=True)
    file_path = cache_dir / f"{endpoint}.json"
    with open(file_path, "w") as f:
        json.dump(data, f)


@pytest.fixture
def client_with_historical_mock(temp_data_path):
    """Create test client with pre-populated cache for historical endpoint.

    The historical endpoint now reads from cache only (no API calls).
    """
    # Pre-populate cache with sample data for common test symbols
    for symbol in ["AAPL", "MSFT", "GOOGL"]:
        income_data = SAMPLE_INCOME_STATEMENT.copy()
        income_data["symbol"] = symbol
        balance_data = SAMPLE_BALANCE_SHEET.copy()
        balance_data["symbol"] = symbol
        _write_cache_file(
            temp_data_path, symbol, "income_statement", {"response": income_data}
        )
        _write_cache_file(
            temp_data_path, symbol, "balance_sheet", {"response": balance_data}
        )

    app.dependency_overrides[get_data_base_path] = lambda: temp_data_path

    client = TestClient(app)
    yield client

    app.dependency_overrides.clear()


# ============================================================================
# Current Fundamentals (yfinance) - Basic tests
# ============================================================================


def test_fundamentals_returns_200(client_with_yfinance_mock):
    """POST /signals/fundamentals with valid symbols returns 200."""
    response = client_with_yfinance_mock.post(
        "/signals/fundamentals",
        json={"symbols": ["AAPL"]},
    )
    assert response.status_code == 200


def test_fundamentals_returns_all_symbols(client_with_yfinance_mock):
    """POST /signals/fundamentals returns data for all requested symbols."""
    response = client_with_yfinance_mock.post(
        "/signals/fundamentals",
        json={"symbols": ["AAPL", "MSFT", "GOOGL"]},
    )
    assert response.status_code == 200

    data = response.json()
    assert "per_symbol" in data
    assert len(data["per_symbol"]) == 3

    symbols_returned = {s["symbol"] for s in data["per_symbol"]}
    assert symbols_returned == {"AAPL", "MSFT", "GOOGL"}


def test_fundamentals_returns_required_fields(client_with_yfinance_mock):
    """POST /signals/fundamentals returns all required response fields."""
    response = client_with_yfinance_mock.post(
        "/signals/fundamentals",
        json={"symbols": ["AAPL"]},
    )
    assert response.status_code == 200

    data = response.json()

    # Top-level fields (no api_status for yfinance)
    assert "as_of_date" in data
    assert "per_symbol" in data

    # Per-symbol fields
    assert len(data["per_symbol"]) == 1
    sym = data["per_symbol"][0]
    assert "symbol" in sym
    assert "ratios" in sym


def test_fundamentals_returns_ratios(client_with_yfinance_mock):
    """POST /signals/fundamentals returns computed ratios."""
    response = client_with_yfinance_mock.post(
        "/signals/fundamentals",
        json={"symbols": ["AAPL"]},
    )
    assert response.status_code == 200

    data = response.json()
    sym = data["per_symbol"][0]
    ratios = sym["ratios"]

    assert ratios is not None
    assert "gross_margin" in ratios
    assert "operating_margin" in ratios
    assert "net_margin" in ratios
    assert "current_ratio" in ratios
    assert "debt_to_equity" in ratios


def test_fundamentals_ratios_values_reasonable(client_with_yfinance_mock):
    """Computed ratios should be reasonable values."""
    response = client_with_yfinance_mock.post(
        "/signals/fundamentals",
        json={"symbols": ["AAPL"]},
    )
    assert response.status_code == 200

    data = response.json()
    ratios = data["per_symbol"][0]["ratios"]

    # Gross margin should be between 0 and 1
    assert ratios["gross_margin"] is not None
    assert 0 < ratios["gross_margin"] < 1

    # Operating margin should be between 0 and 1
    assert ratios["operating_margin"] is not None
    assert 0 < ratios["operating_margin"] < 1

    # Net margin should be between 0 and 1
    assert ratios["net_margin"] is not None
    assert 0 < ratios["net_margin"] < 1


# ============================================================================
# Current Fundamentals - Validation tests
# ============================================================================


def test_fundamentals_empty_symbols_returns_422(client_with_yfinance_mock):
    """POST /signals/fundamentals with empty symbols list returns 422."""
    response = client_with_yfinance_mock.post(
        "/signals/fundamentals",
        json={"symbols": []},
    )
    assert response.status_code == 422


def test_fundamentals_no_symbols_returns_422(client_with_yfinance_mock):
    """POST /signals/fundamentals without symbols field returns 422."""
    response = client_with_yfinance_mock.post(
        "/signals/fundamentals",
        json={},
    )
    assert response.status_code == 422


def test_fundamentals_max_symbols_exceeded_returns_422(client_with_yfinance_mock):
    """POST /signals/fundamentals with too many symbols returns 422."""
    # MAX_FUNDAMENTALS_SYMBOLS is 20
    symbols = [f"SYM{i}" for i in range(21)]
    response = client_with_yfinance_mock.post(
        "/signals/fundamentals",
        json={"symbols": symbols},
    )
    assert response.status_code == 422


def test_fundamentals_uses_today_as_date(client_with_yfinance_mock):
    """Current endpoint always uses today's date."""
    from datetime import date

    response = client_with_yfinance_mock.post(
        "/signals/fundamentals",
        json={"symbols": ["AAPL"]},
    )
    assert response.status_code == 200

    data = response.json()
    assert data["as_of_date"] == date.today().isoformat()


# ============================================================================
# Historical Fundamentals (Alpha Vantage) - Basic tests
# ============================================================================


def test_historical_fundamentals_returns_200(client_with_historical_mock):
    """POST /signals/fundamentals/historical returns 200."""
    response = client_with_historical_mock.post(
        "/signals/fundamentals/historical",
        json={
            "symbols": ["AAPL"],
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
        },
    )
    assert response.status_code == 200


def test_historical_fundamentals_returns_required_fields(client_with_historical_mock):
    """POST /signals/fundamentals/historical returns all required fields."""
    response = client_with_historical_mock.post(
        "/signals/fundamentals/historical",
        json={
            "symbols": ["AAPL"],
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
        },
    )
    assert response.status_code == 200

    data = response.json()
    assert "start_date" in data
    assert "end_date" in data
    assert "data" in data
    # Note: api_status is NOT included in POST (cache-only) response
    # Use PUT endpoint to refresh and get api_status


def test_historical_fundamentals_returns_flat_list(client_with_historical_mock):
    """POST /signals/fundamentals/historical returns flat list of n x m ratios."""
    response = client_with_historical_mock.post(
        "/signals/fundamentals/historical",
        json={
            "symbols": ["AAPL", "MSFT"],
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
        },
    )
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data["data"], list)
    assert len(data["data"]) > 0

    for entry in data["data"]:
        assert "symbol" in entry
        assert "as_of_date" in entry
        assert "gross_margin" in entry


def test_historical_fundamentals_respects_date_range(client_with_historical_mock):
    """Historical endpoint only returns data within the date range."""
    response = client_with_historical_mock.post(
        "/signals/fundamentals/historical",
        json={
            "symbols": ["AAPL"],
            "start_date": "2024-04-01",
            "end_date": "2024-07-31",
        },
    )
    assert response.status_code == 200

    data = response.json()
    for entry in data["data"]:
        assert "2024-04-01" <= entry["as_of_date"] <= "2024-07-31"


def test_historical_fundamentals_multiple_symbols(client_with_historical_mock):
    """Historical endpoint returns data for multiple symbols."""
    response = client_with_historical_mock.post(
        "/signals/fundamentals/historical",
        json={
            "symbols": ["AAPL", "MSFT", "GOOGL"],
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
        },
    )
    assert response.status_code == 200

    data = response.json()
    symbols_in_response = {entry["symbol"] for entry in data["data"]}
    assert symbols_in_response == {"AAPL", "MSFT", "GOOGL"}


def test_historical_fundamentals_requires_date_range(client_with_historical_mock):
    """Historical endpoint requires start_date and end_date."""
    # Missing end_date
    response = client_with_historical_mock.post(
        "/signals/fundamentals/historical",
        json={"symbols": ["AAPL"], "start_date": "2024-01-01"},
    )
    assert response.status_code == 422

    # Missing start_date
    response = client_with_historical_mock.post(
        "/signals/fundamentals/historical",
        json={"symbols": ["AAPL"], "end_date": "2024-12-31"},
    )
    assert response.status_code == 422


# ============================================================================
# Historical Fundamentals - Cache-only behavior tests
# ============================================================================


def test_historical_returns_empty_for_uncached_symbols(temp_data_path):
    """POST returns empty data list for symbols not in cache."""
    # Only cache AAPL, not NOTCACHED
    _write_cache_file(
        temp_data_path,
        "AAPL",
        "income_statement",
        {"response": SAMPLE_INCOME_STATEMENT},
    )
    _write_cache_file(
        temp_data_path, "AAPL", "balance_sheet", {"response": SAMPLE_BALANCE_SHEET}
    )

    app.dependency_overrides[get_data_base_path] = lambda: temp_data_path
    client = TestClient(app)

    try:
        response = client.post(
            "/signals/fundamentals/historical",
            json={
                "symbols": ["NOTCACHED"],
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
            },
        )
        assert response.status_code == 200
        data = response.json()
        # Should return empty data list (symbol not cached)
        assert data["data"] == []
    finally:
        app.dependency_overrides.clear()


def test_historical_reads_from_cache_only(client_with_historical_mock):
    """POST reads from cache and returns consistent results."""
    request_body = {
        "symbols": ["AAPL"],
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
    }

    # Multiple requests should return same data (cache-only, no API calls)
    response1 = client_with_historical_mock.post(
        "/signals/fundamentals/historical", json=request_body
    )
    response2 = client_with_historical_mock.post(
        "/signals/fundamentals/historical", json=request_body
    )

    assert response1.status_code == 200
    assert response2.status_code == 200
    assert response1.json()["data"] == response2.json()["data"]


# ============================================================================
# PUT Fundamentals Refresh - Tests
# ============================================================================


def test_refresh_fundamentals_returns_200(temp_data_path):
    """PUT /signals/fundamentals/historical returns 200."""
    app.dependency_overrides[get_data_base_path] = lambda: temp_data_path
    client = TestClient(app)

    try:
        # Patch where the function is imported (in endpoints module)
        with patch(
            "brain_api.routes.signals.endpoints.refresh_stale_fundamentals"
        ) as mock_refresh:
            from brain_api.core.data_freshness import FundamentalsRefreshResult

            mock_refresh.return_value = FundamentalsRefreshResult(
                refreshed=[],
                skipped=["AAPL"],
                failed=[],
                api_status={"calls_today": 0, "daily_limit": 25, "remaining": 25},
            )

            response = client.put(
                "/signals/fundamentals/historical",
                json={"symbols": ["AAPL"]},
            )
            assert response.status_code == 200
    finally:
        app.dependency_overrides.clear()


def test_refresh_skips_symbols_fetched_today(temp_data_path):
    """PUT endpoint skips symbols that were already fetched today."""
    app.dependency_overrides[get_data_base_path] = lambda: temp_data_path
    client = TestClient(app)

    try:
        # Patch where the function is imported (in endpoints module)
        with patch(
            "brain_api.routes.signals.endpoints.refresh_stale_fundamentals"
        ) as mock_refresh:
            from brain_api.core.data_freshness import FundamentalsRefreshResult

            mock_refresh.return_value = FundamentalsRefreshResult(
                refreshed=[],
                skipped=["AAPL", "MSFT"],  # Both already fetched today
                failed=[],
                api_status={"calls_today": 5, "daily_limit": 25, "remaining": 20},
            )

            response = client.put(
                "/signals/fundamentals/historical",
                json={"symbols": ["AAPL", "MSFT"]},
            )
            assert response.status_code == 200

            data = response.json()
            assert data["refreshed"] == []
            assert set(data["skipped"]) == {"AAPL", "MSFT"}
            assert data["failed"] == []
    finally:
        app.dependency_overrides.clear()


def test_refresh_returns_statistics(temp_data_path):
    """PUT endpoint returns refresh statistics and API status."""
    app.dependency_overrides[get_data_base_path] = lambda: temp_data_path
    client = TestClient(app)

    try:
        # Patch where the function is imported (in endpoints module)
        with patch(
            "brain_api.routes.signals.endpoints.refresh_stale_fundamentals"
        ) as mock_refresh:
            from brain_api.core.data_freshness import FundamentalsRefreshResult

            mock_refresh.return_value = FundamentalsRefreshResult(
                refreshed=["GOOGL"],
                skipped=["AAPL"],
                failed=["MSFT"],
                api_status={"calls_today": 7, "daily_limit": 25, "remaining": 18},
            )

            response = client.put(
                "/signals/fundamentals/historical",
                json={"symbols": ["AAPL", "MSFT", "GOOGL"]},
            )
            assert response.status_code == 200

            data = response.json()
            assert data["refreshed"] == ["GOOGL"]
            assert data["skipped"] == ["AAPL"]
            assert data["failed"] == ["MSFT"]
            assert data["api_status"]["calls_today"] == 7
            assert data["api_status"]["daily_limit"] == 25
            assert data["api_status"]["remaining"] == 18
    finally:
        app.dependency_overrides.clear()
