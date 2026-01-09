"""Tests for HRP allocation endpoint."""

from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from brain_api.main import app

client = TestClient(app)


# ============================================================================
# Mock data helpers
# ============================================================================


def _create_mock_prices(
    symbols: list[str],
    days: int = 300,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Create mock price data for testing.

    Generates realistic-looking price data with different volatilities
    for each symbol to ensure HRP produces varied weights.
    """
    np.random.seed(seed)

    end_date = date.today()
    dates = pd.date_range(end=end_date, periods=days, freq="B")  # Business days

    prices = {}
    for i, symbol in enumerate(symbols):
        # Different volatility per symbol
        volatility = 0.01 + (i * 0.005)
        returns = np.random.normal(0.0005, volatility, days)
        price = 100 * np.exp(np.cumsum(returns))

        df = pd.DataFrame(
            {
                "open": price * (1 + np.random.uniform(-0.01, 0.01, days)),
                "high": price * (1 + np.random.uniform(0, 0.02, days)),
                "low": price * (1 - np.random.uniform(0, 0.02, days)),
                "close": price,
                "volume": np.random.randint(1000000, 10000000, days),
            },
            index=dates,
        )
        prices[symbol] = df

    return prices


def _create_mock_halal_universe() -> dict:
    """Create mock halal universe response."""
    return {
        "stocks": [
            {
                "symbol": "AAPL",
                "name": "Apple Inc",
                "max_weight": 10.0,
                "sources": ["SPUS"],
            },
            {
                "symbol": "MSFT",
                "name": "Microsoft",
                "max_weight": 9.0,
                "sources": ["SPUS"],
            },
            {
                "symbol": "GOOGL",
                "name": "Alphabet",
                "max_weight": 8.0,
                "sources": ["HLAL"],
            },
            {
                "symbol": "AMZN",
                "name": "Amazon",
                "max_weight": 7.0,
                "sources": ["HLAL"],
            },
            {
                "symbol": "NVDA",
                "name": "NVIDIA",
                "max_weight": 6.0,
                "sources": ["SPTE"],
            },
        ],
        "etfs_used": ["SPUS", "HLAL", "SPTE"],
        "total_stocks": 5,
        "fetched_at": "2025-01-01T00:00:00+00:00",
    }


# ============================================================================
# Tests
# ============================================================================


def test_hrp_allocation_returns_expected_structure():
    """Test that /allocation/hrp returns the expected response structure."""
    mock_universe = _create_mock_halal_universe()
    mock_symbols = [s["symbol"] for s in mock_universe["stocks"]]
    mock_prices = _create_mock_prices(mock_symbols)

    with (
        patch(
            "brain_api.routes.allocation.get_halal_universe", return_value=mock_universe
        ),
        patch(
            "brain_api.routes.allocation.load_prices_yfinance", return_value=mock_prices
        ),
    ):
        response = client.post("/allocation/hrp")

    assert response.status_code == 200
    data = response.json()

    # Check required fields exist
    assert "percentage_weights" in data
    assert "symbols_used" in data
    assert "symbols_excluded" in data
    assert "lookback_days" in data
    assert "as_of_date" in data


def test_hrp_weights_sum_to_100():
    """Test that HRP percentage weights sum to 100."""
    mock_universe = _create_mock_halal_universe()
    mock_symbols = [s["symbol"] for s in mock_universe["stocks"]]
    mock_prices = _create_mock_prices(mock_symbols)

    with (
        patch(
            "brain_api.routes.allocation.get_halal_universe", return_value=mock_universe
        ),
        patch(
            "brain_api.routes.allocation.load_prices_yfinance", return_value=mock_prices
        ),
    ):
        response = client.post("/allocation/hrp")

    assert response.status_code == 200
    data = response.json()

    weights = data["percentage_weights"]
    total = sum(weights.values())

    # Should sum to approximately 100
    assert abs(total - 100.0) < 0.1, f"Weights sum to {total}, expected ~100"


def test_hrp_all_symbols_present_or_excluded():
    """Test that all halal symbols are either allocated or excluded."""
    mock_universe = _create_mock_halal_universe()
    mock_symbols = [s["symbol"] for s in mock_universe["stocks"]]
    mock_prices = _create_mock_prices(mock_symbols)

    with (
        patch(
            "brain_api.routes.allocation.get_halal_universe", return_value=mock_universe
        ),
        patch(
            "brain_api.routes.allocation.load_prices_yfinance", return_value=mock_prices
        ),
    ):
        response = client.post("/allocation/hrp")

    assert response.status_code == 200
    data = response.json()

    allocated = set(data["percentage_weights"].keys())
    excluded = set(data["symbols_excluded"])

    # Every symbol should be either allocated or excluded
    for symbol in mock_symbols:
        assert symbol in allocated or symbol in excluded, f"{symbol} missing from both"


def test_hrp_all_weights_positive():
    """Test that all HRP weights are positive (no short positions)."""
    mock_universe = _create_mock_halal_universe()
    mock_symbols = [s["symbol"] for s in mock_universe["stocks"]]
    mock_prices = _create_mock_prices(mock_symbols)

    with (
        patch(
            "brain_api.routes.allocation.get_halal_universe", return_value=mock_universe
        ),
        patch(
            "brain_api.routes.allocation.load_prices_yfinance", return_value=mock_prices
        ),
    ):
        response = client.post("/allocation/hrp")

    assert response.status_code == 200
    data = response.json()

    for symbol, weight in data["percentage_weights"].items():
        assert weight > 0, f"{symbol} has non-positive weight: {weight}"


def test_hrp_custom_lookback_days():
    """Test that custom lookback_days parameter is respected."""
    mock_universe = _create_mock_halal_universe()
    mock_symbols = [s["symbol"] for s in mock_universe["stocks"]]
    mock_prices = _create_mock_prices(mock_symbols)

    with (
        patch(
            "brain_api.routes.allocation.get_halal_universe", return_value=mock_universe
        ),
        patch(
            "brain_api.routes.allocation.load_prices_yfinance", return_value=mock_prices
        ),
    ):
        response = client.post("/allocation/hrp", json={"lookback_days": 126})

    assert response.status_code == 200
    data = response.json()

    assert data["lookback_days"] == 126


def test_hrp_custom_as_of_date():
    """Test that custom as_of_date parameter is respected."""
    mock_universe = _create_mock_halal_universe()
    mock_symbols = [s["symbol"] for s in mock_universe["stocks"]]
    mock_prices = _create_mock_prices(mock_symbols)

    test_date = "2025-01-01"

    with (
        patch(
            "brain_api.routes.allocation.get_halal_universe", return_value=mock_universe
        ),
        patch(
            "brain_api.routes.allocation.load_prices_yfinance", return_value=mock_prices
        ),
    ):
        response = client.post("/allocation/hrp", json={"as_of_date": test_date})

    assert response.status_code == 200
    data = response.json()

    assert data["as_of_date"] == test_date


def test_hrp_excludes_symbols_with_insufficient_data():
    """Test that symbols with insufficient data are excluded."""
    mock_universe = _create_mock_halal_universe()
    mock_symbols = [s["symbol"] for s in mock_universe["stocks"]]

    # Create prices with one symbol having insufficient data
    mock_prices = _create_mock_prices(mock_symbols)
    # Truncate one symbol to only 30 days
    mock_prices["NVDA"] = mock_prices["NVDA"].tail(30)

    with (
        patch(
            "brain_api.routes.allocation.get_halal_universe", return_value=mock_universe
        ),
        patch(
            "brain_api.routes.allocation.load_prices_yfinance", return_value=mock_prices
        ),
    ):
        response = client.post("/allocation/hrp")

    assert response.status_code == 200
    data = response.json()

    # NVDA should be excluded
    assert "NVDA" in data["symbols_excluded"]
    assert "NVDA" not in data["percentage_weights"]


def test_hrp_returns_400_when_no_valid_symbols():
    """Test that 400 is returned when no symbols have sufficient data."""
    mock_universe = _create_mock_halal_universe()

    # Return empty prices for all symbols
    mock_prices = {}

    with (
        patch(
            "brain_api.routes.allocation.get_halal_universe", return_value=mock_universe
        ),
        patch(
            "brain_api.routes.allocation.load_prices_yfinance", return_value=mock_prices
        ),
    ):
        response = client.post("/allocation/hrp")

    assert response.status_code == 400
    assert "No symbols have sufficient data" in response.json()["detail"]


def test_hrp_lookback_days_validation():
    """Test that lookback_days is validated (min 60, max 504)."""
    # Test below minimum
    response = client.post("/allocation/hrp", json={"lookback_days": 30})
    assert response.status_code == 422  # Validation error

    # Test above maximum
    response = client.post("/allocation/hrp", json={"lookback_days": 600})
    assert response.status_code == 422  # Validation error


def test_hrp_deterministic_with_same_data():
    """Test that HRP produces the same weights for the same input data."""
    mock_universe = _create_mock_halal_universe()
    mock_symbols = [s["symbol"] for s in mock_universe["stocks"]]
    mock_prices = _create_mock_prices(mock_symbols, seed=123)

    with (
        patch(
            "brain_api.routes.allocation.get_halal_universe", return_value=mock_universe
        ),
        patch(
            "brain_api.routes.allocation.load_prices_yfinance", return_value=mock_prices
        ),
    ):
        response1 = client.post("/allocation/hrp")
        response2 = client.post("/allocation/hrp")

    assert response1.status_code == 200
    assert response2.status_code == 200

    weights1 = response1.json()["percentage_weights"]
    weights2 = response2.json()["percentage_weights"]

    assert weights1 == weights2, "HRP should be deterministic"


def test_hrp_weights_sorted_by_percentage_descending():
    """Test that percentage_weights are sorted from highest to lowest."""
    mock_universe = _create_mock_halal_universe()
    mock_symbols = [s["symbol"] for s in mock_universe["stocks"]]
    mock_prices = _create_mock_prices(mock_symbols)

    with (
        patch(
            "brain_api.routes.allocation.get_halal_universe", return_value=mock_universe
        ),
        patch(
            "brain_api.routes.allocation.load_prices_yfinance", return_value=mock_prices
        ),
    ):
        response = client.post("/allocation/hrp")

    assert response.status_code == 200
    data = response.json()

    weights = list(data["percentage_weights"].values())
    assert weights == sorted(weights, reverse=True), (
        "Weights should be sorted descending"
    )
