"""Tests for HRP allocation endpoint."""

from datetime import date

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from brain_api.main import app
from brain_api.routes.allocation import get_price_loader

MOCK_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
MOCK_INDIA_SYMBOLS = ["INFY.NS", "TCS.NS", "RELIANCE.NS", "HDFCBANK.NS", "WIPRO.NS"]


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
    dates = pd.date_range(end=end_date, periods=days, freq="B")

    prices = {}
    for i, symbol in enumerate(symbols):
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


@pytest.fixture()
def mock_prices():
    return _create_mock_prices(MOCK_SYMBOLS)


@pytest.fixture()
def hrp_client(mock_prices):
    """TestClient with price loader overridden."""
    app.dependency_overrides[get_price_loader] = lambda: lambda syms, s, e: mock_prices
    yield TestClient(app)
    app.dependency_overrides.clear()


# ============================================================================
# Tests
# ============================================================================


def test_hrp_allocation_returns_expected_structure(hrp_client):
    """Test that /allocation/hrp returns the expected response structure."""
    response = hrp_client.post("/allocation/hrp", json={"symbols": MOCK_SYMBOLS})

    assert response.status_code == 200
    data = response.json()

    assert "percentage_weights" in data
    assert "symbols_used" in data
    assert "symbols_excluded" in data
    assert "lookback_days" in data
    assert "as_of_date" in data


def test_hrp_weights_sum_to_100(hrp_client):
    """Test that HRP percentage weights sum to 100."""
    response = hrp_client.post("/allocation/hrp", json={"symbols": MOCK_SYMBOLS})

    assert response.status_code == 200
    data = response.json()

    weights = data["percentage_weights"]
    total = sum(weights.values())

    assert abs(total - 100.0) < 0.1, f"Weights sum to {total}, expected ~100"


def test_hrp_all_symbols_present_or_excluded(hrp_client):
    """Test that all symbols are either allocated or excluded."""
    response = hrp_client.post("/allocation/hrp", json={"symbols": MOCK_SYMBOLS})

    assert response.status_code == 200
    data = response.json()

    allocated = set(data["percentage_weights"].keys())
    excluded = set(data["symbols_excluded"])

    for symbol in MOCK_SYMBOLS:
        assert symbol in allocated or symbol in excluded, f"{symbol} missing from both"


def test_hrp_all_weights_positive(hrp_client):
    """Test that all HRP weights are positive (no short positions)."""
    response = hrp_client.post("/allocation/hrp", json={"symbols": MOCK_SYMBOLS})

    assert response.status_code == 200
    data = response.json()

    for symbol, weight in data["percentage_weights"].items():
        assert weight > 0, f"{symbol} has non-positive weight: {weight}"


def test_hrp_custom_lookback_days(hrp_client):
    """Test that custom lookback_days parameter is respected."""
    response = hrp_client.post(
        "/allocation/hrp", json={"symbols": MOCK_SYMBOLS, "lookback_days": 126}
    )

    assert response.status_code == 200
    data = response.json()

    assert data["lookback_days"] == 126


def test_hrp_custom_as_of_date(hrp_client):
    """Test that custom as_of_date parameter is respected."""
    test_date = "2025-01-01"
    response = hrp_client.post(
        "/allocation/hrp",
        json={"symbols": MOCK_SYMBOLS, "as_of_date": test_date},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["as_of_date"] == test_date


def test_hrp_excludes_symbols_with_insufficient_data():
    """Test that symbols with insufficient data are excluded."""
    mock_prices = _create_mock_prices(MOCK_SYMBOLS)
    mock_prices["NVDA"] = mock_prices["NVDA"].tail(30)

    app.dependency_overrides[get_price_loader] = lambda: lambda syms, s, e: mock_prices
    client = TestClient(app)
    response = client.post("/allocation/hrp", json={"symbols": MOCK_SYMBOLS})
    app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()

    assert "NVDA" in data["symbols_excluded"]
    assert "NVDA" not in data["percentage_weights"]


def test_hrp_returns_400_when_no_valid_symbols():
    """Test that 400 is returned when no symbols have sufficient data."""
    app.dependency_overrides[get_price_loader] = lambda: lambda syms, s, e: {}
    client = TestClient(app)
    response = client.post("/allocation/hrp", json={"symbols": MOCK_SYMBOLS})
    app.dependency_overrides.clear()

    assert response.status_code == 400
    assert "No symbols have sufficient data" in response.json()["detail"]


def test_hrp_lookback_days_validation():
    """Test that lookback_days is validated (min 60, max 756)."""
    client = TestClient(app)

    response = client.post(
        "/allocation/hrp",
        json={"symbols": MOCK_SYMBOLS, "lookback_days": 30},
    )
    assert response.status_code == 422

    response = client.post(
        "/allocation/hrp",
        json={"symbols": MOCK_SYMBOLS, "lookback_days": 800},
    )
    assert response.status_code == 422


def test_hrp_deterministic_with_same_data(hrp_client):
    """Test that HRP produces the same weights for the same input data."""
    response1 = hrp_client.post("/allocation/hrp", json={"symbols": MOCK_SYMBOLS})
    response2 = hrp_client.post("/allocation/hrp", json={"symbols": MOCK_SYMBOLS})

    assert response1.status_code == 200
    assert response2.status_code == 200

    weights1 = response1.json()["percentage_weights"]
    weights2 = response2.json()["percentage_weights"]

    assert weights1 == weights2, "HRP should be deterministic"


def test_hrp_weights_sorted_by_percentage_descending(hrp_client):
    """Test that percentage_weights are sorted from highest to lowest."""
    response = hrp_client.post("/allocation/hrp", json={"symbols": MOCK_SYMBOLS})

    assert response.status_code == 200
    data = response.json()

    weights = list(data["percentage_weights"].values())
    assert weights == sorted(weights, reverse=True), (
        "Weights should be sorted descending"
    )


# ============================================================================
# Symbols parameter validation tests
# ============================================================================


def test_hrp_empty_symbols_returns_422():
    """Test that empty symbols list returns 422."""
    client = TestClient(app)
    response = client.post("/allocation/hrp", json={"symbols": []})
    assert response.status_code == 422


def test_hrp_missing_symbols_returns_422():
    """Test that missing symbols field returns 422."""
    client = TestClient(app)
    response = client.post("/allocation/hrp", json={"lookback_days": 252})
    assert response.status_code == 422


def test_hrp_lookback_756_accepted(hrp_client):
    """Test that lookback_days=756 (3 years) is accepted."""
    mock_prices = _create_mock_prices(MOCK_SYMBOLS, days=800)
    app.dependency_overrides[get_price_loader] = lambda: lambda syms, s, e: mock_prices

    response = hrp_client.post(
        "/allocation/hrp",
        json={"symbols": MOCK_SYMBOLS, "lookback_days": 756},
    )

    assert response.status_code == 200
    assert response.json()["lookback_days"] == 756


def test_hrp_india_ns_suffixed_symbols():
    """Test that .NS suffixed symbols work correctly."""
    mock_prices = _create_mock_prices(MOCK_INDIA_SYMBOLS)
    app.dependency_overrides[get_price_loader] = lambda: lambda syms, s, e: mock_prices
    client = TestClient(app)
    response = client.post("/allocation/hrp", json={"symbols": MOCK_INDIA_SYMBOLS})
    app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()

    for symbol in data["percentage_weights"]:
        assert symbol.endswith(".NS"), f"Expected .NS suffix on {symbol}"
