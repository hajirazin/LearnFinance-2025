"""Tests for universe endpoints."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from brain_api.main import app

client = TestClient(app)

MOCK_HALAL_UNIVERSE = {
    "stocks": [
        {
            "symbol": "AAPL",
            "name": "Apple Inc",
            "max_weight": 10.0,
            "sources": ["SPUS", "HLAL"],
        },
        {
            "symbol": "MSFT",
            "name": "Microsoft Corp",
            "max_weight": 9.5,
            "sources": ["SPUS"],
        },
        {
            "symbol": "GOOGL",
            "name": "Alphabet Inc",
            "max_weight": 8.0,
            "sources": ["HLAL"],
        },
        {
            "symbol": "AMZN",
            "name": "Amazon.com Inc",
            "max_weight": 7.2,
            "sources": ["SPTE"],
        },
        {
            "symbol": "NVDA",
            "name": "NVIDIA Corp",
            "max_weight": 6.5,
            "sources": ["SPUS", "SPTE"],
        },
    ],
    "etfs_used": ["SPUS", "HLAL", "SPTE"],
    "total_stocks": 5,
    "fetched_at": "2026-01-01T00:00:00+00:00",
}


def _patch_halal():
    return patch(
        "brain_api.routes.universe.get_halal_universe", return_value=MOCK_HALAL_UNIVERSE
    )


def test_get_halal_stocks_returns_expected_structure():
    """Test that /universe/halal returns the expected response structure."""
    with _patch_halal():
        response = client.get("/universe/halal")

    assert response.status_code == 200
    data = response.json()

    assert "stocks" in data
    assert "etfs_used" in data
    assert "total_stocks" in data
    assert "fetched_at" in data

    assert set(data["etfs_used"]) == {"SPUS", "HLAL", "SPTE"}


def test_get_halal_stocks_returns_stocks():
    """Test that /universe/halal returns at least some stocks."""
    with _patch_halal():
        response = client.get("/universe/halal")

    assert response.status_code == 200
    data = response.json()

    assert data["total_stocks"] > 0
    assert len(data["stocks"]) > 0
    assert len(data["stocks"]) == data["total_stocks"]


def test_get_halal_stocks_no_duplicates():
    """Test that /universe/halal returns unique symbols only."""
    with _patch_halal():
        response = client.get("/universe/halal")

    assert response.status_code == 200
    data = response.json()

    symbols = [stock["symbol"] for stock in data["stocks"]]
    assert len(symbols) == len(set(symbols)), "Duplicate symbols found"


def test_get_halal_stocks_stock_structure():
    """Test that each stock has required fields."""
    with _patch_halal():
        response = client.get("/universe/halal")

    assert response.status_code == 200
    data = response.json()

    for stock in data["stocks"]:
        assert "symbol" in stock
        assert "name" in stock
        assert "max_weight" in stock
        assert "sources" in stock

        assert isinstance(stock["symbol"], str)
        assert isinstance(stock["name"], str)
        assert isinstance(stock["max_weight"], int | float)
        assert isinstance(stock["sources"], list)
        assert len(stock["sources"]) > 0


def test_get_halal_stocks_sorted_by_weight():
    """Test that stocks are sorted by max_weight descending."""
    with _patch_halal():
        response = client.get("/universe/halal")

    assert response.status_code == 200
    data = response.json()

    weights = [stock["max_weight"] for stock in data["stocks"]]
    assert weights == sorted(weights, reverse=True), "Stocks not sorted by weight"


# ============================================================================
# S&P 500 Universe Tests
# ============================================================================


def test_get_sp500_universe_returns_expected_structure():
    """Test that get_sp500_universe returns the expected structure."""
    from brain_api.universe.sp500 import get_sp500_universe

    data = get_sp500_universe()

    # Check required fields exist
    assert "stocks" in data
    assert "source" in data
    assert "total_stocks" in data
    assert "fetched_at" in data

    # Check source is datahub.io
    assert data["source"] == "datahub.io"


def test_get_sp500_universe_returns_stocks():
    """Test that get_sp500_universe returns stocks."""
    from brain_api.universe.sp500 import get_sp500_universe

    data = get_sp500_universe()

    # Should have ~500 stocks
    assert data["total_stocks"] > 400
    assert len(data["stocks"]) > 400
    assert len(data["stocks"]) == data["total_stocks"]


def test_get_sp500_universe_stock_structure():
    """Test that each S&P 500 stock has required fields."""
    from brain_api.universe.sp500 import get_sp500_universe

    data = get_sp500_universe()

    for stock in data["stocks"][:10]:  # Check first 10 stocks
        assert "symbol" in stock
        assert "name" in stock
        assert "sector" in stock

        # Validate types
        assert isinstance(stock["symbol"], str)
        assert isinstance(stock["name"], str)
        assert isinstance(stock["sector"], str)
        assert len(stock["symbol"]) > 0


def test_get_sp500_symbols_returns_list():
    """Test that get_sp500_symbols returns a list of symbols."""
    from brain_api.universe.sp500 import get_sp500_symbols

    symbols = get_sp500_symbols()

    assert isinstance(symbols, list)
    assert len(symbols) > 400
    assert all(isinstance(s, str) for s in symbols)


def test_get_sp500_universe_contains_known_stocks():
    """Test that S&P 500 contains well-known stocks."""
    from brain_api.universe.sp500 import get_sp500_symbols

    symbols = get_sp500_symbols()

    # These are long-standing S&P 500 members
    known_stocks = ["AAPL", "MSFT", "AMZN", "GOOGL", "JPM"]
    for stock in known_stocks:
        assert stock in symbols, f"Expected {stock} in S&P 500"


# ============================================================================
# Halal_New Universe Tests
# ============================================================================


def test_get_halal_new_stocks_returns_expected_structure():
    """Test that /universe/halal_new returns the expected response structure."""
    response = client.get("/universe/halal_new")

    assert response.status_code == 200
    data = response.json()

    assert "stocks" in data
    assert "etfs_used" in data
    assert "total_stocks" in data
    assert "fetched_at" in data


def test_get_halal_new_stocks_returns_stocks():
    """Test that /universe/halal_new returns at least some stocks."""
    response = client.get("/universe/halal_new")

    assert response.status_code == 200
    data = response.json()

    assert data["total_stocks"] > 0
    assert len(data["stocks"]) > 0
    assert len(data["stocks"]) == data["total_stocks"]


def test_get_halal_new_stocks_no_duplicates():
    """Test that /universe/halal_new returns unique symbols only."""
    response = client.get("/universe/halal_new")

    assert response.status_code == 200
    data = response.json()

    symbols = [stock["symbol"] for stock in data["stocks"]]
    assert len(symbols) == len(set(symbols)), "Duplicate symbols found"


def test_get_halal_new_stocks_stock_structure():
    """Test that each Halal_New stock has required fields."""
    response = client.get("/universe/halal_new")

    assert response.status_code == 200
    data = response.json()

    for stock in data["stocks"][:10]:  # Check first 10 stocks
        assert "symbol" in stock
        assert "name" in stock
        assert "max_weight" in stock
        assert "sources" in stock

        assert isinstance(stock["symbol"], str)
        assert isinstance(stock["name"], str)
        assert isinstance(stock["max_weight"], int | float)
        assert isinstance(stock["sources"], list)
        assert len(stock["sources"]) > 0


def test_get_halal_new_stocks_sorted_by_weight():
    """Test that Halal_New stocks are sorted by max_weight descending."""
    response = client.get("/universe/halal_new")

    assert response.status_code == 200
    data = response.json()

    weights = [stock["max_weight"] for stock in data["stocks"]]
    assert weights == sorted(weights, reverse=True), "Stocks not sorted by weight"


def test_get_halal_new_etfs_used():
    """Test that /universe/halal_new uses all 5 halal ETFs."""
    response = client.get("/universe/halal_new")

    assert response.status_code == 200
    data = response.json()

    expected_etfs = {"SPUS", "SPTE", "SPWO", "HLAL", "UMMA"}
    assert set(data["etfs_used"]) == expected_etfs


# ============================================================================
# Halal_Filtered Universe Tests
# ============================================================================


def _make_mock_halal_new_universe(count: int = 20) -> dict:
    """Create a mock halal_new universe for testing."""
    stocks = [
        {
            "symbol": f"SYM{i}",
            "name": f"Company {i}",
            "max_weight": 10.0 - i * 0.1,
            "sources": ["SPUS"],
        }
        for i in range(count)
    ]
    return {
        "stocks": stocks,
        "etfs_used": ["SPUS", "SPTE", "SPWO", "HLAL", "UMMA"],
        "total_stocks": count,
        "fetched_at": "2026-01-01T00:00:00+00:00",
    }


def _make_mock_metrics(symbols: list[str]) -> dict[str, dict]:
    """Create mock metrics where all stocks pass the junk filter."""
    metrics = {}
    for i, sym in enumerate(symbols):
        metrics[sym] = {
            "roe": 0.15 + i * 0.01,
            "price": 150.0 + i,
            "sma200": 140.0,
            "beta": 1.0 + i * 0.02,
            "gross_margin": 0.4 + i * 0.005,
            "roic": 0.12 + i * 0.005,
            "earnings_yield": 0.05 + i * 0.002,
            "six_month_return": 0.08 + i * 0.01,
        }
    return metrics


def test_get_halal_filtered_returns_expected_structure():
    """Test that /universe/halal_filtered returns the expected structure."""
    mock_universe = _make_mock_halal_new_universe()
    mock_symbols = [s["symbol"] for s in mock_universe["stocks"]]
    mock_metrics = _make_mock_metrics(mock_symbols)

    with (
        patch(
            "brain_api.universe.halal_filtered.get_halal_new_universe",
            return_value=mock_universe,
        ),
        patch(
            "brain_api.universe.halal_filtered.fetch_stock_metrics",
            return_value=mock_metrics,
        ),
    ):
        response = client.get("/universe/halal_filtered")

    assert response.status_code == 200
    data = response.json()

    assert "stocks" in data
    assert "total_before_filter" in data
    assert "total_after_filter" in data
    assert "total_scored" in data
    assert "top_n" in data
    assert "fetched_at" in data


def test_get_halal_filtered_returns_max_15_stocks():
    """Test that /universe/halal_filtered returns at most 15 stocks."""
    mock_universe = _make_mock_halal_new_universe(count=25)
    mock_symbols = [s["symbol"] for s in mock_universe["stocks"]]
    mock_metrics = _make_mock_metrics(mock_symbols)

    with (
        patch(
            "brain_api.universe.halal_filtered.get_halal_new_universe",
            return_value=mock_universe,
        ),
        patch(
            "brain_api.universe.halal_filtered.fetch_stock_metrics",
            return_value=mock_metrics,
        ),
    ):
        response = client.get("/universe/halal_filtered")

    assert response.status_code == 200
    data = response.json()

    assert len(data["stocks"]) == 15
    assert data["top_n"] == 15


def test_get_halal_filtered_stocks_have_factor_scores():
    """Test that each halal_filtered stock has factor_score and factor_components."""
    mock_universe = _make_mock_halal_new_universe()
    mock_symbols = [s["symbol"] for s in mock_universe["stocks"]]
    mock_metrics = _make_mock_metrics(mock_symbols)

    with (
        patch(
            "brain_api.universe.halal_filtered.get_halal_new_universe",
            return_value=mock_universe,
        ),
        patch(
            "brain_api.universe.halal_filtered.fetch_stock_metrics",
            return_value=mock_metrics,
        ),
    ):
        response = client.get("/universe/halal_filtered")

    assert response.status_code == 200
    data = response.json()

    for stock in data["stocks"]:
        assert "factor_score" in stock
        assert "factor_components" in stock
        assert "metrics" in stock
        assert stock["factor_score"] is not None
