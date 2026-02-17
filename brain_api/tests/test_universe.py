"""Tests for universe endpoints."""

from fastapi.testclient import TestClient

from brain_api.main import app

client = TestClient(app)


def test_get_halal_stocks_returns_expected_structure():
    """Test that /universe/halal returns the expected response structure."""
    response = client.get("/universe/halal")

    assert response.status_code == 200
    data = response.json()

    # Check required fields exist
    assert "stocks" in data
    assert "etfs_used" in data
    assert "total_stocks" in data
    assert "fetched_at" in data

    # Check etfs_used contains expected ETFs
    assert set(data["etfs_used"]) == {"SPUS", "HLAL", "SPTE"}


def test_get_halal_stocks_returns_stocks():
    """Test that /universe/halal returns at least some stocks."""
    response = client.get("/universe/halal")

    assert response.status_code == 200
    data = response.json()

    # Should have some stocks (union of top holdings from 3 ETFs)
    assert data["total_stocks"] > 0
    assert len(data["stocks"]) > 0
    assert len(data["stocks"]) == data["total_stocks"]


def test_get_halal_stocks_no_duplicates():
    """Test that /universe/halal returns unique symbols only."""
    response = client.get("/universe/halal")

    assert response.status_code == 200
    data = response.json()

    symbols = [stock["symbol"] for stock in data["stocks"]]
    assert len(symbols) == len(set(symbols)), "Duplicate symbols found"


def test_get_halal_stocks_stock_structure():
    """Test that each stock has required fields."""
    response = client.get("/universe/halal")

    assert response.status_code == 200
    data = response.json()

    for stock in data["stocks"]:
        assert "symbol" in stock
        assert "name" in stock
        assert "max_weight" in stock
        assert "sources" in stock

        # Validate types
        assert isinstance(stock["symbol"], str)
        assert isinstance(stock["name"], str)
        assert isinstance(stock["max_weight"], int | float)
        assert isinstance(stock["sources"], list)
        assert len(stock["sources"]) > 0  # At least one ETF source


def test_get_halal_stocks_sorted_by_weight():
    """Test that stocks are sorted by max_weight descending."""
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
