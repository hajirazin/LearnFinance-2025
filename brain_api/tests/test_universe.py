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
        assert isinstance(stock["max_weight"], (int, float))
        assert isinstance(stock["sources"], list)
        assert len(stock["sources"]) > 0  # At least one ETF source


def test_get_halal_stocks_sorted_by_weight():
    """Test that stocks are sorted by max_weight descending."""
    response = client.get("/universe/halal")

    assert response.status_code == 200
    data = response.json()

    weights = [stock["max_weight"] for stock in data["stocks"]]
    assert weights == sorted(weights, reverse=True), "Stocks not sorted by weight"

