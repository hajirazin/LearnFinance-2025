"""Tests for universe endpoints."""

from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from brain_api.main import app
from brain_api.universe.stock_filter import YFinanceFetchError

client = TestClient(app)

# ---------------------------------------------------------------------------
# Mock data for S&P 500 (replaces live pd.read_csv from datahub.io)
# ---------------------------------------------------------------------------
MOCK_SP500_DF = pd.DataFrame(
    [
        {"Symbol": "AAPL", "Name": "Apple Inc.", "Sector": "Information Technology"},
        {
            "Symbol": "MSFT",
            "Name": "Microsoft Corp",
            "Sector": "Information Technology",
        },
        {
            "Symbol": "AMZN",
            "Name": "Amazon.com Inc",
            "Sector": "Consumer Discretionary",
        },
        {"Symbol": "GOOGL", "Name": "Alphabet Inc", "Sector": "Communication Services"},
        {"Symbol": "JPM", "Name": "JPMorgan Chase", "Sector": "Financials"},
        {"Symbol": "NVDA", "Name": "NVIDIA Corp", "Sector": "Information Technology"},
        {
            "Symbol": "META",
            "Name": "Meta Platforms",
            "Sector": "Communication Services",
        },
    ]
)


def _patch_sp500_csv():
    return patch("brain_api.universe.sp500.pd.read_csv", return_value=MOCK_SP500_DF)


# ---------------------------------------------------------------------------
# Mock data for Halal_New scrapers (replaces live HTTP to sp-funds / wahed / alpaca)
# ---------------------------------------------------------------------------
_MOCK_SP_FUNDS = {
    "spus": [
        {"symbol": "AAPL", "name": "Apple Inc", "weight": 10.0},
        {"symbol": "MSFT", "name": "Microsoft Corp", "weight": 9.5},
    ],
    "spte": [
        {"symbol": "NVDA", "name": "NVIDIA Corp", "weight": 8.0},
        {"symbol": "AAPL", "name": "Apple Inc", "weight": 7.0},
    ],
    "spwo": [
        {"symbol": "AMZN", "name": "Amazon.com Inc", "weight": 6.0},
    ],
}

_MOCK_WAHED = {
    "hlal": [
        {"symbol": "GOOGL", "name": "Alphabet Inc", "weight": 5.5},
        {"symbol": "MSFT", "name": "Microsoft Corp", "weight": 5.0},
    ],
    "umma": [
        {"symbol": "META", "name": "Meta Platforms", "weight": 4.0},
    ],
}

_MOCK_ALPACA_TRADABLE: set[str] = {
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "SPUS",
    "SPTE",
    "SPWO",
    "HLAL",
    "UMMA",
}


def _patch_halal_new_scrapers():
    """Context-manager stack that mocks the 3 external scraper calls."""
    return (
        patch(
            "brain_api.universe.halal_new.scrape_sp_funds",
            side_effect=lambda slug: _MOCK_SP_FUNDS[slug],
        ),
        patch(
            "brain_api.universe.halal_new.scrape_wahed",
            side_effect=lambda slug: _MOCK_WAHED[slug],
        ),
        patch(
            "brain_api.universe.halal_new.fetch_alpaca_tradable_symbols",
            return_value=_MOCK_ALPACA_TRADABLE,
        ),
    )


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

    with _patch_sp500_csv():
        data = get_sp500_universe()

    assert "stocks" in data
    assert "source" in data
    assert "total_stocks" in data
    assert "fetched_at" in data

    assert data["source"] == "datahub.io"


def test_get_sp500_universe_returns_stocks():
    """Test that get_sp500_universe returns stocks."""
    from brain_api.universe.sp500 import get_sp500_universe

    with _patch_sp500_csv():
        data = get_sp500_universe()

    assert data["total_stocks"] > 0
    assert len(data["stocks"]) > 0
    assert len(data["stocks"]) == data["total_stocks"]


def test_get_sp500_universe_stock_structure():
    """Test that each S&P 500 stock has required fields."""
    from brain_api.universe.sp500 import get_sp500_universe

    with _patch_sp500_csv():
        data = get_sp500_universe()

    for stock in data["stocks"]:
        assert "symbol" in stock
        assert "name" in stock
        assert "sector" in stock

        assert isinstance(stock["symbol"], str)
        assert isinstance(stock["name"], str)
        assert isinstance(stock["sector"], str)
        assert len(stock["symbol"]) > 0


def test_get_sp500_symbols_returns_list():
    """Test that get_sp500_symbols returns a list of symbols."""
    from brain_api.universe.sp500 import get_sp500_symbols

    with _patch_sp500_csv():
        symbols = get_sp500_symbols()

    assert isinstance(symbols, list)
    assert len(symbols) > 0
    assert all(isinstance(s, str) for s in symbols)


def test_get_sp500_universe_contains_known_stocks():
    """Test that S&P 500 contains well-known stocks."""
    from brain_api.universe.sp500 import get_sp500_symbols

    with _patch_sp500_csv():
        symbols = get_sp500_symbols()

    known_stocks = ["AAPL", "MSFT", "AMZN", "GOOGL", "JPM"]
    for stock in known_stocks:
        assert stock in symbols, f"Expected {stock} in S&P 500"


# ============================================================================
# Halal_New Universe Tests
# ============================================================================


def test_get_halal_new_stocks_returns_expected_structure():
    """Test that /universe/halal_new returns the expected response structure."""
    p1, p2, p3 = _patch_halal_new_scrapers()
    with p1, p2, p3:
        response = client.get("/universe/halal_new")

    assert response.status_code == 200
    data = response.json()

    assert "stocks" in data
    assert "etfs_used" in data
    assert "total_stocks" in data
    assert "fetched_at" in data


def test_get_halal_new_stocks_returns_stocks():
    """Test that /universe/halal_new returns at least some stocks."""
    p1, p2, p3 = _patch_halal_new_scrapers()
    with p1, p2, p3:
        response = client.get("/universe/halal_new")

    assert response.status_code == 200
    data = response.json()

    assert data["total_stocks"] > 0
    assert len(data["stocks"]) > 0
    assert len(data["stocks"]) == data["total_stocks"]


def test_get_halal_new_stocks_no_duplicates():
    """Test that /universe/halal_new returns unique symbols only."""
    p1, p2, p3 = _patch_halal_new_scrapers()
    with p1, p2, p3:
        response = client.get("/universe/halal_new")

    assert response.status_code == 200
    data = response.json()

    symbols = [stock["symbol"] for stock in data["stocks"]]
    assert len(symbols) == len(set(symbols)), "Duplicate symbols found"


def test_get_halal_new_stocks_stock_structure():
    """Test that each Halal_New stock has required fields."""
    p1, p2, p3 = _patch_halal_new_scrapers()
    with p1, p2, p3:
        response = client.get("/universe/halal_new")

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


def test_get_halal_new_stocks_sorted_by_weight():
    """Test that Halal_New stocks are sorted by max_weight descending."""
    p1, p2, p3 = _patch_halal_new_scrapers()
    with p1, p2, p3:
        response = client.get("/universe/halal_new")

    assert response.status_code == 200
    data = response.json()

    weights = [stock["max_weight"] for stock in data["stocks"]]
    assert weights == sorted(weights, reverse=True), "Stocks not sorted by weight"


def test_get_halal_new_etfs_used():
    """Test that /universe/halal_new uses all 5 halal ETFs."""
    p1, p2, p3 = _patch_halal_new_scrapers()
    with p1, p2, p3:
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


def test_get_halal_filtered_returns_503_on_yfinance_failure():
    """Test that /universe/halal_filtered returns 503 when yfinance rate-limits us."""
    mock_universe = _make_mock_halal_new_universe()

    with (
        patch(
            "brain_api.universe.halal_filtered.get_halal_new_universe",
            return_value=mock_universe,
        ),
        patch(
            "brain_api.universe.halal_filtered.fetch_stock_metrics",
            side_effect=YFinanceFetchError("18/20 failed (90.0%)"),
        ),
    ):
        response = client.get("/universe/halal_filtered")

    assert response.status_code == 503
    data = response.json()
    assert "18/20" in data["detail"]


# ============================================================================
# Halal_India Universe Tests
# ============================================================================


MOCK_NSE_CONSTITUENTS = [
    {"symbol": "INFY", "name": "Infosys Ltd.", "industry": "IT - Software"},
    {"symbol": "TCS", "name": "Tata Consultancy", "industry": "IT - Software"},
    {"symbol": "RELIANCE", "name": "Reliance Industries", "industry": "Oil & Gas"},
    {"symbol": "HDFCBANK", "name": "HDFC Bank", "industry": "Banking"},
    {"symbol": "WIPRO", "name": "Wipro Ltd.", "industry": "IT - Software"},
    {"symbol": "BAJFINANCE", "name": "Bajaj Finance", "industry": "Finance"},
    {"symbol": "MARUTI", "name": "Maruti Suzuki", "industry": "Automobile"},
    {"symbol": "NESTLEIND", "name": "Nestle India", "industry": "FMCG"},
    {"symbol": "TITAN", "name": "Titan Company", "industry": "Consumer Goods"},
    {"symbol": "ULTRACEMCO", "name": "UltraTech Cement", "industry": "Cement"},
    {"symbol": "DRREDDY", "name": "Dr. Reddys Labs", "industry": "Pharma"},
    {"symbol": "CIPLA", "name": "Cipla Ltd.", "industry": "Pharma"},
    {"symbol": "EICHERMOT", "name": "Eicher Motors", "industry": "Automobile"},
    {"symbol": "DIVISLAB", "name": "Divis Labs", "industry": "Pharma"},
    {"symbol": "BRITANNIA", "name": "Britannia Industries", "industry": "FMCG"},
    {"symbol": "TECHM", "name": "Tech Mahindra", "industry": "IT - Software"},
    {"symbol": "HINDUNILVR", "name": "Hindustan Unilever", "industry": "FMCG"},
    {"symbol": "HEROMOTOCO", "name": "Hero MotoCorp", "industry": "Automobile"},
]


def _make_mock_india_metrics(symbols: list[str]) -> dict[str, dict]:
    """Create mock metrics for .NS suffixed symbols."""
    metrics = {}
    for i, sym in enumerate(symbols):
        metrics[sym] = {
            "roe": 0.15 + i * 0.01,
            "price": 1500.0 + i * 100,
            "sma200": 1400.0,
            "beta": 0.8 + i * 0.05,
            "gross_margin": 0.35 + i * 0.01,
            "roic": 0.10 + i * 0.005,
            "earnings_yield": 0.04 + i * 0.002,
            "six_month_return": 0.06 + i * 0.01,
        }
    return metrics


def test_get_halal_india_returns_expected_structure():
    """Test that /universe/halal_india returns the expected response structure."""
    ns_symbols = [c["symbol"] + ".NS" for c in MOCK_NSE_CONSTITUENTS]
    mock_metrics = _make_mock_india_metrics(ns_symbols)

    with (
        patch(
            "brain_api.universe.halal_india.scrape_nifty500_shariah",
            return_value=MOCK_NSE_CONSTITUENTS,
        ),
        patch(
            "brain_api.universe.halal_india.fetch_stock_metrics",
            return_value=mock_metrics,
        ),
    ):
        response = client.get("/universe/halal_india")

    assert response.status_code == 200
    data = response.json()

    assert "stocks" in data
    assert "source" in data
    assert "symbol_suffix" in data
    assert "total_stocks" in data
    assert "total_scored" in data
    assert "top_n" in data
    assert "fetched_at" in data

    assert data["source"] == "nifty_500_shariah"
    assert data["symbol_suffix"] == ".NS"


def test_get_halal_india_returns_max_15_stocks():
    """Test that /universe/halal_india returns at most 15 stocks."""
    ns_symbols = [c["symbol"] + ".NS" for c in MOCK_NSE_CONSTITUENTS]
    mock_metrics = _make_mock_india_metrics(ns_symbols)

    with (
        patch(
            "brain_api.universe.halal_india.scrape_nifty500_shariah",
            return_value=MOCK_NSE_CONSTITUENTS,
        ),
        patch(
            "brain_api.universe.halal_india.fetch_stock_metrics",
            return_value=mock_metrics,
        ),
    ):
        response = client.get("/universe/halal_india")

    assert response.status_code == 200
    data = response.json()

    assert len(data["stocks"]) == 15
    assert data["top_n"] == 15


def test_get_halal_india_stocks_have_factor_scores():
    """Test that each halal_india stock has factor_score and factor_components."""
    ns_symbols = [c["symbol"] + ".NS" for c in MOCK_NSE_CONSTITUENTS]
    mock_metrics = _make_mock_india_metrics(ns_symbols)

    with (
        patch(
            "brain_api.universe.halal_india.scrape_nifty500_shariah",
            return_value=MOCK_NSE_CONSTITUENTS,
        ),
        patch(
            "brain_api.universe.halal_india.fetch_stock_metrics",
            return_value=mock_metrics,
        ),
    ):
        response = client.get("/universe/halal_india")

    assert response.status_code == 200
    data = response.json()

    for stock in data["stocks"]:
        assert "factor_score" in stock
        assert "factor_components" in stock
        assert "metrics" in stock
        assert stock["factor_score"] is not None


def test_get_halal_india_symbols_are_clean():
    """Test that returned symbols do NOT have .NS suffix."""
    ns_symbols = [c["symbol"] + ".NS" for c in MOCK_NSE_CONSTITUENTS]
    mock_metrics = _make_mock_india_metrics(ns_symbols)

    with (
        patch(
            "brain_api.universe.halal_india.scrape_nifty500_shariah",
            return_value=MOCK_NSE_CONSTITUENTS,
        ),
        patch(
            "brain_api.universe.halal_india.fetch_stock_metrics",
            return_value=mock_metrics,
        ),
    ):
        response = client.get("/universe/halal_india")

    assert response.status_code == 200
    data = response.json()

    for stock in data["stocks"]:
        assert not stock["symbol"].endswith(".NS"), (
            f"Symbol {stock['symbol']} should not have .NS suffix"
        )


def test_get_halal_india_returns_503_on_yfinance_failure():
    """Test that /universe/halal_india returns 503 when yfinance rate-limits us."""
    with (
        patch(
            "brain_api.universe.halal_india.scrape_nifty500_shariah",
            return_value=MOCK_NSE_CONSTITUENTS,
        ),
        patch(
            "brain_api.universe.halal_india.fetch_stock_metrics",
            side_effect=YFinanceFetchError("90/100 failed (90.0%)"),
        ),
    ):
        response = client.get("/universe/halal_india")

    assert response.status_code == 503
    assert "90/100" in response.json()["detail"]


def test_get_halal_india_returns_503_on_nse_failure():
    """Test that /universe/halal_india returns 503 when NSE scraper fails."""
    from brain_api.universe.scrapers.nse import NseFetchError

    with patch(
        "brain_api.universe.halal_india.scrape_nifty500_shariah",
        side_effect=NseFetchError("NSE API returned empty data"),
    ):
        response = client.get("/universe/halal_india")

    assert response.status_code == 503
    assert "NSE API" in response.json()["detail"]


# ============================================================================
# NSE Scraper Unit Tests
# ============================================================================


def test_scrape_nifty500_shariah_parses_response():
    """Test that scrape_nifty500_shariah correctly parses NSE JSON API response."""
    from brain_api.universe.scrapers.nse import scrape_nifty500_shariah

    mock_nse_json = {
        "data": [
            {
                "symbol": "Nifty 500 Shariah",
                "open": 5000.0,
                "dayHigh": 5100.0,
            },
            {
                "symbol": "INFY",
                "meta": {"companyName": "Infosys Ltd.", "industry": "IT - Software"},
                "lastPrice": 1500.0,
            },
            {
                "symbol": "TCS",
                "meta": {
                    "companyName": "Tata Consultancy",
                    "industry": "IT - Software",
                },
                "lastPrice": 3500.0,
            },
        ]
    }

    mock_session = patch(
        "brain_api.universe.scrapers.nse.requests.Session",
    )

    with mock_session as mock_sess_cls:
        session_instance = mock_sess_cls.return_value
        homepage_resp = type(
            "Resp", (), {"status_code": 200, "raise_for_status": lambda self: None}
        )()
        api_resp = type(
            "Resp",
            (),
            {
                "status_code": 200,
                "raise_for_status": lambda self: None,
                "json": lambda self: mock_nse_json,
            },
        )()
        session_instance.get.side_effect = [homepage_resp, api_resp]
        session_instance.headers = {}

        result = scrape_nifty500_shariah()

    assert len(result) == 2
    assert result[0]["symbol"] == "INFY"
    assert result[0]["name"] == "Infosys Ltd."
    assert result[1]["symbol"] == "TCS"


def test_scrape_nifty500_shariah_filters_index_row():
    """Test that the index summary row (with spaces in symbol) is filtered out."""
    from brain_api.universe.scrapers.nse import scrape_nifty500_shariah

    mock_nse_json = {
        "data": [
            {"symbol": "Nifty 500 Shariah", "open": 5000.0},
            {
                "symbol": "RELIANCE",
                "meta": {"companyName": "Reliance", "industry": "Oil"},
            },
        ]
    }

    with patch("brain_api.universe.scrapers.nse.requests.Session") as mock_sess_cls:
        session_instance = mock_sess_cls.return_value
        homepage_resp = type(
            "Resp", (), {"status_code": 200, "raise_for_status": lambda self: None}
        )()
        api_resp = type(
            "Resp",
            (),
            {
                "status_code": 200,
                "raise_for_status": lambda self: None,
                "json": lambda self: mock_nse_json,
            },
        )()
        session_instance.get.side_effect = [homepage_resp, api_resp]
        session_instance.headers = {}

        result = scrape_nifty500_shariah()

    assert len(result) == 1
    assert result[0]["symbol"] == "RELIANCE"


def test_scrape_nifty500_shariah_raises_on_empty_data():
    """Test that NseFetchError is raised when data array is empty."""
    from brain_api.universe.scrapers.nse import NseFetchError, scrape_nifty500_shariah

    mock_nse_json = {"data": []}

    with patch("brain_api.universe.scrapers.nse.requests.Session") as mock_sess_cls:
        session_instance = mock_sess_cls.return_value
        homepage_resp = type(
            "Resp", (), {"status_code": 200, "raise_for_status": lambda self: None}
        )()
        api_resp = type(
            "Resp",
            (),
            {
                "status_code": 200,
                "raise_for_status": lambda self: None,
                "json": lambda self: mock_nse_json,
            },
        )()
        session_instance.get.side_effect = [homepage_resp, api_resp]
        session_instance.headers = {}

        with pytest.raises(NseFetchError, match="empty data"):
            scrape_nifty500_shariah()


def test_scrape_nifty500_shariah_raises_on_http_error():
    """Test that NseFetchError is raised on HTTP errors."""
    import requests as req

    from brain_api.universe.scrapers.nse import NseFetchError, scrape_nifty500_shariah

    with patch("brain_api.universe.scrapers.nse.requests.Session") as mock_sess_cls:
        session_instance = mock_sess_cls.return_value
        session_instance.get.side_effect = req.RequestException("Connection refused")
        session_instance.headers = {}

        with pytest.raises(NseFetchError, match="NSE session attempts failed"):
            scrape_nifty500_shariah()
