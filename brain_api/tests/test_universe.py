"""Tests for universe endpoints."""

from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from brain_api.main import app

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
# filter_symbols_by_min_history Unit Tests
# ============================================================================


def test_filter_symbols_by_min_history_passes_qualifying():
    """Test that symbols with enough history pass through."""
    from datetime import date

    symbols = ["AAPL", "MSFT", "SOLS"]
    mock_prices = {
        "AAPL": pd.DataFrame(
            {"close": range(300)}, index=pd.date_range("2025-01-01", periods=300)
        ),
        "MSFT": pd.DataFrame(
            {"close": range(252)}, index=pd.date_range("2025-03-01", periods=252)
        ),
        "SOLS": pd.DataFrame(
            {"close": range(89)}, index=pd.date_range("2025-10-20", periods=89)
        ),
    }

    with patch("brain_api.core.prices.load_prices_yfinance", return_value=mock_prices):
        from brain_api.core.prices import filter_symbols_by_min_history

        qualifying, excluded = filter_symbols_by_min_history(
            symbols, min_trading_days=252, reference_date=date(2026, 3, 1)
        )

    assert qualifying == ["AAPL", "MSFT"]
    assert len(excluded) == 1
    assert excluded[0] == ("SOLS", 89)


def test_filter_symbols_by_min_history_boundary_252():
    """Test boundary: exactly 252 days passes, 251 excluded."""
    from datetime import date

    symbols = ["EXACT", "SHORT"]
    mock_prices = {
        "EXACT": pd.DataFrame(
            {"close": range(252)}, index=pd.date_range("2025-01-01", periods=252)
        ),
        "SHORT": pd.DataFrame(
            {"close": range(251)}, index=pd.date_range("2025-01-01", periods=251)
        ),
    }

    with patch("brain_api.core.prices.load_prices_yfinance", return_value=mock_prices):
        from brain_api.core.prices import filter_symbols_by_min_history

        qualifying, excluded = filter_symbols_by_min_history(
            symbols, min_trading_days=252, reference_date=date(2026, 3, 1)
        )

    assert qualifying == ["EXACT"]
    assert excluded == [("SHORT", 251)]


def test_filter_symbols_by_min_history_no_data():
    """Test that symbols with no price data are excluded with 0 days."""
    from datetime import date

    symbols = ["GOOD", "MISSING"]
    mock_prices = {
        "GOOD": pd.DataFrame(
            {"close": range(300)}, index=pd.date_range("2025-01-01", periods=300)
        ),
    }

    with patch("brain_api.core.prices.load_prices_yfinance", return_value=mock_prices):
        from brain_api.core.prices import filter_symbols_by_min_history

        qualifying, excluded = filter_symbols_by_min_history(
            symbols, min_trading_days=252, reference_date=date(2026, 3, 1)
        )

    assert qualifying == ["GOOD"]
    assert excluded == [("MISSING", 0)]


def test_filter_symbols_by_min_history_empty_input():
    """Test that empty symbol list returns empty results."""
    from datetime import date

    from brain_api.core.prices import filter_symbols_by_min_history

    qualifying, excluded = filter_symbols_by_min_history(
        [], min_trading_days=252, reference_date=date(2026, 3, 1)
    )

    assert qualifying == []
    assert excluded == []


# ============================================================================
# compute_min_walkforward_days Unit Tests
# ============================================================================


def test_compute_min_walkforward_days_default_10yr_window():
    """Dynamic threshold for default 10-year training window is ~2600+ days."""
    from datetime import date

    from brain_api.core.prices import compute_min_walkforward_days

    with patch(
        "brain_api.core.config.resolve_training_window",
        return_value=(date(2016, 1, 1), date(2026, 2, 27)),
    ):
        result = compute_min_walkforward_days(date(2026, 2, 28))

    assert result > 2500
    assert result < 3000


def test_compute_min_walkforward_days_5yr_window():
    """A 5-year training window produces a proportionally smaller threshold."""
    from datetime import date

    from brain_api.core.prices import compute_min_walkforward_days

    with patch(
        "brain_api.core.config.resolve_training_window",
        return_value=(date(2021, 1, 1), date(2026, 2, 27)),
    ):
        result = compute_min_walkforward_days(date(2026, 2, 28))

    assert result > 1200
    assert result < 1600


def test_compute_min_walkforward_days_includes_lstm_buffer():
    """Threshold includes buffer beyond the raw training window span."""
    from datetime import date

    from brain_api.core.prices import compute_min_walkforward_days

    training_start = date(2016, 1, 1)
    cutoff = date(2026, 2, 28)
    raw_span_days = (cutoff - training_start).days
    raw_trading_days = int(raw_span_days * 252 / 365)

    with patch(
        "brain_api.core.config.resolve_training_window",
        return_value=(training_start, date(2026, 2, 27)),
    ):
        result = compute_min_walkforward_days(cutoff)

    assert result > raw_trading_days


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


def _make_mock_batch_inference_result(symbols: list[str]):
    """Create a mock BatchInferenceResult for testing halal_filtered."""
    from brain_api.core.patchtst.inference import BatchInferenceResult, SymbolPrediction

    predictions = [
        SymbolPrediction(
            symbol=sym,
            predicted_weekly_return_pct=round(5.0 - i * 0.3, 4),
            direction="UP" if (5.0 - i * 0.3) > 0 else "DOWN",
            has_enough_history=True,
            history_days_used=120,
            data_end_date="2026-02-27",
            target_week_start="2026-03-02",
            target_week_end="2026-03-06",
        )
        for i, sym in enumerate(symbols)
    ]
    # Sort desc (mimics run_batch_inference)
    predictions.sort(key=lambda p: p.predicted_weekly_return_pct or 0, reverse=True)
    return BatchInferenceResult(
        predictions=predictions, model_version="v2026-03-01-abc123"
    )


def _mock_history_filter_pass_all(symbols: list[str]):
    """Return a mock for filter_symbols_by_min_history that passes all symbols."""
    return lambda syms, min_days, ref_date: (syms, [])


def _patch_min_wf_days():
    """Mock compute_min_walkforward_days to avoid resolve_training_window dependency."""
    return patch(
        "brain_api.universe.halal_filtered.compute_min_walkforward_days",
        return_value=2660,
    )


def test_get_halal_filtered_returns_expected_structure():
    """Test that /universe/halal_filtered returns the expected structure."""
    mock_universe = _make_mock_halal_new_universe()
    symbols = [s["symbol"] for s in mock_universe["stocks"]]
    mock_result = _make_mock_batch_inference_result(symbols)

    with (
        patch(
            "brain_api.universe.halal_filtered.get_halal_new_universe",
            return_value=mock_universe,
        ),
        _patch_min_wf_days(),
        patch(
            "brain_api.universe.halal_filtered.filter_symbols_by_min_history",
            side_effect=_mock_history_filter_pass_all(symbols),
        ),
        patch(
            "brain_api.universe.halal_filtered.run_batch_inference",
            return_value=mock_result,
        ),
    ):
        response = client.get("/universe/halal_filtered")

    assert response.status_code == 200
    data = response.json()

    assert "stocks" in data
    assert "total_candidates" in data
    assert "total_universe" in data
    assert "filtered_insufficient_history" in data
    assert "top_n" in data
    assert "selection_method" in data
    assert data["selection_method"] == "patchtst_forecast_rank_band"
    assert "model_version" in data
    assert "fetched_at" in data
    assert data["partition"] == "halal_filtered_alpha"
    assert "period_key" in data
    assert data["k_in"] == 15
    assert data["k_hold"] == 30
    assert data["previous_period_key_used"] is None
    assert data["kept_count"] == 0
    assert data["fillers_count"] == len(data["stocks"])
    assert data["evicted_from_previous"] == {}


def test_get_halal_filtered_returns_max_15_stocks():
    """Test that /universe/halal_filtered returns at most 15 stocks."""
    mock_universe = _make_mock_halal_new_universe(count=25)
    symbols = [s["symbol"] for s in mock_universe["stocks"]]
    mock_result = _make_mock_batch_inference_result(symbols)

    with (
        patch(
            "brain_api.universe.halal_filtered.get_halal_new_universe",
            return_value=mock_universe,
        ),
        _patch_min_wf_days(),
        patch(
            "brain_api.universe.halal_filtered.filter_symbols_by_min_history",
            side_effect=_mock_history_filter_pass_all(symbols),
        ),
        patch(
            "brain_api.universe.halal_filtered.run_batch_inference",
            return_value=mock_result,
        ),
    ):
        response = client.get("/universe/halal_filtered")

    assert response.status_code == 200
    data = response.json()

    assert len(data["stocks"]) == 15
    assert data["top_n"] == 15


def test_get_halal_filtered_stocks_have_predicted_returns():
    """Test that each halal_filtered stock has predicted_weekly_return_pct and rank."""
    mock_universe = _make_mock_halal_new_universe()
    symbols = [s["symbol"] for s in mock_universe["stocks"]]
    mock_result = _make_mock_batch_inference_result(symbols)

    with (
        patch(
            "brain_api.universe.halal_filtered.get_halal_new_universe",
            return_value=mock_universe,
        ),
        _patch_min_wf_days(),
        patch(
            "brain_api.universe.halal_filtered.filter_symbols_by_min_history",
            side_effect=_mock_history_filter_pass_all(symbols),
        ),
        patch(
            "brain_api.universe.halal_filtered.run_batch_inference",
            return_value=mock_result,
        ),
    ):
        response = client.get("/universe/halal_filtered")

    assert response.status_code == 200
    data = response.json()

    for stock in data["stocks"]:
        assert "predicted_weekly_return_pct" in stock
        assert "rank" in stock
        assert stock["predicted_weekly_return_pct"] is not None
        assert stock["rank"] >= 1


def test_get_halal_filtered_returns_503_when_no_model():
    """Test that /universe/halal_filtered returns 503 when no PatchTST model."""
    mock_universe = _make_mock_halal_new_universe()
    symbols = [s["symbol"] for s in mock_universe["stocks"]]

    with (
        patch(
            "brain_api.universe.halal_filtered.get_halal_new_universe",
            return_value=mock_universe,
        ),
        _patch_min_wf_days(),
        patch(
            "brain_api.universe.halal_filtered.filter_symbols_by_min_history",
            side_effect=_mock_history_filter_pass_all(symbols),
        ),
        patch(
            "brain_api.universe.halal_filtered.run_batch_inference",
            side_effect=ValueError("No current PatchTST model version available"),
        ),
    ):
        response = client.get("/universe/halal_filtered")

    assert response.status_code == 503
    data = response.json()
    assert "PatchTST" in data["detail"]


def test_get_halal_filtered_sorted_by_return_desc():
    """Test that top N stocks are sorted by predicted return (highest first)."""
    mock_universe = _make_mock_halal_new_universe()
    symbols = [s["symbol"] for s in mock_universe["stocks"]]
    mock_result = _make_mock_batch_inference_result(symbols)

    with (
        patch(
            "brain_api.universe.halal_filtered.get_halal_new_universe",
            return_value=mock_universe,
        ),
        _patch_min_wf_days(),
        patch(
            "brain_api.universe.halal_filtered.filter_symbols_by_min_history",
            side_effect=_mock_history_filter_pass_all(symbols),
        ),
        patch(
            "brain_api.universe.halal_filtered.run_batch_inference",
            return_value=mock_result,
        ),
    ):
        response = client.get("/universe/halal_filtered")

    data = response.json()
    returns = [s["predicted_weekly_return_pct"] for s in data["stocks"]]
    assert returns == sorted(returns, reverse=True)


def test_get_halal_filtered_excludes_none_predictions():
    """Test that predictions with None return are excluded from results."""
    from brain_api.core.patchtst.inference import BatchInferenceResult, SymbolPrediction

    mock_universe = _make_mock_halal_new_universe(count=5)
    symbols = [s["symbol"] for s in mock_universe["stocks"]]

    predictions = [
        SymbolPrediction(
            symbol=symbols[0],
            predicted_weekly_return_pct=2.5,
            direction="UP",
            has_enough_history=True,
            history_days_used=120,
            data_end_date="2026-02-27",
            target_week_start="2026-03-02",
            target_week_end="2026-03-06",
        ),
        SymbolPrediction(
            symbol=symbols[1],
            predicted_weekly_return_pct=None,
            direction="FLAT",
            has_enough_history=False,
            history_days_used=10,
            data_end_date=None,
            target_week_start="2026-03-02",
            target_week_end="2026-03-06",
        ),
        SymbolPrediction(
            symbol=symbols[2],
            predicted_weekly_return_pct=1.0,
            direction="UP",
            has_enough_history=True,
            history_days_used=120,
            data_end_date="2026-02-27",
            target_week_start="2026-03-02",
            target_week_end="2026-03-06",
        ),
    ]
    mock_result = BatchInferenceResult(
        predictions=predictions, model_version="v2026-03-01-abc123"
    )

    with (
        patch(
            "brain_api.universe.halal_filtered.get_halal_new_universe",
            return_value=mock_universe,
        ),
        _patch_min_wf_days(),
        patch(
            "brain_api.universe.halal_filtered.filter_symbols_by_min_history",
            side_effect=_mock_history_filter_pass_all(symbols),
        ),
        patch(
            "brain_api.universe.halal_filtered.run_batch_inference",
            return_value=mock_result,
        ),
    ):
        response = client.get("/universe/halal_filtered")

    data = response.json()
    result_symbols = [s["symbol"] for s in data["stocks"]]
    assert symbols[1] not in result_symbols
    assert data["total_candidates"] == 2


def test_get_halal_filtered_fewer_than_15_valid():
    """Test that if fewer than 15 valid predictions, all valid are returned."""
    mock_universe = _make_mock_halal_new_universe(count=5)
    symbols = [s["symbol"] for s in mock_universe["stocks"]]
    mock_result = _make_mock_batch_inference_result(symbols)

    with (
        patch(
            "brain_api.universe.halal_filtered.get_halal_new_universe",
            return_value=mock_universe,
        ),
        _patch_min_wf_days(),
        patch(
            "brain_api.universe.halal_filtered.filter_symbols_by_min_history",
            side_effect=_mock_history_filter_pass_all(symbols),
        ),
        patch(
            "brain_api.universe.halal_filtered.run_batch_inference",
            return_value=mock_result,
        ),
    ):
        response = client.get("/universe/halal_filtered")

    data = response.json()
    assert len(data["stocks"]) == 5
    assert data["total_candidates"] == 5


def test_get_halal_filtered_excludes_short_history_symbols():
    """Test that symbols with insufficient history are excluded before PatchTST."""
    mock_universe = _make_mock_halal_new_universe(count=20)
    all_symbols = [s["symbol"] for s in mock_universe["stocks"]]
    short_history = [("SYM0", 89), ("SYM1", 200)]
    qualifying = [s for s in all_symbols if s not in {"SYM0", "SYM1"}]
    mock_result = _make_mock_batch_inference_result(qualifying)

    with (
        patch(
            "brain_api.universe.halal_filtered.get_halal_new_universe",
            return_value=mock_universe,
        ),
        _patch_min_wf_days(),
        patch(
            "brain_api.universe.halal_filtered.filter_symbols_by_min_history",
            return_value=(qualifying, short_history),
        ),
        patch(
            "brain_api.universe.halal_filtered.run_batch_inference",
            return_value=mock_result,
        ) as mock_inference,
    ):
        response = client.get("/universe/halal_filtered")

    assert response.status_code == 200
    data = response.json()

    result_symbols = [s["symbol"] for s in data["stocks"]]
    assert "SYM0" not in result_symbols
    assert "SYM1" not in result_symbols
    assert data["filtered_insufficient_history"] == 2

    passed_to_inference = mock_inference.call_args[0][0]
    assert "SYM0" not in passed_to_inference
    assert "SYM1" not in passed_to_inference


def test_get_halal_filtered_short_history_count_in_response():
    """Test that filtered_insufficient_history count is correct."""
    mock_universe = _make_mock_halal_new_universe(count=10)
    all_symbols = [s["symbol"] for s in mock_universe["stocks"]]
    excluded = [("SYM0", 50), ("SYM3", 100), ("SYM7", 0)]
    qualifying = [s for s in all_symbols if s not in {"SYM0", "SYM3", "SYM7"}]
    mock_result = _make_mock_batch_inference_result(qualifying)

    with (
        patch(
            "brain_api.universe.halal_filtered.get_halal_new_universe",
            return_value=mock_universe,
        ),
        _patch_min_wf_days(),
        patch(
            "brain_api.universe.halal_filtered.filter_symbols_by_min_history",
            return_value=(qualifying, excluded),
        ),
        patch(
            "brain_api.universe.halal_filtered.run_batch_inference",
            return_value=mock_result,
        ),
    ):
        response = client.get("/universe/halal_filtered")

    data = response.json()
    assert data["filtered_insufficient_history"] == 3
    assert data["total_universe"] == 10


def test_get_halal_filtered_all_short_history_raises():
    """Test that if all symbols are excluded, the build raises (no silent fallback).

    With rank-band sticky selection, an empty ``current_scores`` map
    cannot be selected from -- the selector raises ``ValueError`` and
    the route surfaces it as 503. This is the correct behaviour per
    AGENTS.md AI rule 1 (no silent fallbacks): a fully-empty PatchTST
    score map indicates a real upstream failure (e.g. PatchTST broken,
    halal_new shrank to nothing eligible) and must not be served as a
    silently-empty universe.
    """
    from brain_api.core.patchtst.inference import BatchInferenceResult

    mock_universe = _make_mock_halal_new_universe(count=5)
    all_symbols = [s["symbol"] for s in mock_universe["stocks"]]
    excluded = [(s, 50) for s in all_symbols]
    mock_result = BatchInferenceResult(
        predictions=[], model_version="v2026-03-01-abc123"
    )

    with (
        patch(
            "brain_api.universe.halal_filtered.get_halal_new_universe",
            return_value=mock_universe,
        ),
        _patch_min_wf_days(),
        patch(
            "brain_api.universe.halal_filtered.filter_symbols_by_min_history",
            return_value=([], excluded),
        ),
        patch(
            "brain_api.universe.halal_filtered.run_batch_inference",
            return_value=mock_result,
        ),
    ):
        response = client.get("/universe/halal_filtered")

    assert response.status_code == 503


def test_get_halal_filtered_cold_start_matches_legacy_top15_for_unique_scores():
    """Cold-start rank-band selection on unique scores must match legacy top-15.

    Math invariant: when ``previous_selected_set is None`` and every
    score is unique, ``select_with_rank_band`` falls through to top-K_in
    by score desc -- the same ordering the legacy ``valid[:15]``
    blanket-top-15 produced. This is the safety property guaranteeing
    the first deploy is non-disruptive.
    """
    mock_universe = _make_mock_halal_new_universe(count=20)
    symbols = [s["symbol"] for s in mock_universe["stocks"]]
    mock_result = _make_mock_batch_inference_result(symbols)
    legacy_top15 = [p.symbol for p in mock_result.predictions[:15]]

    with (
        patch(
            "brain_api.universe.halal_filtered.get_halal_new_universe",
            return_value=mock_universe,
        ),
        _patch_min_wf_days(),
        patch(
            "brain_api.universe.halal_filtered.filter_symbols_by_min_history",
            side_effect=_mock_history_filter_pass_all(symbols),
        ),
        patch(
            "brain_api.universe.halal_filtered.run_batch_inference",
            return_value=mock_result,
        ),
    ):
        response = client.get("/universe/halal_filtered")

    assert response.status_code == 200
    data = response.json()

    selected = [s["symbol"] for s in data["stocks"]]
    assert selected == legacy_top15
    assert all(s["selection_reason"] == "top_rank" for s in data["stocks"])
    assert data["kept_count"] == 0
    assert data["fillers_count"] == 15


def test_get_halal_filtered_warm_start_keeps_held_stock_in_hold_band():
    """Warm-start: a previously-held stock whose rank slipped to <= K_hold is kept.

    Seeds the screening repository with a previous month's selected set
    that includes a symbol the next month ranks below K_in (=15) but
    within K_hold (=30). The second build must keep that symbol with
    ``selection_reason='sticky'`` and add fillers from this month's top
    of the rank.
    """
    from datetime import date

    from brain_api.core.screening_orchestration import persist_screening_rows
    from brain_api.core.sticky_selection import iso_year_week_of_month_anchor
    from brain_api.core.strategy_partitions import HALAL_FILTERED_ALPHA_PARTITION
    from brain_api.storage.screening_history import ScreeningHistoryRepository

    mock_universe = _make_mock_halal_new_universe(count=30)
    symbols = [s["symbol"] for s in mock_universe["stocks"]]
    mock_result = _make_mock_batch_inference_result(symbols)
    scores_by_symbol = {
        p.symbol: p.predicted_weekly_return_pct for p in mock_result.predictions
    }

    with patch(
        "brain_api.universe.halal_filtered.resolve_cutoff_date",
        return_value=date(2026, 4, 25),
    ):
        repo = ScreeningHistoryRepository()
        previous_period_key = iso_year_week_of_month_anchor(date(2026, 3, 15))
        prev_selected = {symbols[20]}
        prev_scores = scores_by_symbol
        persist_screening_rows(
            repo=repo,
            partition=HALAL_FILTERED_ALPHA_PARTITION,
            period_key=previous_period_key,
            as_of_date="2026-03-13",
            run_id="seed",
            scores=prev_scores,
            selected_set=prev_selected,
            selection_reasons={symbols[20]: "sticky"},
        )

        with (
            patch(
                "brain_api.universe.halal_filtered.get_halal_new_universe",
                return_value=mock_universe,
            ),
            _patch_min_wf_days(),
            patch(
                "brain_api.universe.halal_filtered.filter_symbols_by_min_history",
                side_effect=_mock_history_filter_pass_all(symbols),
            ),
            patch(
                "brain_api.universe.halal_filtered.run_batch_inference",
                return_value=mock_result,
            ),
        ):
            response = client.get("/universe/halal_filtered")

    assert response.status_code == 200
    data = response.json()

    selected = [s["symbol"] for s in data["stocks"]]
    assert symbols[20] in selected
    sticky_entry = next(s for s in data["stocks"] if s["symbol"] == symbols[20])
    assert sticky_entry["selection_reason"] == "sticky"
    assert data["kept_count"] == 1
    assert data["fillers_count"] == 14
    assert data["previous_period_key_used"] == previous_period_key


def test_get_halal_filtered_warm_start_evicts_dropped_from_universe():
    """Warm-start: previously-held stock missing from this month's scores evicts.

    Eviction reason should be ``dropped_from_universe`` and reported in
    ``evicted_from_previous``. Filler picks the next best symbol.
    """
    from datetime import date

    from brain_api.core.screening_orchestration import persist_screening_rows
    from brain_api.core.sticky_selection import iso_year_week_of_month_anchor
    from brain_api.core.strategy_partitions import HALAL_FILTERED_ALPHA_PARTITION
    from brain_api.storage.screening_history import ScreeningHistoryRepository

    mock_universe = _make_mock_halal_new_universe(count=20)
    symbols = [s["symbol"] for s in mock_universe["stocks"]]
    mock_result = _make_mock_batch_inference_result(symbols)
    delisted = "DELISTED_SYM"
    prev_scores = {
        p.symbol: p.predicted_weekly_return_pct for p in mock_result.predictions
    }
    prev_scores[delisted] = 99.0

    with patch(
        "brain_api.universe.halal_filtered.resolve_cutoff_date",
        return_value=date(2026, 4, 25),
    ):
        repo = ScreeningHistoryRepository()
        previous_period_key = iso_year_week_of_month_anchor(date(2026, 3, 15))
        persist_screening_rows(
            repo=repo,
            partition=HALAL_FILTERED_ALPHA_PARTITION,
            period_key=previous_period_key,
            as_of_date="2026-03-13",
            run_id="seed",
            scores=prev_scores,
            selected_set={delisted},
            selection_reasons={delisted: "top_rank"},
        )

        with (
            patch(
                "brain_api.universe.halal_filtered.get_halal_new_universe",
                return_value=mock_universe,
            ),
            _patch_min_wf_days(),
            patch(
                "brain_api.universe.halal_filtered.filter_symbols_by_min_history",
                side_effect=_mock_history_filter_pass_all(symbols),
            ),
            patch(
                "brain_api.universe.halal_filtered.run_batch_inference",
                return_value=mock_result,
            ),
        ):
            response = client.get("/universe/halal_filtered")

    assert response.status_code == 200
    data = response.json()
    assert delisted not in [s["symbol"] for s in data["stocks"]]
    assert delisted in data["evicted_from_previous"]
    assert data["evicted_from_previous"][delisted] == "dropped_from_universe"


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


def _make_mock_nifty_shariah_500_universe(
    constituents: list[dict] | None = None,
) -> dict:
    """Create a mock NiftyShariah500 universe with .NS-suffixed symbols."""
    if constituents is None:
        constituents = MOCK_NSE_CONSTITUENTS
    stocks = [{**c, "symbol": c["symbol"] + ".NS"} for c in constituents]
    return {
        "stocks": stocks,
        "source": "nifty_500_shariah",
        "symbol_suffix": ".NS",
        "total_stocks": len(stocks),
        "fetched_at": "2026-01-01T00:00:00+00:00",
    }


def _make_mock_india_batch_inference(symbols: list[str]):
    """Create a mock BatchInferenceResult for India PatchTST."""
    from brain_api.core.patchtst.inference import BatchInferenceResult, SymbolPrediction

    predictions = [
        SymbolPrediction(
            symbol=sym,
            predicted_weekly_return_pct=round(5.0 - i * 0.3, 4),
            direction="UP" if (5.0 - i * 0.3) > 0 else "DOWN",
            has_enough_history=True,
            history_days_used=120,
            data_end_date="2026-02-27",
            target_week_start="2026-03-02",
            target_week_end="2026-03-06",
        )
        for i, sym in enumerate(symbols)
    ]
    predictions.sort(key=lambda p: p.predicted_weekly_return_pct or 0, reverse=True)
    return BatchInferenceResult(
        predictions=predictions, model_version="v2026-03-01-india123"
    )


def _patch_india_min_wf_days():
    """Mock compute_min_walkforward_days for halal_india tests."""
    return patch(
        "brain_api.universe.halal_india.compute_min_walkforward_days",
        return_value=2660,
    )


def _mock_india_history_filter_pass_all(symbols: list[str]):
    """Return a mock for filter_symbols_by_min_history that passes all symbols."""
    return lambda syms, min_days, ref_date: (syms, [])


def test_get_halal_india_returns_expected_structure():
    """Test that /universe/halal_india returns the expected response structure."""
    mock_universe = _make_mock_nifty_shariah_500_universe()
    ns_symbols = [s["symbol"] for s in mock_universe["stocks"]]
    mock_result = _make_mock_india_batch_inference(ns_symbols)

    with (
        patch(
            "brain_api.universe.halal_india.get_nifty_shariah_500_universe",
            return_value=mock_universe,
        ),
        _patch_india_min_wf_days(),
        patch(
            "brain_api.universe.halal_india.filter_symbols_by_min_history",
            side_effect=_mock_india_history_filter_pass_all(ns_symbols),
        ),
        patch(
            "brain_api.universe.halal_india.run_batch_inference",
            return_value=mock_result,
        ),
    ):
        response = client.get("/universe/halal_india")

    assert response.status_code == 200
    data = response.json()

    assert "stocks" in data
    assert "total_candidates" in data
    assert "total_universe" in data
    assert "filtered_insufficient_history" in data
    assert "top_n" in data
    assert "selection_method" in data
    assert data["selection_method"] == "patchtst_forecast_rank_band"
    assert "model_version" in data
    assert "symbol_suffix" in data
    assert data["symbol_suffix"] == ".NS"
    assert "fetched_at" in data
    # Rank-band sticky additive fields
    assert data["partition"] == "halal_india_filtered_alpha"
    assert "period_key" in data
    assert "previous_period_key_used" in data
    assert "kept_count" in data
    assert "fillers_count" in data
    assert "evicted_from_previous" in data
    assert isinstance(data["evicted_from_previous"], dict)
    assert data["k_in"] == 15
    assert data["k_hold"] == 30


def test_get_halal_india_returns_max_15_stocks():
    """Test that /universe/halal_india returns at most 15 stocks."""
    mock_universe = _make_mock_nifty_shariah_500_universe()
    ns_symbols = [s["symbol"] for s in mock_universe["stocks"]]
    mock_result = _make_mock_india_batch_inference(ns_symbols)

    with (
        patch(
            "brain_api.universe.halal_india.get_nifty_shariah_500_universe",
            return_value=mock_universe,
        ),
        _patch_india_min_wf_days(),
        patch(
            "brain_api.universe.halal_india.filter_symbols_by_min_history",
            side_effect=_mock_india_history_filter_pass_all(ns_symbols),
        ),
        patch(
            "brain_api.universe.halal_india.run_batch_inference",
            return_value=mock_result,
        ),
    ):
        response = client.get("/universe/halal_india")

    assert response.status_code == 200
    data = response.json()

    assert len(data["stocks"]) == 15
    assert data["top_n"] == 15


def test_get_halal_india_stocks_have_predicted_returns():
    """Test that each halal_india stock has predicted_weekly_return_pct + rank + selection_reason.

    Per-stock ``model_version`` was REMOVED in the rank-band sticky
    rewrite (mirroring halal_filtered). ``model_version`` now lives at
    the TOP level only. This test asserts the new shape.
    """
    mock_universe = _make_mock_nifty_shariah_500_universe()
    ns_symbols = [s["symbol"] for s in mock_universe["stocks"]]
    mock_result = _make_mock_india_batch_inference(ns_symbols)

    with (
        patch(
            "brain_api.universe.halal_india.get_nifty_shariah_500_universe",
            return_value=mock_universe,
        ),
        _patch_india_min_wf_days(),
        patch(
            "brain_api.universe.halal_india.filter_symbols_by_min_history",
            side_effect=_mock_india_history_filter_pass_all(ns_symbols),
        ),
        patch(
            "brain_api.universe.halal_india.run_batch_inference",
            return_value=mock_result,
        ),
    ):
        response = client.get("/universe/halal_india")

    assert response.status_code == 200
    data = response.json()

    for stock in data["stocks"]:
        assert "predicted_weekly_return_pct" in stock
        assert "rank" in stock
        assert "selection_reason" in stock
        assert stock["selection_reason"] in ("sticky", "top_rank")
        assert stock["predicted_weekly_return_pct"] is not None
        assert stock["rank"] >= 1
        assert "model_version" not in stock, (
            "Per-stock model_version is removed in rank-band sticky shape"
        )


def test_get_halal_india_symbols_have_ns_suffix():
    """Test that returned symbols have .NS suffix (yfinance-ready)."""
    mock_universe = _make_mock_nifty_shariah_500_universe()
    ns_symbols = [s["symbol"] for s in mock_universe["stocks"]]
    mock_result = _make_mock_india_batch_inference(ns_symbols)

    with (
        patch(
            "brain_api.universe.halal_india.get_nifty_shariah_500_universe",
            return_value=mock_universe,
        ),
        _patch_india_min_wf_days(),
        patch(
            "brain_api.universe.halal_india.filter_symbols_by_min_history",
            side_effect=_mock_india_history_filter_pass_all(ns_symbols),
        ),
        patch(
            "brain_api.universe.halal_india.run_batch_inference",
            return_value=mock_result,
        ),
    ):
        response = client.get("/universe/halal_india")

    assert response.status_code == 200
    data = response.json()

    for stock in data["stocks"]:
        assert stock["symbol"].endswith(".NS"), (
            f"Symbol {stock['symbol']} should have .NS suffix"
        )


def test_get_halal_india_returns_503_when_no_model():
    """Test that /universe/halal_india returns 503 when no India PatchTST model."""
    mock_universe = _make_mock_nifty_shariah_500_universe()
    ns_symbols = [s["symbol"] for s in mock_universe["stocks"]]

    with (
        patch(
            "brain_api.universe.halal_india.get_nifty_shariah_500_universe",
            return_value=mock_universe,
        ),
        _patch_india_min_wf_days(),
        patch(
            "brain_api.universe.halal_india.filter_symbols_by_min_history",
            side_effect=_mock_india_history_filter_pass_all(ns_symbols),
        ),
        patch(
            "brain_api.universe.halal_india.run_batch_inference",
            side_effect=ValueError("No current PatchTST model version available"),
        ),
    ):
        response = client.get("/universe/halal_india")

    assert response.status_code == 503
    assert "PatchTST" in response.json()["detail"]


def test_get_halal_india_returns_503_on_nse_failure():
    """Test that /universe/halal_india returns 503 when NSE scraper fails."""
    from brain_api.universe.scrapers.nse import NseFetchError

    with patch(
        "brain_api.universe.halal_india.get_nifty_shariah_500_universe",
        side_effect=NseFetchError("NSE API returned empty data"),
    ):
        response = client.get("/universe/halal_india")

    assert response.status_code == 503
    assert "NSE API" in response.json()["detail"]


# ============================================================================
# Halal_India rank-band sticky tests (single-stage screening_history path)
# ============================================================================


def _make_mock_nifty_shariah_500_universe_n(count: int) -> dict:
    """Build a synthetic ``count``-stock NiftyShariah500 universe with .NS-suffixed symbols."""
    constituents = [
        {"symbol": f"INSYM{i:03d}", "name": f"Stock {i}", "industry": "Synthetic"}
        for i in range(count)
    ]
    stocks = [{**c, "symbol": c["symbol"] + ".NS"} for c in constituents]
    return {
        "stocks": stocks,
        "source": "nifty_500_shariah",
        "symbol_suffix": ".NS",
        "total_stocks": len(stocks),
        "fetched_at": "2026-01-01T00:00:00+00:00",
    }


def test_get_halal_india_cold_start_picks_top_15_by_score():
    """Cold-start: empty screening_history -> chosen 15 are top by score, .NS-suffixed."""
    from datetime import date

    mock_universe = _make_mock_nifty_shariah_500_universe_n(30)
    ns_symbols = [s["symbol"] for s in mock_universe["stocks"]]
    mock_result = _make_mock_india_batch_inference(ns_symbols)
    expected_top = [p.symbol for p in mock_result.predictions[:15]]

    with (
        patch(
            "brain_api.universe.halal_india.resolve_cutoff_date",
            return_value=date(2026, 4, 25),
        ),
        patch(
            "brain_api.universe.halal_india.get_nifty_shariah_500_universe",
            return_value=mock_universe,
        ),
        _patch_india_min_wf_days(),
        patch(
            "brain_api.universe.halal_india.filter_symbols_by_min_history",
            side_effect=_mock_india_history_filter_pass_all(ns_symbols),
        ),
        patch(
            "brain_api.universe.halal_india.run_batch_inference",
            return_value=mock_result,
        ),
    ):
        response = client.get("/universe/halal_india")

    assert response.status_code == 200
    data = response.json()
    assert [s["symbol"] for s in data["stocks"]] == expected_top
    assert all(s["symbol"].endswith(".NS") for s in data["stocks"])
    assert data["partition"] == "halal_india_filtered_alpha"
    assert data["period_key"] == "202615"  # First Monday of April 2026 = Apr 6
    assert data["previous_period_key_used"] is None
    assert data["kept_count"] == 0
    assert data["fillers_count"] == 15
    assert data["evicted_from_previous"] == {}
    assert all(s["selection_reason"] == "top_rank" for s in data["stocks"])


def test_get_halal_india_warm_start_keeps_sticky_within_k_hold():
    """Warm-start: previously-held .NS stock at rank between K_in and K_hold stays sticky."""
    from datetime import date

    from brain_api.core.screening_orchestration import persist_screening_rows
    from brain_api.core.sticky_selection import iso_year_week_of_month_anchor
    from brain_api.core.strategy_partitions import (
        HALAL_INDIA_FILTERED_ALPHA_PARTITION,
    )
    from brain_api.storage.screening_history import ScreeningHistoryRepository

    mock_universe = _make_mock_nifty_shariah_500_universe_n(30)
    ns_symbols = [s["symbol"] for s in mock_universe["stocks"]]
    mock_result = _make_mock_india_batch_inference(ns_symbols)
    scores_by_symbol = {
        p.symbol: p.predicted_weekly_return_pct for p in mock_result.predictions
    }
    sorted_symbols = [p.symbol for p in mock_result.predictions]

    with patch(
        "brain_api.universe.halal_india.resolve_cutoff_date",
        return_value=date(2026, 4, 25),
    ):
        repo = ScreeningHistoryRepository()
        previous_period_key = iso_year_week_of_month_anchor(date(2026, 3, 15))
        # Seed previous round with a .NS symbol that ranks 21 this period
        # (within K_hold=30 but beyond K_in=15).
        sticky_symbol = sorted_symbols[20]
        persist_screening_rows(
            repo=repo,
            partition=HALAL_INDIA_FILTERED_ALPHA_PARTITION,
            period_key=previous_period_key,
            as_of_date="2026-03-13",
            run_id="seed",
            scores=scores_by_symbol,
            selected_set={sticky_symbol},
            selection_reasons={sticky_symbol: "sticky"},
        )

        with (
            patch(
                "brain_api.universe.halal_india.get_nifty_shariah_500_universe",
                return_value=mock_universe,
            ),
            _patch_india_min_wf_days(),
            patch(
                "brain_api.universe.halal_india.filter_symbols_by_min_history",
                side_effect=_mock_india_history_filter_pass_all(ns_symbols),
            ),
            patch(
                "brain_api.universe.halal_india.run_batch_inference",
                return_value=mock_result,
            ),
        ):
            response = client.get("/universe/halal_india")

    assert response.status_code == 200
    data = response.json()
    selected_symbols = [s["symbol"] for s in data["stocks"]]
    assert sticky_symbol in selected_symbols
    assert all(sym.endswith(".NS") for sym in selected_symbols)
    sticky_entry = next(s for s in data["stocks"] if s["symbol"] == sticky_symbol)
    assert sticky_entry["selection_reason"] == "sticky"
    assert data["kept_count"] == 1
    assert data["fillers_count"] == 14
    assert data["previous_period_key_used"] == previous_period_key


def test_get_halal_india_warm_start_evicts_dropped_from_universe():
    """Warm-start: previously-held .NS stock missing this month -> evicted_from_previous.

    ``evicted_from_previous`` is a ``dict[str, str]`` (symbol -> reason).
    The .NS-suffixed delisted symbol must appear as a KEY with value
    ``'dropped_from_universe'``.
    """
    from datetime import date

    from brain_api.core.screening_orchestration import persist_screening_rows
    from brain_api.core.sticky_selection import iso_year_week_of_month_anchor
    from brain_api.core.strategy_partitions import (
        HALAL_INDIA_FILTERED_ALPHA_PARTITION,
    )
    from brain_api.storage.screening_history import ScreeningHistoryRepository

    mock_universe = _make_mock_nifty_shariah_500_universe_n(20)
    ns_symbols = [s["symbol"] for s in mock_universe["stocks"]]
    mock_result = _make_mock_india_batch_inference(ns_symbols)
    delisted = "DELISTED.NS"
    prev_scores = {
        p.symbol: p.predicted_weekly_return_pct for p in mock_result.predictions
    }
    prev_scores[delisted] = 99.0  # Previously top-ranked

    with patch(
        "brain_api.universe.halal_india.resolve_cutoff_date",
        return_value=date(2026, 4, 25),
    ):
        repo = ScreeningHistoryRepository()
        previous_period_key = iso_year_week_of_month_anchor(date(2026, 3, 15))
        persist_screening_rows(
            repo=repo,
            partition=HALAL_INDIA_FILTERED_ALPHA_PARTITION,
            period_key=previous_period_key,
            as_of_date="2026-03-13",
            run_id="seed",
            scores=prev_scores,
            selected_set={delisted},
            selection_reasons={delisted: "top_rank"},
        )

        with (
            patch(
                "brain_api.universe.halal_india.get_nifty_shariah_500_universe",
                return_value=mock_universe,
            ),
            _patch_india_min_wf_days(),
            patch(
                "brain_api.universe.halal_india.filter_symbols_by_min_history",
                side_effect=_mock_india_history_filter_pass_all(ns_symbols),
            ),
            patch(
                "brain_api.universe.halal_india.run_batch_inference",
                return_value=mock_result,
            ),
        ):
            response = client.get("/universe/halal_india")

    assert response.status_code == 200
    data = response.json()
    assert delisted not in [s["symbol"] for s in data["stocks"]]
    assert delisted in data["evicted_from_previous"]
    assert data["evicted_from_previous"][delisted] == "dropped_from_universe"


def test_get_halal_india_empty_scores_raises():
    """If India PatchTST returns nothing valid -> ValueError -> 503 (no fallback)."""
    from brain_api.core.patchtst.inference import BatchInferenceResult

    mock_universe = _make_mock_nifty_shariah_500_universe()
    ns_symbols = [s["symbol"] for s in mock_universe["stocks"]]
    empty_result = BatchInferenceResult(
        predictions=[], model_version="v2026-03-01-india123"
    )

    with (
        patch(
            "brain_api.universe.halal_india.get_nifty_shariah_500_universe",
            return_value=mock_universe,
        ),
        _patch_india_min_wf_days(),
        patch(
            "brain_api.universe.halal_india.filter_symbols_by_min_history",
            side_effect=_mock_india_history_filter_pass_all(ns_symbols),
        ),
        patch(
            "brain_api.universe.halal_india.run_batch_inference",
            return_value=empty_result,
        ),
    ):
        response = client.get("/universe/halal_india")

    # ValueError from select_with_rank_band surfaces as 503 from the route.
    assert response.status_code == 503


def test_get_halal_india_fewer_than_khold_valid_scores_still_works():
    """Rank-band tolerates fewer-than-K_hold valid scores (only top-15 is required)."""
    from datetime import date

    # 20 valid scores -- enough for K_in=15 but fewer than K_hold=30.
    mock_universe = _make_mock_nifty_shariah_500_universe_n(20)
    ns_symbols = [s["symbol"] for s in mock_universe["stocks"]]
    mock_result = _make_mock_india_batch_inference(ns_symbols)

    with (
        patch(
            "brain_api.universe.halal_india.resolve_cutoff_date",
            return_value=date(2026, 4, 25),
        ),
        patch(
            "brain_api.universe.halal_india.get_nifty_shariah_500_universe",
            return_value=mock_universe,
        ),
        _patch_india_min_wf_days(),
        patch(
            "brain_api.universe.halal_india.filter_symbols_by_min_history",
            side_effect=_mock_india_history_filter_pass_all(ns_symbols),
        ),
        patch(
            "brain_api.universe.halal_india.run_batch_inference",
            return_value=mock_result,
        ),
    ):
        response = client.get("/universe/halal_india")

    assert response.status_code == 200
    data = response.json()
    assert len(data["stocks"]) == 15
    assert data["total_candidates"] == 20


def test_get_halal_india_old_cache_shape_loads():
    """Old-shape cached file (no rank-band fields) must still load with .NS preserved.

    Pre-PR cache files have ``selection_method='patchtst_forecast'``,
    per-stock ``model_version``, and lack rank-band fields. Downstream
    consumers (``get_halal_india_symbols``, LLM/email handlers) read
    ``stocks[*].symbol`` and the top-level ``model_version`` /
    ``selection_method`` only -- those keep the same shape after this
    PR, so old caches must not break consumers.
    """
    from datetime import date

    from brain_api.universe.cache import load_cached_universe, save_universe_cache
    from brain_api.universe.halal_india import get_halal_india_symbols

    legacy = {
        "stocks": [
            {
                "symbol": "RELIANCE.NS",
                "predicted_weekly_return_pct": 5.0,
                "rank": 1,
                "model_version": "v2026-03-01-india123",
            },
            {
                "symbol": "TCS.NS",
                "predicted_weekly_return_pct": 4.0,
                "rank": 2,
                "model_version": "v2026-03-01-india123",
            },
        ],
        "total_candidates": 2,
        "total_universe": 200,
        "filtered_insufficient_history": 0,
        "top_n": 15,
        "selection_method": "patchtst_forecast",
        "model_version": "v2026-03-01-india123",
        "symbol_suffix": ".NS",
        "fetched_at": "2026-04-01T00:00:00+00:00",
    }
    today = date.today()
    save_universe_cache("halal_india", legacy, today)

    loaded = load_cached_universe("halal_india", today)
    assert loaded is not None
    assert [s["symbol"] for s in loaded["stocks"]] == ["RELIANCE.NS", "TCS.NS"]
    assert loaded["selection_method"] == "patchtst_forecast"

    # get_halal_india_symbols must still extract symbols from old shape via cache hit.
    symbols = get_halal_india_symbols()
    assert symbols == ["RELIANCE.NS", "TCS.NS"]


def test_halal_india_partition_isolated_from_weekly_alpha_hrp():
    """Cross-table isolation: weekly halal_india_alpha rows must not leak into monthly halal_india_filtered_alpha reads.

    Weekly India Alpha-HRP writes to ``stage1_weight_history`` under
    ``universe='halal_india_alpha'``. The monthly halal_india builder
    reads from ``screening_history`` under
    ``partition='halal_india_filtered_alpha'``. Even if the two share
    a period_key value, the reads must not cross tables.
    """
    from brain_api.core.screening_orchestration import persist_screening_rows
    from brain_api.core.strategy_partitions import (
        HALAL_INDIA_ALPHA_PARTITION,
        HALAL_INDIA_FILTERED_ALPHA_PARTITION,
    )
    from brain_api.storage.screening_history import ScreeningHistoryRepository
    from brain_api.storage.sticky_history import StickyHistoryRepository, WeightRow

    period_key = "202615"
    sticky_repo = StickyHistoryRepository()
    sticky_repo.persist_stage1(
        [
            WeightRow(
                universe=HALAL_INDIA_ALPHA_PARTITION,
                year_week=period_key,
                as_of_date="2026-04-06",
                stock="RELIANCE.NS",
                stage1_rank=1,
                initial_allocation_pct=None,
                signal_score=5.0,
                final_allocation_pct=10.0,
                selected_in_final=True,
                selection_reason="top_rank",
                run_id="weekly-test",
            )
        ]
    )

    screening_repo = ScreeningHistoryRepository()
    persist_screening_rows(
        repo=screening_repo,
        partition=HALAL_INDIA_FILTERED_ALPHA_PARTITION,
        period_key=period_key,
        as_of_date="2026-04-06",
        run_id="monthly-test",
        scores={"INFY.NS": 7.0, "TCS.NS": 6.0},
        selected_set={"INFY.NS"},
        selection_reasons={"INFY.NS": "top_rank"},
    )

    # Reading the screening_history at a later period_key must NOT see RELIANCE.NS
    # (which only lives in stage1_weight_history).
    prev = screening_repo.read_previous_selected_set(
        partition=HALAL_INDIA_FILTERED_ALPHA_PARTITION,
        current_period_key="202620",
    )
    assert prev is not None
    assert prev.selected_set == {"INFY.NS"}
    assert "RELIANCE.NS" not in prev.selected_set

    # Reading sticky two-stage at the same period must NOT see the screening row.
    sticky_prev = sticky_repo.read_previous_final_set(
        universe=HALAL_INDIA_ALPHA_PARTITION,
        current_year_week="202620",
    )
    assert sticky_prev is not None
    assert sticky_prev.final_set == {"RELIANCE.NS"}
    assert "INFY.NS" not in sticky_prev.final_set


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


# ============================================================================
# NiftyShariah500 Universe Endpoint Tests
# ============================================================================


def test_get_nifty_shariah_500_returns_expected_structure():
    """Test that /universe/nifty_shariah_500 returns the expected response structure."""
    with patch(
        "brain_api.universe.nifty_shariah_500.scrape_nifty500_shariah",
        return_value=MOCK_NSE_CONSTITUENTS,
    ):
        response = client.get("/universe/nifty_shariah_500")

    assert response.status_code == 200
    data = response.json()

    assert "stocks" in data
    assert "source" in data
    assert "symbol_suffix" in data
    assert "total_stocks" in data
    assert "fetched_at" in data

    assert data["source"] == "nifty_500_shariah"
    assert data["symbol_suffix"] == ".NS"


def test_get_nifty_shariah_500_symbols_have_ns_suffix():
    """Test that all NiftyShariah500 symbols include .NS suffix."""
    with patch(
        "brain_api.universe.nifty_shariah_500.scrape_nifty500_shariah",
        return_value=MOCK_NSE_CONSTITUENTS,
    ):
        response = client.get("/universe/nifty_shariah_500")

    assert response.status_code == 200
    data = response.json()

    for stock in data["stocks"]:
        assert stock["symbol"].endswith(".NS"), (
            f"Symbol {stock['symbol']} should have .NS suffix"
        )


def test_get_nifty_shariah_500_returns_all_constituents():
    """Test that /universe/nifty_shariah_500 returns ALL constituents (not capped at 15)."""
    with patch(
        "brain_api.universe.nifty_shariah_500.scrape_nifty500_shariah",
        return_value=MOCK_NSE_CONSTITUENTS,
    ):
        response = client.get("/universe/nifty_shariah_500")

    assert response.status_code == 200
    data = response.json()

    assert len(data["stocks"]) == len(MOCK_NSE_CONSTITUENTS)
    assert data["total_stocks"] == len(MOCK_NSE_CONSTITUENTS)


def test_get_nifty_shariah_500_returns_503_on_nse_failure():
    """Test that /universe/nifty_shariah_500 returns 503 when NSE scraper fails."""
    from brain_api.universe.scrapers.nse import NseFetchError

    with patch(
        "brain_api.universe.nifty_shariah_500.scrape_nifty500_shariah",
        side_effect=NseFetchError("NSE API returned empty data"),
    ):
        response = client.get("/universe/nifty_shariah_500")

    assert response.status_code == 503
    assert "NSE API" in response.json()["detail"]
