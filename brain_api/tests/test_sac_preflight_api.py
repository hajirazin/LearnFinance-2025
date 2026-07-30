from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from brain_api.core.fundamentals import FundamentalsCacheError
from brain_api.core.sac.readiness import SACReadinessIssue, SACTrainingReadiness
from brain_api.main import app
from brain_api.routes.training.sac import preflight as preflight_module

client = TestClient(app)


@pytest.fixture
def strict_preflight_dependencies(monkeypatch):
    """Provide complete inputs so each test can break one readiness contract."""
    symbols = ["AAA"]
    monkeypatch.setattr(
        preflight_module,
        "get_bucket",
        lambda model_type, universe: SimpleNamespace(
            symbols_resolver=lambda: symbols,
        ),
    )
    monkeypatch.setattr(
        preflight_module,
        "resolve_training_window",
        lambda: (date(2024, 1, 1), date(2024, 2, 1)),
    )
    monkeypatch.setattr(
        preflight_module,
        "SnapshotLocalStorage",
        lambda forecaster_type: object(),
    )
    monkeypatch.setattr(
        preflight_module,
        "ensure_snapshot_for_bucket",
        lambda **kwargs: True,
    )

    price_index = pd.date_range("2023-12-20", "2024-02-01", freq="D")
    prices = {
        "AAA": pd.DataFrame(
            {
                "open": 100.0,
                "close": 100.0,
            },
            index=price_index,
        )
    }
    monkeypatch.setattr(
        preflight_module,
        "load_prices_yfinance",
        lambda requested_symbols, start_date, end_date: prices,
    )

    news_index = pd.date_range("2023-12-20", "2024-02-01", freq="D")
    news = {
        "AAA": pd.DataFrame(
            {
                "sentiment_score": 0.0,
                "article_count": 0,
                "avg_confidence": 0.0,
            },
            index=news_index,
        )
    }
    monkeypatch.setattr(
        preflight_module,
        "load_historical_news_sentiment",
        lambda requested_symbols, start_date, end_date: news,
    )

    fundamentals = {
        "AAA": pd.DataFrame(
            {
                "gross_margin": [0.5],
                "operating_margin": [0.3],
                "net_margin": [0.2],
                "current_ratio": [1.5],
                "debt_to_equity": [0.4],
            },
            index=pd.DatetimeIndex(["2023-01-01"]),
        )
    }
    monkeypatch.setattr(
        preflight_module,
        "load_historical_fundamentals_from_cache",
        lambda requested_symbols, start_date, end_date: fundamentals,
    )
    return prices, fundamentals


def test_sac_preflight_returns_exact_missing_and_errors(monkeypatch):
    readiness = SACTrainingReadiness.from_issues(
        universe="halal_filtered",
        symbols=["AAA"],
        missing=[
            SACReadinessIssue(
                source="fundamentals",
                symbol="AAA",
                detail="filing availability unresolved",
                retryable=True,
            )
        ],
        errors=[
            SACReadinessIssue(
                source="news",
                symbol="AAA",
                detail="provider error",
                retryable=True,
            )
        ],
    )
    monkeypatch.setattr(
        "brain_api.routes.training.sac.preflight.assess_sac_training_readiness",
        lambda universe, *, force=False: readiness,
    )

    response = client.post("/train/sac/preflight", json={"universe": "halal_filtered"})

    assert response.status_code == 200
    assert response.json() == {
        "universe": "halal_filtered",
        "symbols": ["AAA"],
        "ready": False,
        "missing": [
            {
                "source": "fundamentals",
                "detail": "filing availability unresolved",
                "symbol": "AAA",
                "retryable": True,
            }
        ],
        "errors": [
            {
                "source": "news",
                "detail": "provider error",
                "symbol": "AAA",
                "retryable": True,
            }
        ],
    }


def test_sac_preflight_rejects_unknown_universe_at_api():
    response = client.post("/train/sac/preflight", json={"universe": "unknown"})

    assert response.status_code == 422


def test_sac_preflight_reports_missing_price_history(
    monkeypatch,
    strict_preflight_dependencies,
):
    monkeypatch.setattr(
        preflight_module,
        "load_prices_yfinance",
        lambda requested_symbols, start_date, end_date: {},
    )

    response = client.post(
        "/train/sac/preflight",
        json={"universe": "halal_filtered", "force": True},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ready"] is False
    assert payload["missing"] == [
        {
            "source": "prices",
            "detail": "Missing daily price history for AAA",
            "symbol": "AAA",
            "retryable": True,
        }
    ]


def test_sac_preflight_rejects_filing_history_that_starts_after_first_cutoff(
    monkeypatch,
    strict_preflight_dependencies,
):
    _, fundamentals = strict_preflight_dependencies
    fundamentals["AAA"].index = pd.DatetimeIndex(["2024-01-15"])

    response = client.post(
        "/train/sac/preflight",
        json={"universe": "halal_filtered", "force": True},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ready"] is False
    assert payload["missing"] == [
        {
            "source": "fundamentals",
            "detail": (
                "No filing was available before every SAC training cutoff for AAA"
            ),
            "symbol": "AAA",
            "retryable": True,
        }
    ]


def test_sac_preflight_reports_corrupt_fundamentals_cache_as_non_retryable_error(
    monkeypatch,
    strict_preflight_dependencies,
):
    def raise_corrupt_cache(requested_symbols, start_date, end_date):
        raise FundamentalsCacheError(
            "Malformed fundamentals cache for AAA: invalid JSON"
        )

    monkeypatch.setattr(
        preflight_module,
        "load_historical_fundamentals_from_cache",
        raise_corrupt_cache,
    )

    response = client.post(
        "/train/sac/preflight",
        json={"universe": "halal_filtered", "force": True},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ready"] is False
    assert payload["errors"] == [
        {
            "source": "fundamentals",
            "detail": "Malformed fundamentals cache for AAA: invalid JSON",
            "symbol": "AAA",
            "retryable": False,
        }
    ]
