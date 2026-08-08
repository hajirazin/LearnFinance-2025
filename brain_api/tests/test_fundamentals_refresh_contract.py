"""Operational contract tests for point-in-time fundamentals refresh."""

from datetime import date
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from brain_api.core.data_freshness import (
    DataFreshnessResult,
    refresh_stale_fundamentals,
)
from brain_api.core.fundamentals.refresh_policy import RefreshAction
from brain_api.core.fundamentals.storage import save_raw_response
from brain_api.main import app

client = TestClient(app)


def test_refresh_reenriches_unprovenanced_cache(tmp_path, monkeypatch):
    unresolved = {"quarterlyReports": [{"fiscalDateEnding": "2020-03-31"}]}
    save_raw_response(tmp_path, "AAPL", "income_statement", unresolved)
    save_raw_response(tmp_path, "AAPL", "balance_sheet", unresolved)
    monkeypatch.setenv("SEC_USER_AGENT", "LearnFinance test@example.com")
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test-key")

    fetcher = MagicMock()
    fetcher.decide_action_for_symbol.return_value = RefreshAction.ENRICH_ONLY
    fetcher.eligibility_client = MagicMock()
    eligibility = MagicMock()
    eligibility.sec_eligible = False
    eligibility.cik = None
    fetcher.eligibility_client.classify.return_value = eligibility
    fetcher._pending_new_filing = set()
    fetcher.get_api_status.return_value = {
        "calls_today": 0,
        "daily_limit": 25,
        "remaining": 25,
    }
    with patch(
        "brain_api.core.data_freshness.FundamentalsFetcher",
        return_value=fetcher,
    ):
        result = refresh_stale_fundamentals(["AAPL"], base_path=tmp_path)

    fetcher.fetch_symbol.assert_called_once_with("AAPL", force_refresh=False)
    assert result.refreshed == ["AAPL"]
    assert result.skipped == []
    assert result.failed == []


def test_refresh_marks_empty_provider_result_as_failed(tmp_path, monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "LearnFinance test@example.com")
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test-key")
    fetcher = MagicMock()
    fetcher.decide_action_for_symbol.return_value = RefreshAction.PULL
    fetcher.eligibility_client = MagicMock()
    eligibility = MagicMock()
    eligibility.sec_eligible = False
    eligibility.cik = None
    fetcher.eligibility_client.classify.return_value = eligibility
    fetcher._pending_new_filing = set()
    fetcher.fetch_symbol.side_effect = RuntimeError(
        "Alpha Vantage returned no usable quarterly income statement for AAPL"
    )
    fetcher.get_api_status.return_value = {}
    with patch(
        "brain_api.core.data_freshness.FundamentalsFetcher",
        return_value=fetcher,
    ):
        result = refresh_stale_fundamentals(["AAPL"], base_path=tmp_path)

    assert result.refreshed == []
    assert result.failed == ["AAPL"]
    assert result.errors == {
        "AAPL": "Alpha Vantage returned no usable quarterly income statement for AAPL"
    }


def test_refresh_distinguishes_corrupt_cache_from_missing_enrichment(
    tmp_path,
    monkeypatch,
):
    cache_dir = tmp_path / "raw" / "fundamentals" / "AAPL"
    cache_dir.mkdir(parents=True)
    (cache_dir / "income_statement.json").write_text("{invalid json")
    monkeypatch.setenv("SEC_USER_AGENT", "LearnFinance test@example.com")
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test-key")

    result = refresh_stale_fundamentals(["AAPL"], base_path=tmp_path)

    assert result.failed == ["AAPL"]
    assert result.errors["AAPL"]


def test_refresh_training_data_api_returns_503_for_failed_fundamentals(monkeypatch):
    monkeypatch.setattr(
        "brain_api.routes.etl_training_refresh.get_etl_symbols",
        lambda universe: ["AAPL"],
    )
    monkeypatch.setattr(
        "brain_api.routes.etl_training_refresh.ensure_fresh_training_data",
        lambda **kwargs: DataFreshnessResult(
            fundamentals_failed=["AAPL"],
            fundamentals_errors={
                "AAPL": ("Alpha Vantage returned no usable quarterly income statement")
            },
            duration_seconds=1.0,
        ),
    )

    response = client.post(
        "/etl/refresh-training-data",
        json={
            "universe": "halal_filtered",
            "start_date": date(2024, 1, 1).isoformat(),
            "end_date": date(2024, 12, 31).isoformat(),
        },
    )

    assert response.status_code == 503
    assert response.json()["detail"] == {
        "source": "fundamentals",
        "failed_symbols": ["AAPL"],
        "errors": {
            "AAPL": "Alpha Vantage returned no usable quarterly income statement"
        },
        "message": "Fundamentals refresh did not produce usable enriched data",
    }
