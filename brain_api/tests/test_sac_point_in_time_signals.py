"""Deterministic correctness tests for point-in-time SAC signal inputs."""

import json
import math
from datetime import date

import pytest

from brain_api.core.fundamentals.fetcher import FundamentalsFetcher
from brain_api.core.fundamentals.sec_filings import (
    SECFilingAvailability,
    enrich_statement_periods_with_filing_availability,
)
from brain_api.core.fundamentals.storage import load_raw_response, save_raw_response
from brain_api.core.news_sentiment.fetcher import NewsProviderError, YFinanceNewsFetcher
from brain_api.core.news_sentiment.weekly import (
    DailyNewsObservation,
    NewsObservationError,
    aggregate_weekly_news_observation,
)


def test_sec_enrichment_resolves_only_exact_report_dates():
    payload = {
        "quarterlyReports": [
            {"fiscalDateEnding": "2026-03-31"},
            {"fiscalDateEnding": "2025-12-31"},
        ]
    }
    filing = SECFilingAvailability(
        report_date="2026-03-31",
        filing_date="2026-05-04",
        accession_number="0001-26-000001",
        form="10-Q",
        source="https://www.sec.gov/example",
    )

    enriched = enrich_statement_periods_with_filing_availability(payload, [filing])

    resolved, unresolved = enriched["quarterlyReports"]
    assert resolved["filingDate"] == "2026-05-04"
    assert resolved["accessionNumber"] == filing.accession_number
    assert resolved["filingSource"] == filing.source
    assert "filingDate" not in unresolved


def test_weekly_news_uses_article_confidence_and_recency_weights():
    as_of = date(2026, 5, 4)
    rows = [
        DailyNewsObservation(date(2026, 5, 4), 0.8, 2, 0.5),
        DailyNewsObservation(date(2026, 4, 27), -0.4, 1, 1.0),
    ]

    result = aggregate_weekly_news_observation(rows, as_of)

    recent_weight = 2 * 0.5
    old_weight = math.exp(-1)
    expected = (0.8 * recent_weight - 0.4 * old_weight) / (recent_weight + old_weight)
    assert result.sentiment_score == pytest.approx(expected)
    assert result.coverage == 1.0
    assert result.article_count == 3
    assert not result.confirmed_no_articles


def test_confirmed_zero_news_is_neutral_but_missing_window_fails():
    as_of = date(2026, 5, 4)
    result = aggregate_weekly_news_observation(
        [DailyNewsObservation(as_of, 0.0, 0, 0.0)],
        as_of,
    )

    assert result.sentiment_score == 0.0
    assert result.coverage == 0.0
    assert result.confirmed_no_articles
    with pytest.raises(NewsObservationError, match="No provider-checked"):
        aggregate_weekly_news_observation([], as_of)


def test_news_provider_error_is_not_converted_to_confirmed_zero(monkeypatch):
    class BrokenTicker:
        @property
        def news(self):
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr(
        "brain_api.core.news_sentiment.fetcher.yf.Ticker",
        lambda _symbol: BrokenTicker(),
    )

    with pytest.raises(NewsProviderError, match="provider unavailable"):
        YFinanceNewsFetcher().fetch("AAPL", 10)


def test_legacy_nested_fundamentals_cache_migrates_without_data_loss(tmp_path):
    legacy = (
        tmp_path
        / "raw"
        / "fundamentals"
        / "raw"
        / "fundamentals"
        / "AAPL"
        / "income_statement.json"
    )
    legacy.parent.mkdir(parents=True)
    expected = {"response": {"quarterlyReports": [{"fiscalDateEnding": "2026-03-31"}]}}
    legacy.write_text(json.dumps(expected))

    loaded = load_raw_response(tmp_path, "AAPL", "income_statement")

    canonical = tmp_path / "raw" / "fundamentals" / "AAPL" / "income_statement.json"
    assert loaded == expected
    assert json.loads(canonical.read_text()) == expected
    assert legacy.exists()


def test_cached_alpha_vantage_periods_are_sec_enriched_before_use(tmp_path):
    payload = {
        "quarterlyReports": [
            {
                "fiscalDateEnding": "2026-03-31",
                "totalRevenue": "100",
            }
        ]
    }
    save_raw_response(tmp_path, "AAPL", "income_statement", payload)
    save_raw_response(tmp_path, "AAPL", "balance_sheet", payload)
    filing = SECFilingAvailability(
        report_date="2026-03-31",
        filing_date="2026-05-04",
        accession_number="0001-26-000001",
        form="10-Q",
        source="https://www.sec.gov/example",
    )

    class Provider:
        def __init__(self):
            self.calls = 0

        def fetch_symbol_filings(self, symbol):
            assert symbol == "AAPL"
            self.calls += 1
            return [filing]

    provider = Provider()
    fetcher = FundamentalsFetcher(
        api_key="unused",
        base_path=tmp_path,
        filing_provider=provider,
    )
    try:
        result = fetcher.fetch_symbol("AAPL")
    finally:
        fetcher.close()

    assert result.from_cache
    assert provider.calls == 1
    persisted = load_raw_response(tmp_path, "AAPL", "income_statement")
    report = persisted["response"]["quarterlyReports"][0]
    assert report["filingDate"] == "2026-05-04"
    assert report["accessionNumber"] == filing.accession_number
    assert report["filingSource"] == filing.source


def test_unresolved_cached_periods_fail_without_sec_provider(tmp_path, monkeypatch):
    payload = {"quarterlyReports": [{"fiscalDateEnding": "2026-03-31"}]}
    save_raw_response(tmp_path, "AAPL", "income_statement", payload)
    save_raw_response(tmp_path, "AAPL", "balance_sheet", payload)
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    fetcher = FundamentalsFetcher(api_key="unused", base_path=tmp_path)
    try:
        with pytest.raises(RuntimeError, match="SEC filing availability is required"):
            fetcher.fetch_symbol("AAPL")
    finally:
        fetcher.close()
