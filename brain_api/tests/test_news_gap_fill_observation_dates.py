"""Regression tests for provider-checked news observation dates."""

from datetime import UTC, date, datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from brain_api.core.finbert import SentimentScore
from brain_api.core.news_api.alpaca import (
    AlpacaNewsArticle,
    AlpacaNewsClient,
    AlpacaNewsProviderError,
)
from brain_api.etl.gap_detection import find_gaps
from brain_api.etl.gap_fill import fill_sentiment_gaps


def _article(
    *,
    article_id: str,
    symbol: str,
    created_at: datetime,
) -> AlpacaNewsArticle:
    return AlpacaNewsArticle(
        id=article_id,
        headline=f"{symbol} headline",
        summary=f"{symbol} summary",
        author="Reporter",
        created_at=created_at,
        updated_at=created_at,
        url=f"https://example.com/{article_id}",
        symbols=[symbol],
        source="test",
    )


def _positive_score() -> SentimentScore:
    return SentimentScore(
        label="positive",
        p_pos=0.8,
        p_neg=0.1,
        p_neu=0.1,
        score=0.7,
        confidence=0.8,
    )


def _run_gap_fill(
    *,
    parquet_path,
    symbols: list[str],
    start_date: date,
    end_date: date,
    articles: list[AlpacaNewsArticle] | None = None,
    provider_error: Exception | None = None,
):
    client = MagicMock()
    client.call_count = 1
    if provider_error is not None:
        client.fetch_news_for_date.side_effect = provider_error
    else:
        client.fetch_news_for_date.return_value = articles or []

    scorer = MagicMock()
    scorer.score_batch.side_effect = lambda texts: [_positive_score() for _ in texts]

    with (
        patch("brain_api.etl.gap_fill.get_etl_symbols", return_value=symbols),
        patch("brain_api.etl.gap_fill.AlpacaNewsClient", return_value=client),
        patch("brain_api.etl.gap_fill.FinBERTScorer", return_value=scorer),
    ):
        result = fill_sentiment_gaps(
            universe="halal",
            start_date=start_date,
            end_date=end_date,
            parquet_path=parquet_path,
            local_only=True,
        )

    return result, scorer


def test_off_date_article_does_not_fill_requested_observation_date(tmp_path):
    gap_date = date(2023, 12, 30)
    parquet_path = tmp_path / "daily_sentiment.parquet"
    off_date_article = _article(
        article_id="off-date",
        symbol="PANW",
        created_at=datetime(2023, 12, 29, 23, 59, tzinfo=UTC),
    )

    result, scorer = _run_gap_fill(
        parquet_path=parquet_path,
        symbols=["PANW"],
        start_date=gap_date,
        end_date=gap_date,
        articles=[off_date_article],
    )

    assert result.success
    scorer.score_batch.assert_not_called()
    row = pd.read_parquet(parquet_path).iloc[0]
    assert row["date"] == gap_date
    assert row["symbol"] == "PANW"
    assert row["sentiment_score"] == 0.0
    assert row["article_count"] == 0
    assert find_gaps(["PANW"], gap_date, gap_date, parquet_path) == []


def test_only_on_date_articles_mark_symbols_as_matched(tmp_path):
    gap_date = date(2023, 12, 30)
    parquet_path = tmp_path / "daily_sentiment.parquet"
    articles = [
        _article(
            article_id="on-date",
            symbol="PANW",
            created_at=datetime(2023, 12, 30, 8, 0),
        ),
        _article(
            article_id="off-date",
            symbol="AAPL",
            created_at=datetime(2023, 12, 29, 23, 59, tzinfo=UTC),
        ),
    ]

    result, scorer = _run_gap_fill(
        parquet_path=parquet_path,
        symbols=["PANW", "AAPL"],
        start_date=gap_date,
        end_date=gap_date,
        articles=articles,
    )

    assert result.success
    scorer.score_batch.assert_called_once()
    assert len(scorer.score_batch.call_args.args[0]) == 1

    rows = pd.read_parquet(parquet_path).set_index("symbol")
    assert set(rows.index) == {"PANW", "AAPL"}
    assert set(rows["date"]) == {gap_date}
    assert rows.loc["PANW", "article_count"] == 1
    assert rows.loc["AAPL", "article_count"] == 0


def test_empty_successful_provider_response_creates_confirmed_zero_rows(tmp_path):
    historical_date = date(2023, 12, 30)
    current_date = datetime.now(UTC).date()
    parquet_path = tmp_path / "daily_sentiment.parquet"

    with (
        patch(
            "brain_api.etl.gap_fill.find_gaps",
            side_effect=[
                [
                    (historical_date, "PANW"),
                    (historical_date, "AAPL"),
                    (current_date, "PANW"),
                    (current_date, "AAPL"),
                ],
                [(current_date, "PANW"), (current_date, "AAPL")],
            ],
        ),
        patch(
            "brain_api.etl.gap_fill.get_gap_statistics",
            return_value={"gaps_found": 2},
        ),
    ):
        result, scorer = _run_gap_fill(
            parquet_path=parquet_path,
            symbols=["PANW", "AAPL"],
            start_date=historical_date,
            end_date=current_date,
            articles=[],
        )

    assert result.success
    scorer.score_batch.assert_not_called()
    rows = pd.read_parquet(parquet_path)
    assert set(zip(rows["date"], rows["symbol"], strict=True)) == {
        (historical_date, "PANW"),
        (historical_date, "AAPL"),
    }
    assert (rows["sentiment_score"] == 0.0).all()
    assert (rows["article_count"] == 0).all()
    assert (rows["avg_confidence"] == 0.0).all()
    assert (rows["p_pos_avg"] == 0.0).all()
    assert (rows["p_neg_avg"] == 0.0).all()
    assert (rows["total_articles"] == 0).all()


def test_provider_failure_does_not_create_zero_rows(tmp_path):
    gap_date = date(2023, 12, 30)
    parquet_path = tmp_path / "daily_sentiment.parquet"

    result, scorer = _run_gap_fill(
        parquet_path=parquet_path,
        symbols=["PANW"],
        start_date=gap_date,
        end_date=gap_date,
        provider_error=AlpacaNewsProviderError("authentication failed"),
    )

    assert not result.success
    assert result.progress.status == "failed"
    assert result.progress.error == "authentication failed"
    assert not parquet_path.exists()
    scorer.score_batch.assert_not_called()


def test_missing_provider_credentials_raise_instead_of_returning_empty():
    with (
        patch("brain_api.core.news_api.alpaca.get_alpaca_api_key", return_value=""),
        patch("brain_api.core.news_api.alpaca.get_alpaca_api_secret", return_value=""),
    ):
        client = AlpacaNewsClient(rate_limit_delay=0)

    with pytest.raises(AlpacaNewsProviderError, match="credentials"):
        client.fetch_news_for_date(["PANW"], date(2023, 12, 30))


@pytest.mark.parametrize(
    ("response_setup", "expected_message"),
    [
        (
            lambda response: setattr(
                response.raise_for_status,
                "side_effect",
                requests.HTTPError("unauthorized"),
            ),
            "request failed",
        ),
        (
            lambda response: setattr(
                response.json,
                "side_effect",
                ValueError("not json"),
            ),
            "invalid JSON",
        ),
        (
            lambda response: setattr(
                response.json,
                "return_value",
                {"error": "unusable"},
            ),
            "usable news collection",
        ),
    ],
)
def test_provider_failures_raise_instead_of_returning_empty(
    response_setup,
    expected_message,
):
    response = MagicMock()
    response_setup(response)
    client = AlpacaNewsClient(
        api_key="key",
        api_secret="secret",
        rate_limit_delay=0,
    )

    with (
        patch("brain_api.core.news_api.alpaca.requests.get", return_value=response),
        pytest.raises(AlpacaNewsProviderError, match=expected_message),
    ):
        client.fetch_news_for_date(["PANW"], date(2023, 12, 30))


def test_genuinely_empty_news_collection_returns_empty():
    response = MagicMock()
    response.json.return_value = {"news": []}
    client = AlpacaNewsClient(
        api_key="key",
        api_secret="secret",
        rate_limit_delay=0,
    )

    with patch(
        "brain_api.core.news_api.alpaca.requests.get",
        return_value=response,
    ):
        assert client.fetch_news_for_date(["PANW"], date(2023, 12, 30)) == []
