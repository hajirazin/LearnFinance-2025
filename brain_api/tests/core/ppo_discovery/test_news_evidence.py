"""News evidence tests: exhaust pagination, abort rules, verified zero."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from brain_api.core.finbert import SentimentScore
from brain_api.core.news_api.alpaca import (
    AlpacaNewsArticle,
    AlpacaNewsClient,
    AlpacaNewsProviderError,
)
from brain_api.core.ppo_discovery.news_evidence import (
    NewsEvidenceError,
    article_dedupe_key,
    assign_articles_to_symbols,
    fetch_news_exhaustive,
    materialize_news_evidence,
    news_window_for_cutoff,
    persist_weekly_news_features,
    score_texts_or_abort,
)
from tests.core.ppo_discovery.factories import make_news


def _article(
    *,
    article_id: str,
    created: datetime,
    symbol: str = "AAPL",
    headline: str = "Hello",
    url: str = "https://example.com/a",
) -> AlpacaNewsArticle:
    return AlpacaNewsArticle(
        id=article_id,
        headline=headline,
        summary="summary",
        author="desk",
        created_at=created,
        updated_at=created,
        url=url,
        symbols=[symbol],
        source="benzinga",
    )


def test_window_is_half_open_previous_to_current() -> None:
    end = datetime(2026, 8, 31, 13, 0, tzinfo=UTC)
    start = datetime(2026, 8, 24, 13, 0, tzinfo=UTC)
    window = news_window_for_cutoff(end, start)
    assert window.start == start
    assert window.end == end
    cold = news_window_for_cutoff(end, None)
    assert cold.start == end - timedelta(days=7)


def test_pagination_exhausts_until_token_absent() -> None:
    cutoff = datetime(2026, 8, 31, tzinfo=UTC)
    window = news_window_for_cutoff(cutoff, None)
    pages = {
        None: ([_article(article_id="1", created=cutoff, headline="one")], "tok"),
        "tok": ([_article(article_id="2", created=cutoff, headline="two")], None),
    }

    def fetch_page(**kwargs):
        return pages[kwargs["page_token"]]

    articles, manifest = fetch_news_exhaustive(
        MagicMock(), ["AAPL"], window, fetch_page=fetch_page
    )
    assert len(articles) == 2
    assert manifest["pages"] == 2


def test_provider_error_and_429_abort() -> None:
    cutoff = datetime(2026, 8, 31, tzinfo=UTC)
    window = news_window_for_cutoff(cutoff, None)

    def boom(**kwargs):
        raise AlpacaNewsProviderError("429")

    with pytest.raises(NewsEvidenceError, match="provider failure"):
        fetch_news_exhaustive(MagicMock(), ["AAPL"], window, fetch_page=boom)


def test_alpaca_page_retries_429_then_succeeds() -> None:
    cutoff = datetime(2026, 8, 31, tzinfo=UTC)
    window = news_window_for_cutoff(cutoff, None)
    client = AlpacaNewsClient(api_key="k", api_secret="s", rate_limit_delay=0)
    busy = MagicMock(status_code=429)
    ok = MagicMock(status_code=200)
    ok.json.return_value = {"news": [], "next_page_token": None}
    ok.raise_for_status = MagicMock()
    with (
        patch("brain_api.core.news_api.alpaca.requests.get", side_effect=[busy, ok]),
        patch("brain_api.core.news_api.alpaca.time.sleep"),
    ):
        articles, token = client.fetch_news_page(
            symbols=["AAPL"], start=window.start, end=window.end
        )
    assert articles == []
    assert token is None


def test_page_cap_with_remaining_token_aborts() -> None:
    cutoff = datetime(2026, 8, 31, tzinfo=UTC)
    window = news_window_for_cutoff(cutoff, None)
    article = _article(article_id="1", created=cutoff)

    def fetch_page(**kwargs):
        return [article], "still-more"

    with pytest.raises(NewsEvidenceError, match="article cap"):
        fetch_news_exhaustive(
            MagicMock(), ["AAPL"], window, article_cap=1, fetch_page=fetch_page
        )


def test_dedupe_id_then_url_created_headline() -> None:
    created = datetime(2026, 8, 30, tzinfo=UTC)
    a = _article(article_id="x", created=created)
    b = _article(article_id="x", created=created, headline="other")
    c = _article(article_id="", created=created, url="https://u", headline="h")
    d = _article(article_id="", created=created, url="https://u", headline="h")
    assert article_dedupe_key(a) == article_dedupe_key(b)
    assigned = assign_articles_to_symbols([a, b, c, d], ["AAPL"])
    assert len(assigned["AAPL"]) == 2


def test_missing_created_at_aborts() -> None:
    cutoff = datetime(2026, 8, 31, tzinfo=UTC)
    article = _article(article_id="1", created=cutoff)
    article.created_at = None  # type: ignore[assignment]

    def fetch_page(**kwargs):
        return [article], None

    with pytest.raises(NewsEvidenceError, match="created_at"):
        materialize_news_evidence(
            ["AAPL"],
            cutoff,
            fetch_page=fetch_page,
            score_fn=lambda texts: [
                SentimentScore("neutral", 0.0, 0.0, 1.0, 0.0, 1.0) for _ in texts
            ],
        )


def test_verified_zero_news_is_valid() -> None:
    cutoff = datetime(2026, 8, 31, tzinfo=UTC)

    def fetch_page(**kwargs):
        return [], None

    features = materialize_news_evidence(
        ["AAPL"], cutoff, fetch_page=fetch_page, score_fn=lambda texts: []
    )
    row = features["AAPL"]
    assert row.query_complete is True
    assert row.raw_sentiment == 0.0
    assert row.article_count == 0
    assert row.news_recency == 0.0
    assert row.log1p_article_count == 0.0
    assert row.has_news == 0


def test_finbert_exception_aborts() -> None:
    scorer = MagicMock()
    scorer._inference_lock = MagicMock()
    scorer._inference_lock.__enter__.return_value = None
    scorer._inference_lock.__exit__.return_value = False
    scorer._ensure_loaded = MagicMock()
    scorer._pipeline = MagicMock(side_effect=RuntimeError("cuda boom"))
    with pytest.raises(NewsEvidenceError, match="FinBERT scoring failed"):
        score_texts_or_abort(scorer, ["hello world"])


def test_persist_cutoff_is_idempotent_without_force(tmp_path) -> None:
    cutoff = datetime(2026, 8, 31, tzinfo=UTC)

    def fetch_page(**kwargs):
        return [], None

    features = materialize_news_evidence(
        ["AAPL"], cutoff, fetch_page=fetch_page, score_fn=lambda texts: []
    )
    window = news_window_for_cutoff(cutoff, None)
    persist_weekly_news_features(
        cutoff, features, window=window, base_path=tmp_path, force=False
    )
    persist_weekly_news_features(
        cutoff, features, window=window, base_path=tmp_path, force=False
    )
    import pyarrow.parquet as pq

    table = pq.read_table(
        tmp_path / "ppo_discovery" / "news" / "weekly_features.parquet"
    )
    assert table.num_rows == 1
    persist_weekly_news_features(
        cutoff, features, window=window, base_path=tmp_path, force=True
    )
    table = pq.read_table(
        tmp_path / "ppo_discovery" / "news" / "weekly_features.parquet"
    )
    assert table.num_rows == 1


def test_yfinance_is_not_called(tmp_path) -> None:
    cutoff = datetime(2026, 8, 31, tzinfo=UTC)

    def fetch_page(**kwargs):
        return [], None

    with patch("yfinance.Ticker") as ticker:
        features = materialize_news_evidence(
            ["AAPL"], cutoff, fetch_page=fetch_page, score_fn=lambda texts: []
        )
        persist_weekly_news_features(
            cutoff,
            features,
            window=news_window_for_cutoff(cutoff, None),
            base_path=tmp_path,
        )
        ticker.assert_not_called()
    assert (tmp_path / "ppo_discovery" / "news" / "weekly_features.parquet").exists()


def test_window_excludes_start_includes_end() -> None:
    cutoff = datetime(2026, 8, 31, 13, 0, tzinfo=UTC)
    prev = datetime(2026, 8, 24, 13, 0, tzinfo=UTC)
    inside = _article(article_id="in", created=cutoff, headline="in")
    boundary_start = _article(article_id="start", created=prev, headline="start")
    too_late = _article(
        article_id="late",
        created=cutoff + timedelta(seconds=1),
        headline="late",
    )

    def fetch_page(**kwargs):
        return [inside, boundary_start, too_late], None

    features = materialize_news_evidence(
        ["AAPL"],
        cutoff,
        previous_cutoff=prev,
        fetch_page=fetch_page,
        score_fn=lambda texts: [
            SentimentScore("positive", 0.8, 0.1, 0.1, 0.7, 0.8) for _ in texts
        ],
    )
    assert features["AAPL"].article_count == 1


def test_make_news_helper_has_audit_fields() -> None:
    row = make_news("MSFT")
    assert row.average_confidence > 0
    assert row.unique_source_count >= 0
