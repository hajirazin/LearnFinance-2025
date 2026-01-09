"""API-level tests for news sentiment signals endpoint."""

import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from brain_api.core.finbert import SentimentScore
from brain_api.core.news_sentiment import Article
from brain_api.main import app
from brain_api.routes.signals import (
    get_data_base_path,
    get_news_fetcher,
    get_sentiment_parquet_path,
    get_sentiment_scorer,
)

# ============================================================================
# Mock implementations for testing
# ============================================================================


class MockNewsFetcher:
    """Mock news fetcher that returns deterministic fake articles."""

    def __init__(self, articles_per_symbol: int = 5):
        self.articles_per_symbol = articles_per_symbol

    def fetch(self, symbol: str, max_articles: int) -> list[Article]:
        """Return mock articles for testing."""
        articles = []
        count = min(max_articles, self.articles_per_symbol)

        for i in range(count):
            # Deterministic timestamp based on symbol and index
            days_ago = i + 1
            published = datetime(2025, 12, 30 - days_ago, 12, 0, 0, tzinfo=UTC)

            articles.append(
                Article(
                    title=f"{symbol} stock news article {i + 1}",
                    publisher=f"Publisher {i + 1}",
                    link=f"https://example.com/{symbol.lower()}/article-{i + 1}",
                    published=published,
                    summary=f"Summary of article {i + 1} about {symbol}.",
                )
            )

        return articles


class MockNewsFetcherEmpty:
    """Mock news fetcher that returns no articles."""

    def fetch(self, symbol: str, max_articles: int) -> list[Article]:
        """Return empty list."""
        return []


class MockNewsFetcherPartial:
    """Mock news fetcher that returns articles only for certain symbols."""

    def __init__(self, symbols_with_news: set[str]):
        self.symbols_with_news = symbols_with_news
        self._inner = MockNewsFetcher(articles_per_symbol=5)

    def fetch(self, symbol: str, max_articles: int) -> list[Article]:
        """Return articles only for symbols in the allowed set."""
        if symbol in self.symbols_with_news:
            return self._inner.fetch(symbol, max_articles)
        return []


class MockSentimentScorer:
    """Mock sentiment scorer that returns deterministic results."""

    def score(self, text: str) -> SentimentScore:
        """Score a single text."""
        return self.score_batch([text])[0]

    def score_batch(self, texts: list[str]) -> list[SentimentScore]:
        """Score batch of texts with deterministic results based on content."""
        results = []
        for text in texts:
            # Deterministic sentiment based on article number in text
            text_lower = text.lower()

            if "article 1" in text_lower:
                # Positive sentiment
                results.append(
                    SentimentScore(
                        label="positive",
                        p_pos=0.8,
                        p_neg=0.1,
                        p_neu=0.1,
                        score=0.7,
                        confidence=0.8,
                    )
                )
            elif "article 2" in text_lower:
                # Negative sentiment
                results.append(
                    SentimentScore(
                        label="negative",
                        p_pos=0.1,
                        p_neg=0.75,
                        p_neu=0.15,
                        score=-0.65,
                        confidence=0.75,
                    )
                )
            else:
                # Neutral sentiment for others
                results.append(
                    SentimentScore(
                        label="neutral",
                        p_pos=0.3,
                        p_neg=0.3,
                        p_neu=0.4,
                        score=0.0,
                        confidence=0.4,
                    )
                )

        return results


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_data_path():
    """Create a temporary data directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def client_with_mocks(temp_data_path):
    """Create test client with mocked dependencies."""
    app.dependency_overrides[get_news_fetcher] = lambda: MockNewsFetcher()
    app.dependency_overrides[get_sentiment_scorer] = lambda: MockSentimentScorer()
    app.dependency_overrides[get_data_base_path] = lambda: temp_data_path

    client = TestClient(app)
    yield client

    app.dependency_overrides.clear()


# ============================================================================
# Scenario 1: Basic successful request
# ============================================================================


def test_news_sentiment_returns_200(client_with_mocks):
    """POST /signals/news with valid symbols returns 200."""
    response = client_with_mocks.post(
        "/signals/news",
        json={"symbols": ["AAPL", "MSFT"]},
    )
    assert response.status_code == 200


def test_news_sentiment_returns_all_symbols(client_with_mocks):
    """POST /signals/news returns data for all requested symbols."""
    response = client_with_mocks.post(
        "/signals/news",
        json={"symbols": ["AAPL", "MSFT", "GOOGL"]},
    )
    assert response.status_code == 200

    data = response.json()
    assert "per_symbol" in data
    assert len(data["per_symbol"]) == 3

    symbols_returned = {s["symbol"] for s in data["per_symbol"]}
    assert symbols_returned == {"AAPL", "MSFT", "GOOGL"}


def test_news_sentiment_returns_required_fields(client_with_mocks):
    """POST /signals/news returns all required response fields."""
    response = client_with_mocks.post(
        "/signals/news",
        json={"symbols": ["AAPL"]},
    )
    assert response.status_code == 200

    data = response.json()

    # Top-level fields
    assert "run_id" in data
    assert "attempt" in data
    assert "as_of_date" in data
    assert "from_cache" in data
    assert "per_symbol" in data

    # Per-symbol fields
    assert len(data["per_symbol"]) == 1
    sym = data["per_symbol"][0]
    assert "symbol" in sym
    assert "article_count_fetched" in sym
    assert "article_count_used" in sym
    assert "sentiment_score" in sym
    assert "insufficient_news" in sym
    assert "top_k_articles" in sym


def test_news_sentiment_returns_sentiment_score(client_with_mocks):
    """POST /signals/news returns numeric sentiment score."""
    response = client_with_mocks.post(
        "/signals/news",
        json={"symbols": ["AAPL"]},
    )
    assert response.status_code == 200

    data = response.json()
    sym = data["per_symbol"][0]

    assert isinstance(sym["sentiment_score"], float | int)
    assert -1 <= sym["sentiment_score"] <= 1


def test_news_sentiment_returns_top_k_articles(client_with_mocks):
    """POST /signals/news returns top K articles with FinBERT scores."""
    response = client_with_mocks.post(
        "/signals/news",
        json={"symbols": ["AAPL"], "return_top_k": 3, "max_articles_per_symbol": 5},
    )
    assert response.status_code == 200

    data = response.json()
    sym = data["per_symbol"][0]

    # Should return exactly 3 articles (top K)
    assert len(sym["top_k_articles"]) == 3

    # Check article fields
    article = sym["top_k_articles"][0]
    assert "title" in article
    assert "publisher" in article
    assert "link" in article
    assert "published" in article
    assert "finbert_label" in article
    assert "finbert_p_pos" in article
    assert "finbert_p_neg" in article
    assert "finbert_p_neu" in article
    assert "article_score" in article


# ============================================================================
# Scenario 2: Request validation (constraint tests per user rule)
# ============================================================================


def test_news_sentiment_empty_symbols_returns_422(client_with_mocks):
    """POST /signals/news with empty symbols list returns 422."""
    response = client_with_mocks.post(
        "/signals/news",
        json={"symbols": []},
    )
    assert response.status_code == 422


def test_news_sentiment_no_symbols_returns_422(client_with_mocks):
    """POST /signals/news without symbols field returns 422."""
    response = client_with_mocks.post(
        "/signals/news",
        json={},
    )
    assert response.status_code == 422


def test_news_sentiment_max_symbols_exceeded_returns_422(client_with_mocks):
    """POST /signals/news with too many symbols returns 422."""
    # MAX_SYMBOLS is 50
    symbols = [f"SYM{i}" for i in range(51)]
    response = client_with_mocks.post(
        "/signals/news",
        json={"symbols": symbols},
    )
    assert response.status_code == 422


def test_news_sentiment_max_articles_cap_enforced(client_with_mocks):
    """POST /signals/news enforces max_articles_per_symbol cap."""
    # MAX_ARTICLES_PER_SYMBOL is 30
    response = client_with_mocks.post(
        "/signals/news",
        json={"symbols": ["AAPL"], "max_articles_per_symbol": 31},
    )
    assert response.status_code == 422


def test_news_sentiment_max_return_top_k_cap_enforced(client_with_mocks):
    """POST /signals/news enforces return_top_k cap."""
    # MAX_RETURN_TOP_K is 10
    response = client_with_mocks.post(
        "/signals/news",
        json={"symbols": ["AAPL"], "return_top_k": 11},
    )
    assert response.status_code == 422


def test_news_sentiment_min_articles_enforced(client_with_mocks):
    """POST /signals/news enforces min value for max_articles_per_symbol."""
    response = client_with_mocks.post(
        "/signals/news",
        json={"symbols": ["AAPL"], "max_articles_per_symbol": 0},
    )
    assert response.status_code == 422


def test_news_sentiment_min_return_top_k_enforced(client_with_mocks):
    """POST /signals/news enforces min value for return_top_k."""
    response = client_with_mocks.post(
        "/signals/news",
        json={"symbols": ["AAPL"], "return_top_k": 0},
    )
    assert response.status_code == 422


# ============================================================================
# Scenario 3: Custom parameters
# ============================================================================


def test_news_sentiment_custom_as_of_date(client_with_mocks):
    """POST /signals/news respects custom as_of_date."""
    response = client_with_mocks.post(
        "/signals/news",
        json={"symbols": ["AAPL"], "as_of_date": "2025-01-15"},
    )
    assert response.status_code == 200

    data = response.json()
    assert data["as_of_date"] == "2025-01-15"


def test_news_sentiment_custom_run_id(client_with_mocks):
    """POST /signals/news respects custom run_id."""
    response = client_with_mocks.post(
        "/signals/news",
        json={"symbols": ["AAPL"], "run_id": "custom:test-run"},
    )
    assert response.status_code == 200

    data = response.json()
    assert data["run_id"] == "custom:test-run"


def test_news_sentiment_custom_attempt(client_with_mocks):
    """POST /signals/news respects custom attempt number."""
    response = client_with_mocks.post(
        "/signals/news",
        json={"symbols": ["AAPL"], "attempt": 3},
    )
    assert response.status_code == 200

    data = response.json()
    assert data["attempt"] == 3


def test_news_sentiment_return_top_k_respects_limit(client_with_mocks):
    """POST /signals/news returns at most return_top_k articles."""
    response = client_with_mocks.post(
        "/signals/news",
        json={"symbols": ["AAPL"], "max_articles_per_symbol": 10, "return_top_k": 2},
    )
    assert response.status_code == 200

    data = response.json()
    sym = data["per_symbol"][0]

    # Should return at most 2 articles
    assert len(sym["top_k_articles"]) <= 2


# ============================================================================
# Scenario 4: No news available (insufficient news handling)
# ============================================================================


def test_news_sentiment_no_articles_marks_insufficient(temp_data_path):
    """When no articles found, symbol is marked as insufficient_news=True."""
    app.dependency_overrides.clear()
    app.dependency_overrides[get_news_fetcher] = lambda: MockNewsFetcherEmpty()
    app.dependency_overrides[get_sentiment_scorer] = lambda: MockSentimentScorer()
    app.dependency_overrides[get_data_base_path] = lambda: temp_data_path

    client = TestClient(app)

    try:
        response = client.post(
            "/signals/news",
            json={"symbols": ["AAPL"]},
        )
        assert response.status_code == 200

        data = response.json()
        sym = data["per_symbol"][0]

        assert sym["insufficient_news"] is True
        assert sym["article_count_fetched"] == 0
        assert sym["sentiment_score"] == 0.0
        assert sym["top_k_articles"] == []
    finally:
        app.dependency_overrides.clear()


def test_news_sentiment_partial_coverage(temp_data_path):
    """Some symbols with news, some without, handles correctly."""
    app.dependency_overrides.clear()
    app.dependency_overrides[get_news_fetcher] = lambda: MockNewsFetcherPartial(
        symbols_with_news={"AAPL", "MSFT"}
    )
    app.dependency_overrides[get_sentiment_scorer] = lambda: MockSentimentScorer()
    app.dependency_overrides[get_data_base_path] = lambda: temp_data_path

    client = TestClient(app)

    try:
        response = client.post(
            "/signals/news",
            json={"symbols": ["AAPL", "UNKNOWNSYMBOL", "MSFT"]},
        )
        assert response.status_code == 200

        data = response.json()
        assert len(data["per_symbol"]) == 3

        # Find each symbol
        results_by_symbol = {s["symbol"]: s for s in data["per_symbol"]}

        # AAPL and MSFT should have news
        assert results_by_symbol["AAPL"]["insufficient_news"] is False
        assert results_by_symbol["AAPL"]["article_count_fetched"] > 0

        assert results_by_symbol["MSFT"]["insufficient_news"] is False
        assert results_by_symbol["MSFT"]["article_count_fetched"] > 0

        # UNKNOWNSYMBOL should have no news
        assert results_by_symbol["UNKNOWNSYMBOL"]["insufficient_news"] is True
        assert results_by_symbol["UNKNOWNSYMBOL"]["article_count_fetched"] == 0
    finally:
        app.dependency_overrides.clear()


# ============================================================================
# Scenario 5: Caching/idempotency
# ============================================================================


def test_news_sentiment_caches_result(client_with_mocks):
    """Second request with same run_id+attempt returns cached result."""
    request_body = {
        "symbols": ["AAPL"],
        "run_id": "paper:2025-12-30",
        "attempt": 1,
    }

    # First request - should not be cached
    response1 = client_with_mocks.post("/signals/news", json=request_body)
    assert response1.status_code == 200
    data1 = response1.json()
    assert data1["from_cache"] is False

    # Second request - should be cached
    response2 = client_with_mocks.post("/signals/news", json=request_body)
    assert response2.status_code == 200
    data2 = response2.json()
    assert data2["from_cache"] is True


def test_news_sentiment_different_attempt_not_cached(client_with_mocks):
    """Different attempt number creates new (non-cached) result."""
    # First request with attempt=1
    response1 = client_with_mocks.post(
        "/signals/news",
        json={"symbols": ["AAPL"], "run_id": "paper:2025-12-30", "attempt": 1},
    )
    assert response1.status_code == 200
    assert response1.json()["from_cache"] is False

    # Second request with attempt=2
    response2 = client_with_mocks.post(
        "/signals/news",
        json={"symbols": ["AAPL"], "run_id": "paper:2025-12-30", "attempt": 2},
    )
    assert response2.status_code == 200
    assert response2.json()["from_cache"] is False


# ============================================================================
# Scenario 6: Top K sorting (most positive first)
# ============================================================================


def test_news_sentiment_top_k_sorted_by_score_desc(client_with_mocks):
    """Top K articles are sorted by article_score descending."""
    response = client_with_mocks.post(
        "/signals/news",
        json={"symbols": ["AAPL"], "max_articles_per_symbol": 5, "return_top_k": 5},
    )
    assert response.status_code == 200

    data = response.json()
    articles = data["per_symbol"][0]["top_k_articles"]

    # Extract scores
    scores = [a["article_score"] for a in articles]

    # Should be sorted descending (most positive first)
    assert scores == sorted(scores, reverse=True)


# ============================================================================
# Scenario 7: Default values
# ============================================================================


def test_news_sentiment_default_run_id(client_with_mocks):
    """Default run_id is paper:<as_of_date>."""
    response = client_with_mocks.post(
        "/signals/news",
        json={"symbols": ["AAPL"], "as_of_date": "2025-12-30"},
    )
    assert response.status_code == 200

    data = response.json()
    assert data["run_id"] == "paper:2025-12-30"


def test_news_sentiment_default_attempt(client_with_mocks):
    """Default attempt is 1."""
    response = client_with_mocks.post(
        "/signals/news",
        json={"symbols": ["AAPL"]},
    )
    assert response.status_code == 200

    data = response.json()
    assert data["attempt"] == 1


# ============================================================================
# Historical News Sentiment Endpoint Tests
# ============================================================================

# Sample parquet data for testing
SAMPLE_SENTIMENT_DATA = pd.DataFrame(
    {
        "date": ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
        "symbol": ["AAPL", "MSFT", "AAPL", "MSFT"],
        "sentiment_score": [0.5, -0.3, 0.2, 0.1],
        "article_count": [10, 5, 8, 3],
        "avg_confidence": [0.85, 0.90, 0.88, 0.75],
        "p_pos_avg": [0.6, 0.2, 0.5, 0.4],
        "p_neg_avg": [0.1, 0.5, 0.3, 0.3],
        "total_articles": [10, 5, 8, 3],
    }
)


@pytest.fixture
def temp_parquet_path():
    """Create a temporary parquet file with sample data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = Path(tmpdir) / "daily_sentiment.parquet"
        SAMPLE_SENTIMENT_DATA.to_parquet(parquet_path, index=False)
        yield parquet_path


@pytest.fixture
def client_with_parquet(temp_parquet_path):
    """Create test client with mock parquet path."""
    app.dependency_overrides[get_sentiment_parquet_path] = lambda: temp_parquet_path

    client = TestClient(app)
    yield client

    app.dependency_overrides.clear()


@pytest.fixture
def client_with_missing_parquet():
    """Create test client with non-existent parquet path."""
    app.dependency_overrides[get_sentiment_parquet_path] = lambda: Path(
        "/nonexistent/file.parquet"
    )

    client = TestClient(app)
    yield client

    app.dependency_overrides.clear()


# ============================================================================
# Historical Sentiment - Basic tests
# ============================================================================


def test_historical_sentiment_returns_200(client_with_parquet):
    """POST /signals/news/historical with valid params returns 200."""
    response = client_with_parquet.post(
        "/signals/news/historical",
        json={
            "symbols": ["AAPL"],
            "start_date": "2024-01-02",
            "end_date": "2024-01-03",
        },
    )
    assert response.status_code == 200


def test_historical_sentiment_returns_required_fields(client_with_parquet):
    """POST /signals/news/historical returns all required response fields."""
    response = client_with_parquet.post(
        "/signals/news/historical",
        json={
            "symbols": ["AAPL"],
            "start_date": "2024-01-02",
            "end_date": "2024-01-03",
        },
    )
    assert response.status_code == 200

    data = response.json()
    assert "start_date" in data
    assert "end_date" in data
    assert "data" in data


def test_historical_sentiment_returns_data_points(client_with_parquet):
    """POST /signals/news/historical returns sentiment data points."""
    response = client_with_parquet.post(
        "/signals/news/historical",
        json={
            "symbols": ["AAPL"],
            "start_date": "2024-01-02",
            "end_date": "2024-01-03",
        },
    )
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data["data"], list)
    assert len(data["data"]) > 0

    # Check data point fields
    point = data["data"][0]
    assert "symbol" in point
    assert "date" in point
    assert "sentiment_score" in point
    assert "article_count" in point
    assert "p_pos_avg" in point
    assert "p_neg_avg" in point


def test_historical_sentiment_correct_data(client_with_parquet):
    """POST /signals/news/historical returns correct sentiment values."""
    response = client_with_parquet.post(
        "/signals/news/historical",
        json={
            "symbols": ["AAPL"],
            "start_date": "2024-01-02",
            "end_date": "2024-01-02",
        },
    )
    assert response.status_code == 200

    data = response.json()
    # Should have exactly 1 data point for AAPL on 2024-01-02
    assert len(data["data"]) == 1

    point = data["data"][0]
    assert point["symbol"] == "AAPL"
    assert point["date"] == "2024-01-02"
    assert point["sentiment_score"] == 0.5
    assert point["article_count"] == 10


# ============================================================================
# Historical Sentiment - Validation tests
# ============================================================================


def test_historical_sentiment_empty_symbols_returns_422(client_with_parquet):
    """POST /signals/news/historical with empty symbols returns 422."""
    response = client_with_parquet.post(
        "/signals/news/historical",
        json={
            "symbols": [],
            "start_date": "2024-01-02",
            "end_date": "2024-01-03",
        },
    )
    assert response.status_code == 422


def test_historical_sentiment_no_symbols_returns_422(client_with_parquet):
    """POST /signals/news/historical without symbols returns 422."""
    response = client_with_parquet.post(
        "/signals/news/historical",
        json={
            "start_date": "2024-01-02",
            "end_date": "2024-01-03",
        },
    )
    assert response.status_code == 422


def test_historical_sentiment_max_symbols_exceeded_returns_422(client_with_parquet):
    """POST /signals/news/historical with too many symbols returns 422."""
    # MAX_HISTORICAL_SENTIMENT_SYMBOLS is 20
    symbols = [f"SYM{i}" for i in range(21)]
    response = client_with_parquet.post(
        "/signals/news/historical",
        json={
            "symbols": symbols,
            "start_date": "2024-01-02",
            "end_date": "2024-01-03",
        },
    )
    assert response.status_code == 422


def test_historical_sentiment_requires_start_date(client_with_parquet):
    """POST /signals/news/historical requires start_date."""
    response = client_with_parquet.post(
        "/signals/news/historical",
        json={
            "symbols": ["AAPL"],
            "end_date": "2024-01-03",
        },
    )
    assert response.status_code == 422


def test_historical_sentiment_requires_end_date(client_with_parquet):
    """POST /signals/news/historical requires end_date."""
    response = client_with_parquet.post(
        "/signals/news/historical",
        json={
            "symbols": ["AAPL"],
            "start_date": "2024-01-02",
        },
    )
    assert response.status_code == 422


# ============================================================================
# Historical Sentiment - Neutral fallback tests
# ============================================================================


def test_historical_sentiment_neutral_for_missing_symbol(client_with_parquet):
    """Missing symbol in parquet returns neutral sentiment."""
    response = client_with_parquet.post(
        "/signals/news/historical",
        json={
            "symbols": ["UNKNOWNSYM"],
            "start_date": "2024-01-02",
            "end_date": "2024-01-02",
        },
    )
    assert response.status_code == 200

    data = response.json()
    assert len(data["data"]) == 1

    point = data["data"][0]
    assert point["symbol"] == "UNKNOWNSYM"
    assert point["sentiment_score"] == 0.0  # Neutral
    assert point["article_count"] is None  # No data marker
    assert point["p_pos_avg"] is None
    assert point["p_neg_avg"] is None


def test_historical_sentiment_neutral_for_missing_date(client_with_parquet):
    """Missing date in parquet returns neutral sentiment."""
    response = client_with_parquet.post(
        "/signals/news/historical",
        json={
            "symbols": ["AAPL"],
            "start_date": "2023-01-01",  # Before data range
            "end_date": "2023-01-01",
        },
    )
    assert response.status_code == 200

    data = response.json()
    assert len(data["data"]) == 1

    point = data["data"][0]
    assert point["symbol"] == "AAPL"
    assert point["date"] == "2023-01-01"
    assert point["sentiment_score"] == 0.0  # Neutral
    assert point["article_count"] is None


def test_historical_sentiment_returns_all_requested_combos(client_with_parquet):
    """Returns all date+symbol combinations, filling missing with neutral."""
    response = client_with_parquet.post(
        "/signals/news/historical",
        json={
            "symbols": ["AAPL", "GOOGL"],  # GOOGL not in sample data
            "start_date": "2024-01-02",
            "end_date": "2024-01-03",
        },
    )
    assert response.status_code == 200

    data = response.json()
    # 2 symbols x 2 dates = 4 data points
    assert len(data["data"]) == 4

    # Group by symbol
    by_symbol = {}
    for point in data["data"]:
        by_symbol.setdefault(point["symbol"], []).append(point)

    assert set(by_symbol.keys()) == {"AAPL", "GOOGL"}

    # AAPL should have real data
    aapl_points = by_symbol["AAPL"]
    assert len(aapl_points) == 2
    for p in aapl_points:
        assert p["sentiment_score"] != 0.0 or p["article_count"] is not None

    # GOOGL should have neutral data (all None markers)
    googl_points = by_symbol["GOOGL"]
    assert len(googl_points) == 2
    for p in googl_points:
        assert p["sentiment_score"] == 0.0
        assert p["article_count"] is None


def test_historical_sentiment_missing_parquet_returns_neutral(
    client_with_missing_parquet,
):
    """When parquet file doesn't exist, all data is neutral."""
    response = client_with_missing_parquet.post(
        "/signals/news/historical",
        json={
            "symbols": ["AAPL"],
            "start_date": "2024-01-02",
            "end_date": "2024-01-02",
        },
    )
    assert response.status_code == 200

    data = response.json()
    assert len(data["data"]) == 1

    point = data["data"][0]
    assert point["sentiment_score"] == 0.0
    assert point["article_count"] is None


# ============================================================================
# Historical Sentiment - Multiple symbols test
# ============================================================================


def test_historical_sentiment_multiple_symbols(client_with_parquet):
    """POST /signals/news/historical with multiple symbols returns all."""
    response = client_with_parquet.post(
        "/signals/news/historical",
        json={
            "symbols": ["AAPL", "MSFT"],
            "start_date": "2024-01-02",
            "end_date": "2024-01-03",
        },
    )
    assert response.status_code == 200

    data = response.json()
    # 2 symbols x 2 dates = 4 data points
    assert len(data["data"]) == 4

    symbols_in_response = {p["symbol"] for p in data["data"]}
    assert symbols_in_response == {"AAPL", "MSFT"}
