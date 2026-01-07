"""Tests for news sentiment ETL endpoints."""

import time
from datetime import date
from unittest.mock import patch

from fastapi.testclient import TestClient

from brain_api.main import app
from brain_api.etl.gap_fill import _create_zero_article_rows

client = TestClient(app)


class TestETLNewsEndpoints:
    """Tests for /etl/news-sentiment endpoints."""

    def test_start_job_returns_202(self):
        """POST /etl/news-sentiment should return 202 with job_id."""
        # Mock the background task to avoid actual ETL execution
        with patch("brain_api.routes.etl._run_etl_job"):
            response = client.post(
                "/etl/news-sentiment",
                json={
                    "batch_size": 64,
                    "max_articles": 10,
                    "local_only": True,
                },
            )

        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "pending"
        assert "message" in data

    def test_start_job_with_defaults(self):
        """POST /etl/news-sentiment with empty body uses defaults."""
        with patch("brain_api.routes.etl._run_etl_job"):
            response = client.post("/etl/news-sentiment", json={})

        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data

    def test_get_job_status_not_found(self):
        """GET /etl/news-sentiment/{job_id} returns 404 for unknown job."""
        response = client.get("/etl/news-sentiment/nonexistent")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_get_job_status_after_creation(self):
        """GET /etl/news-sentiment/{job_id} returns job status."""
        # Create a job first
        with patch("brain_api.routes.etl._run_etl_job"):
            create_response = client.post(
                "/etl/news-sentiment",
                json={"max_articles": 10, "local_only": True},
            )

        job_id = create_response.json()["job_id"]

        # Get status
        response = client.get(f"/etl/news-sentiment/{job_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert data["status"] in ["pending", "running", "completed", "failed"]
        assert "started_at" in data
        assert "config" in data

    def test_list_jobs(self):
        """GET /etl/news-sentiment/jobs returns job list."""
        # Create a few jobs
        with patch("brain_api.routes.etl._run_etl_job"):
            for _ in range(3):
                client.post(
                    "/etl/news-sentiment",
                    json={"max_articles": 10, "local_only": True},
                )

        response = client.get("/etl/news-sentiment/jobs")

        assert response.status_code == 200
        data = response.json()
        assert "jobs" in data
        assert "total" in data
        assert data["total"] >= 3
        assert len(data["jobs"]) >= 3

    def test_job_config_preserved(self):
        """Job config should be preserved and returned in status."""
        with patch("brain_api.routes.etl._run_etl_job"):
            create_response = client.post(
                "/etl/news-sentiment",
                json={
                    "batch_size": 128,
                    "max_articles": 50,
                    "sentiment_threshold": 0.2,
                    "filter_to_halal": True,
                    "local_only": True,
                },
            )

        job_id = create_response.json()["job_id"]
        response = client.get(f"/etl/news-sentiment/{job_id}")

        data = response.json()
        assert data["config"]["batch_size"] == 128
        assert data["config"]["max_articles"] == 50
        assert data["config"]["sentiment_threshold"] == 0.2
        assert data["config"]["filter_to_halal"] is True
        assert data["config"]["local_only"] is True

    def test_invalid_batch_size(self):
        """POST with invalid batch_size should return 422."""
        response = client.post(
            "/etl/news-sentiment",
            json={"batch_size": 0},  # Invalid: must be >= 1
        )

        assert response.status_code == 422

    def test_invalid_threshold(self):
        """POST with invalid sentiment_threshold should return 422."""
        response = client.post(
            "/etl/news-sentiment",
            json={"sentiment_threshold": 1.5},  # Invalid: must be <= 1.0
        )

        assert response.status_code == 422


class TestETLJobLifecycle:
    """Integration tests for job lifecycle (requires mocking pipeline)."""

    def test_job_transitions_to_running(self):
        """Job should transition from pending to running when started."""
        from brain_api.routes.etl import _jobs

        # Create job and let background task start
        with patch(
            "brain_api.routes.etl.run_pipeline",
            return_value={"status": "completed"},
        ):
            response = client.post(
                "/etl/news-sentiment",
                json={"max_articles": 10, "local_only": True},
            )

        job_id = response.json()["job_id"]

        # Give background task time to start
        time.sleep(0.1)

        # Check job in internal store
        job = _jobs.get(job_id)
        assert job is not None
        # Status should be running or completed (depending on timing)
        assert job.status in ["running", "completed"]


class TestTickerAliases:
    """Tests for ticker alias functionality."""

    def test_expand_with_aliases(self):
        """expand_with_aliases should add historical tickers."""
        from brain_api.core.ticker_aliases import expand_with_aliases

        symbols = {"META", "AAPL"}
        expanded = expand_with_aliases(symbols)

        assert "META" in expanded
        assert "FB" in expanded  # Historical alias
        assert "AAPL" in expanded

    def test_normalize_symbol(self):
        """normalize_symbol should map old tickers to current."""
        from brain_api.core.ticker_aliases import normalize_symbol

        assert normalize_symbol("FB") == "META"
        assert normalize_symbol("AAPL") == "AAPL"
        assert normalize_symbol("META") == "META"

    def test_normalize_symbols_deduplicates(self):
        """normalize_symbols should deduplicate after normalization."""
        from brain_api.core.ticker_aliases import normalize_symbols

        symbols = ["FB", "AAPL", "META"]
        normalized = normalize_symbols(symbols)

        # FB and META should merge to just META
        assert "META" in normalized
        assert "AAPL" in normalized
        assert len(normalized) == 2


class TestSentimentScore:
    """Tests for unified SentimentScore."""

    def test_sentiment_score_creation(self):
        """SentimentScore should be creatable with all fields."""
        from brain_api.core.finbert import SentimentScore

        score = SentimentScore(
            label="positive",
            p_pos=0.8,
            p_neg=0.1,
            p_neu=0.1,
            score=0.7,
            confidence=0.8,
        )

        assert score.label == "positive"
        assert score.score == 0.7
        assert score.confidence == 0.8

    def test_passes_threshold(self):
        """passes_threshold should check |p_pos - p_neg| >= threshold."""
        from brain_api.core.finbert import SentimentScore

        strong = SentimentScore(
            label="positive",
            p_pos=0.8,
            p_neg=0.1,
            p_neu=0.1,
            score=0.7,
            confidence=0.8,
        )
        weak = SentimentScore(
            label="neutral",
            p_pos=0.35,
            p_neg=0.30,
            p_neu=0.35,
            score=0.05,
            confidence=0.35,
        )

        assert strong.passes_threshold(0.5) is True
        assert weak.passes_threshold(0.1) is False

    def test_to_dict_from_dict(self):
        """SentimentScore should serialize and deserialize correctly."""
        from brain_api.core.finbert import SentimentScore

        original = SentimentScore(
            label="negative",
            p_pos=0.1,
            p_neg=0.8,
            p_neu=0.1,
            score=-0.7,
            confidence=0.8,
        )

        data = original.to_dict()
        restored = SentimentScore.from_dict(data)

        assert restored.label == original.label
        assert restored.score == original.score
        assert restored.confidence == original.confidence


class TestZeroArticleRows:
    """Tests for zero-article gap tracking functionality."""

    def test_create_zero_article_rows_empty_list(self):
        """_create_zero_article_rows should return empty list for empty input."""
        result = _create_zero_article_rows([])
        assert result == []

    def test_create_zero_article_rows_single_gap(self):
        """_create_zero_article_rows should create row with correct structure."""
        gaps = [(date(2025, 1, 15), "AAPL")]
        result = _create_zero_article_rows(gaps)

        assert len(result) == 1
        row = result[0]
        assert row["date"] == date(2025, 1, 15)
        assert row["symbol"] == "AAPL"
        assert row["sentiment_score"] == 0.0
        assert row["article_count"] == 0
        assert row["avg_confidence"] == 0.0
        assert row["p_pos_avg"] == 0.0
        assert row["p_neg_avg"] == 0.0
        assert row["total_articles"] == 0

    def test_create_zero_article_rows_multiple_gaps(self):
        """_create_zero_article_rows should handle multiple gaps."""
        gaps = [
            (date(2025, 1, 15), "AAPL"),
            (date(2025, 1, 15), "MSFT"),
            (date(2025, 1, 16), "AAPL"),
        ]
        result = _create_zero_article_rows(gaps)

        assert len(result) == 3
        symbols = {row["symbol"] for row in result}
        assert symbols == {"AAPL", "MSFT"}
        dates = {row["date"] for row in result}
        assert dates == {date(2025, 1, 15), date(2025, 1, 16)}

    def test_create_zero_article_rows_all_fields_present(self):
        """_create_zero_article_rows should include all required parquet fields."""
        gaps = [(date(2025, 1, 15), "NVDA")]
        result = _create_zero_article_rows(gaps)

        required_fields = [
            "date",
            "symbol",
            "sentiment_score",
            "article_count",
            "avg_confidence",
            "p_pos_avg",
            "p_neg_avg",
            "total_articles",
        ]
        for field in required_fields:
            assert field in result[0], f"Missing field: {field}"


class TestGapFillUnmatchedSymbols:
    """Tests for gap fill handling of unmatched symbols."""

    def test_unmatched_symbols_recorded_as_zero_article(self, tmp_path):
        """When articles don't match gap symbols, those symbols get zero-article rows."""
        from datetime import datetime
        from unittest.mock import MagicMock

        from brain_api.core.finbert import SentimentScore
        from brain_api.core.news_api.alpaca import AlpacaNewsArticle
        from brain_api.etl.gap_fill import fill_sentiment_gaps

        import pandas as pd

        # Create path for parquet (file doesn't need to exist - _append_to_parquet handles it)
        parquet_path = tmp_path / "test_sentiment.parquet"

        # Mock dependencies
        with (
            patch("brain_api.etl.gap_fill.get_halal_symbols") as mock_symbols,
            patch("brain_api.etl.gap_fill.find_gaps") as mock_find_gaps,
            patch("brain_api.etl.gap_fill.categorize_gaps") as mock_categorize,
            patch("brain_api.etl.gap_fill.AlpacaNewsClient") as mock_client_cls,
            patch("brain_api.etl.gap_fill.FinBERTScorer") as mock_scorer_cls,
            patch("brain_api.etl.gap_fill.get_gap_statistics") as mock_stats,
        ):
            # Setup: 3 gap symbols to fill
            mock_symbols.return_value = ["AAPL", "MSFT", "GOOGL"]

            # Gap for a date in the past (not today)
            gap_date = date(2024, 6, 15)
            gaps = [(gap_date, "AAPL"), (gap_date, "MSFT"), (gap_date, "GOOGL")]
            mock_find_gaps.return_value = gaps
            mock_categorize.return_value = (gaps, [])  # All fillable

            # Alpaca returns articles, but they mention OTHER symbols (not our gaps)
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.call_count = 1

            # Return an article that mentions NVDA and TSLA (NOT AAPL, MSFT, GOOGL)
            mock_article = AlpacaNewsArticle(
                id="test123",
                headline="NVDA and TSLA surge",
                summary="Tech stocks rally",
                author="Test",
                created_at=datetime(2024, 6, 15, 10, 0, 0),
                updated_at=datetime(2024, 6, 15, 10, 0, 0),
                url="http://test.com",
                symbols=["NVDA", "TSLA"],  # NOT matching our gap symbols
                source="test",
            )
            mock_client.fetch_news_for_date.return_value = [mock_article]

            # Scorer returns a score for the article
            mock_scorer = MagicMock()
            mock_scorer_cls.return_value = mock_scorer
            mock_scorer.score_batch.return_value = [
                SentimentScore(
                    label="positive",
                    p_pos=0.8,
                    p_neg=0.1,
                    p_neu=0.1,
                    score=0.7,
                    confidence=0.8,
                )
            ]

            mock_stats.return_value = {"gaps_found": 0}

            # Run gap fill
            result = fill_sentiment_gaps(
                start_date=date(2024, 1, 1),
                end_date=date(2024, 12, 31),
                parquet_path=parquet_path,
            )

            assert result.success

            # Read the parquet file and check that AAPL, MSFT, GOOGL have zero-article rows
            df = pd.read_parquet(parquet_path)

            # All 3 symbols should have zero-article rows since articles didn't match
            assert len(df) == 3
            for symbol in ["AAPL", "MSFT", "GOOGL"]:
                symbol_row = df[df["symbol"] == symbol]
                assert len(symbol_row) == 1, f"Expected 1 row for {symbol}"
                assert symbol_row.iloc[0]["article_count"] == 0
                assert symbol_row.iloc[0]["sentiment_score"] == 0.0


class TestSentimentGapsEndpoint:
    """Tests for /etl/sentiment-gaps endpoint."""

    def test_start_sentiment_gaps_job_returns_202(self):
        """POST /etl/sentiment-gaps should return 202 with job_id."""
        with patch("brain_api.routes.etl._run_gap_fill_job"):
            response = client.post(
                "/etl/sentiment-gaps",
                json={"start_date": "2025-01-01"},
            )

        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "pending"
        assert "message" in data

    def test_start_sentiment_gaps_job_with_end_date(self):
        """POST /etl/sentiment-gaps should accept end_date parameter."""
        with patch("brain_api.routes.etl._run_gap_fill_job"):
            response = client.post(
                "/etl/sentiment-gaps",
                json={
                    "start_date": "2024-01-01",
                    "end_date": "2024-12-31",
                },
            )

        assert response.status_code == 202

    def test_get_sentiment_gaps_status_not_found(self):
        """GET /etl/sentiment-gaps/{job_id} returns 404 for unknown job."""
        response = client.get("/etl/sentiment-gaps/nonexistent")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_get_sentiment_gaps_status_after_creation(self):
        """GET /etl/sentiment-gaps/{job_id} returns job status."""
        with patch("brain_api.routes.etl._run_gap_fill_job"):
            create_response = client.post(
                "/etl/sentiment-gaps",
                json={"start_date": "2025-01-01"},
            )

        job_id = create_response.json()["job_id"]
        response = client.get(f"/etl/sentiment-gaps/{job_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert data["status"] in ["pending", "running", "completed", "failed"]
        assert "started_at" in data
        assert "config" in data

    def test_sentiment_gaps_invalid_date_format(self):
        """POST with invalid date format should return 400."""
        response = client.post(
            "/etl/sentiment-gaps",
            json={"start_date": "not-a-date"},
        )

        # FastAPI returns 400 for date parsing errors
        assert response.status_code == 400

    def test_sentiment_gaps_missing_start_date(self):
        """POST without start_date should return 422."""
        response = client.post(
            "/etl/sentiment-gaps",
            json={},
        )

        assert response.status_code == 422



