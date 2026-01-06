"""Tests for news sentiment ETL endpoints."""

import time
from unittest.mock import patch

from fastapi.testclient import TestClient

from brain_api.main import app

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


