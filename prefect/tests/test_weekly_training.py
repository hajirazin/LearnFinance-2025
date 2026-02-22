"""Tests for weekly training flow."""

from unittest.mock import patch

import pytest

from flows import weekly_training  # noqa: F401 - needed for patching
from flows.models import (
    RefreshTrainingDataResponse,
    TrainingResponse,
    TrainingSummaryEmailResponse,
    TrainingSummaryResponse,
)
from flows.weekly_training import weekly_training_flow


@pytest.fixture
def mock_refresh_response():
    """Mock response for refresh training data endpoint."""
    return {
        "sentiment_gaps_filled": 10,
        "sentiment_gaps_remaining": 0,
        "fundamentals_refreshed": ["AAPL", "MSFT"],
        "fundamentals_skipped": ["GOOGL"],
        "fundamentals_failed": [],
        "duration_seconds": 5.5,
    }


@pytest.fixture
def mock_training_response():
    """Mock response for training endpoints."""
    return {
        "version": "v1.0.0",
        "data_window_start": "2020-01-01",
        "data_window_end": "2024-01-01",
        "metrics": {"loss": 0.01},
        "promoted": True,
        "prior_version": None,
    }


@pytest.fixture
def mock_summary_response():
    """Mock response for training summary endpoint."""
    return {
        "summary": {
            "para_1_overall": "All models trained successfully.",
            "para_2_lstm": "LSTM shows good performance.",
            "para_3_patchtst": "PatchTST leverages OHLCV approach.",
            "para_4_ppo": "PPO demonstrates solid returns.",
            "para_5_sac": "SAC shows promising results.",
            "para_6_recommendations": "Continue monitoring.",
        },
        "provider": "openai",
        "model_used": "gpt-4o-mini",
        "tokens_used": 500,
    }


@pytest.fixture
def mock_email_response():
    """Mock response for training summary email endpoint."""
    return {
        "is_success": True,
        "subject": "Training Summary: 2020-01-01 to 2024-01-01",
        "body": "<html><body>Training summary email body</body></html>",
    }


class TestModels:
    """Test Pydantic models."""

    def test_refresh_training_data_response(self, mock_refresh_response):
        """Test RefreshTrainingDataResponse model."""
        response = RefreshTrainingDataResponse(**mock_refresh_response)
        assert response.sentiment_gaps_filled == 10
        assert response.fundamentals_refreshed == ["AAPL", "MSFT"]
        assert response.duration_seconds == 5.5

    def test_training_response(self, mock_training_response):
        """Test TrainingResponse model."""
        response = TrainingResponse(**mock_training_response)
        assert response.version == "v1.0.0"
        assert response.promoted is True

    def test_training_response_with_optional_fields(self):
        """Test TrainingResponse with optional fields."""
        response = TrainingResponse(
            version="v2.0.0",
            data_window_start="2020-01-01",
            data_window_end="2024-01-01",
            metrics={"mae": 0.05, "rmse": 0.1},
            promoted=False,
            prior_version="v1.0.0",
            hf_repo="user/model",
            hf_url="https://huggingface.co/user/model",
            symbols_used=["AAPL", "MSFT"],
        )
        assert response.prior_version == "v1.0.0"
        assert response.hf_repo == "user/model"
        assert response.symbols_used == ["AAPL", "MSFT"]

    def test_training_summary_response(self, mock_summary_response):
        """Test TrainingSummaryResponse model."""
        response = TrainingSummaryResponse(**mock_summary_response)
        assert response.provider == "openai"
        assert response.model_used == "gpt-4o-mini"
        assert response.tokens_used == 500
        assert "para_1_overall" in response.summary


class TestFlow:
    """Test the full flow with mocked tasks."""

    @patch("flows.weekly_training.send_training_summary_email")
    @patch("flows.weekly_training.generate_training_summary")
    @patch("flows.weekly_training.train_sac")
    @patch("flows.weekly_training.train_ppo")
    @patch("flows.weekly_training.train_patchtst")
    @patch("flows.weekly_training.train_lstm")
    @patch("flows.weekly_training.refresh_training_data")
    def test_weekly_training_flow(
        self,
        mock_refresh,
        mock_lstm,
        mock_patchtst,
        mock_ppo,
        mock_sac,
        mock_summary,
        mock_email,
        mock_refresh_response,
        mock_training_response,
        mock_summary_response,
        mock_email_response,
    ):
        """Test full weekly training flow execution."""
        # Setup mocks
        mock_refresh.return_value = RefreshTrainingDataResponse(**mock_refresh_response)

        training_resp = TrainingResponse(**mock_training_response)
        mock_lstm.submit.return_value.result.return_value = training_resp
        mock_patchtst.submit.return_value.result.return_value = training_resp
        mock_ppo.submit.return_value.result.return_value = training_resp
        mock_sac.submit.return_value.result.return_value = training_resp

        summary_resp = TrainingSummaryResponse(**mock_summary_response)
        mock_summary.return_value = summary_resp

        email_resp = TrainingSummaryEmailResponse(**mock_email_response)
        mock_email.return_value = email_resp

        # Run flow
        result = weekly_training_flow()

        # Verify result structure
        assert result["refresh"]["sentiment_gaps_filled"] == 10
        assert result["refresh"]["fundamentals_refreshed"] == 2
        assert result["lstm"]["version"] == "v1.0.0"
        assert result["lstm"]["promoted"] is True
        assert result["patchtst"]["version"] == "v1.0.0"
        assert result["ppo"]["version"] == "v1.0.0"
        assert result["sac"]["version"] == "v1.0.0"
        assert result["summary"]["provider"] == "openai"
        assert result["summary"]["model_used"] == "gpt-4o-mini"
        assert result["email"]["is_success"] is True
        assert "Training Summary" in result["email"]["subject"]

        # Verify task calls -- no universe call, refresh called without symbols
        mock_refresh.assert_called_once_with()
        mock_lstm.submit.assert_called_once()
        mock_patchtst.submit.assert_called_once()
        mock_ppo.submit.assert_called_once()
        mock_sac.submit.assert_called_once()
        mock_summary.assert_called_once()
        mock_email.assert_called_once()
