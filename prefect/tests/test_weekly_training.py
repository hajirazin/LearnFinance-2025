"""Tests for weekly training flow."""

from unittest.mock import patch

import pytest

from flows import weekly_training  # noqa: F401 - needed for patching
from flows.models import (
    HalalUniverseResponse,
    RefreshTrainingDataResponse,
    TrainingResponse,
)
from flows.weekly_training import weekly_training_flow


@pytest.fixture
def mock_universe_response():
    """Mock response for halal universe endpoint."""
    return {
        "stocks": [
            {
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "max_weight": 10.5,
                "sources": ["SPUS", "HLAL"],
            },
            {
                "symbol": "MSFT",
                "name": "Microsoft Corp.",
                "max_weight": 8.2,
                "sources": ["SPUS"],
            },
            {
                "symbol": "GOOGL",
                "name": "Alphabet Inc.",
                "max_weight": 5.1,
                "sources": ["HLAL"],
            },
        ],
        "etfs_used": ["SPUS", "HLAL", "SPTE"],
        "total_stocks": 3,
        "fetched_at": "2026-02-03T12:00:00+00:00",
    }


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


class TestModels:
    """Test Pydantic models."""

    def test_halal_universe_response(self, mock_universe_response):
        """Test HalalUniverseResponse model."""
        response = HalalUniverseResponse(**mock_universe_response)
        assert response.symbols == ["AAPL", "MSFT", "GOOGL"]
        assert response.total_stocks == 3
        assert response.etfs_used == ["SPUS", "HLAL", "SPTE"]
        assert len(response.stocks) == 3
        assert response.stocks[0].symbol == "AAPL"
        assert response.stocks[0].max_weight == 10.5

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


class TestFlow:
    """Test the full flow with mocked tasks."""

    @patch("flows.weekly_training.train_sac")
    @patch("flows.weekly_training.train_ppo")
    @patch("flows.weekly_training.train_patchtst")
    @patch("flows.weekly_training.train_lstm")
    @patch("flows.weekly_training.refresh_training_data")
    @patch("flows.weekly_training.get_halal_universe")
    def test_weekly_training_flow(
        self,
        mock_universe,
        mock_refresh,
        mock_lstm,
        mock_patchtst,
        mock_ppo,
        mock_sac,
        mock_universe_response,
        mock_refresh_response,
        mock_training_response,
    ):
        """Test full weekly training flow execution."""
        # Setup mocks
        mock_universe.return_value = HalalUniverseResponse(**mock_universe_response)
        mock_refresh.return_value = RefreshTrainingDataResponse(**mock_refresh_response)

        training_resp = TrainingResponse(**mock_training_response)
        mock_lstm.submit.return_value.result.return_value = training_resp
        mock_patchtst.submit.return_value.result.return_value = training_resp
        mock_ppo.submit.return_value.result.return_value = training_resp
        mock_sac.submit.return_value.result.return_value = training_resp

        # Run flow
        result = weekly_training_flow()

        # Verify result structure
        assert result["universe_count"] == 3
        assert result["refresh"]["sentiment_gaps_filled"] == 10
        assert result["refresh"]["fundamentals_refreshed"] == 2
        assert result["lstm"]["version"] == "v1.0.0"
        assert result["lstm"]["promoted"] is True
        assert result["patchtst"]["version"] == "v1.0.0"
        assert result["ppo"]["version"] == "v1.0.0"
        assert result["sac"]["version"] == "v1.0.0"

        # Verify task calls
        mock_universe.assert_called_once()
        mock_refresh.assert_called_once_with(["AAPL", "MSFT", "GOOGL"])
        mock_lstm.submit.assert_called_once()
        mock_patchtst.submit.assert_called_once()
        mock_ppo.submit.assert_called_once()
        mock_sac.submit.assert_called_once()

    @patch("flows.weekly_training.train_sac")
    @patch("flows.weekly_training.train_ppo")
    @patch("flows.weekly_training.train_patchtst")
    @patch("flows.weekly_training.train_lstm")
    @patch("flows.weekly_training.refresh_training_data")
    @patch("flows.weekly_training.get_halal_universe")
    def test_flow_passes_symbols_to_refresh(
        self,
        mock_universe,
        mock_refresh,
        mock_lstm,
        mock_patchtst,
        mock_ppo,
        mock_sac,
    ):
        """Test that flow passes universe symbols to refresh_training_data."""
        # Setup mocks with different symbols
        mock_universe.return_value = HalalUniverseResponse(
            stocks=[
                {
                    "symbol": "TSLA",
                    "name": "Tesla Inc.",
                    "max_weight": 7.5,
                    "sources": ["SPUS"],
                },
                {
                    "symbol": "NVDA",
                    "name": "NVIDIA Corp.",
                    "max_weight": 6.2,
                    "sources": ["HLAL"],
                },
            ],
            etfs_used=["SPUS", "HLAL"],
            total_stocks=2,
            fetched_at="2026-02-03T12:00:00+00:00",
        )
        mock_refresh.return_value = RefreshTrainingDataResponse(
            sentiment_gaps_filled=5,
            sentiment_gaps_remaining=0,
            fundamentals_refreshed=["TSLA"],
            fundamentals_skipped=["NVDA"],
            fundamentals_failed=[],
            duration_seconds=2.0,
        )

        training_resp = TrainingResponse(
            version="v1.0.0",
            data_window_start="2020-01-01",
            data_window_end="2024-01-01",
            metrics={},
            promoted=True,
        )
        mock_lstm.submit.return_value.result.return_value = training_resp
        mock_patchtst.submit.return_value.result.return_value = training_resp
        mock_ppo.submit.return_value.result.return_value = training_resp
        mock_sac.submit.return_value.result.return_value = training_resp

        # Run flow
        weekly_training_flow()

        # Verify refresh was called with correct symbols
        mock_refresh.assert_called_once_with(["TSLA", "NVDA"])
