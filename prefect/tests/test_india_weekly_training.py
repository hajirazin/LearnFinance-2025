"""Tests for India weekly training flow."""

from unittest.mock import patch

import pytest

from flows import india_weekly_training  # noqa: F401 - needed for patching
from flows.india_weekly_training import india_weekly_training_flow
from flows.models import (
    TrainingResponse,
    TrainingSummaryEmailResponse,
    TrainingSummaryResponse,
)


@pytest.fixture
def mock_nifty_shariah_500_response():
    """Mock response for NiftyShariah500 universe endpoint."""
    return {
        "stocks": [
            {"symbol": f"SYM{i}.NS", "name": f"Company {i}", "industry": "IT"}
            for i in range(210)
        ],
        "source": "nifty_500_shariah",
        "symbol_suffix": ".NS",
        "total_stocks": 210,
        "fetched_at": "2026-03-01T00:00:00+00:00",
    }


@pytest.fixture
def mock_india_training_response():
    """Mock response for India PatchTST training endpoint."""
    return {
        "version": "v2026-03-01-india123",
        "data_window_start": "2015-01-01",
        "data_window_end": "2025-12-26",
        "metrics": {"train_loss": 0.01, "val_loss": 0.02, "baseline_loss": 0.05},
        "promoted": True,
        "prior_version": None,
        "num_input_channels": 5,
        "signals_used": ["ohlcv"],
    }


@pytest.fixture
def mock_halal_india_response():
    """Mock response for halal_india universe endpoint."""
    return {
        "stocks": [
            {
                "symbol": f"TOP{i}.NS",
                "predicted_weekly_return_pct": 5.0 - i * 0.3,
                "rank": i + 1,
                "model_version": "v2026-03-01-india123",
            }
            for i in range(15)
        ],
        "total_candidates": 180,
        "total_universe": 210,
        "filtered_insufficient_history": 30,
        "top_n": 15,
        "selection_method": "patchtst_forecast",
        "model_version": "v2026-03-01-india123",
        "symbol_suffix": ".NS",
        "fetched_at": "2026-03-01T00:00:00+00:00",
    }


@pytest.fixture
def mock_india_summary_response():
    """Mock response for India training summary endpoint."""
    return {
        "summary": {
            "para_1_overall": "India PatchTST training completed successfully.",
            "para_2_patchtst": "Model shows good performance on NSE data.",
            "para_3_recommendations": "Continue monitoring India pipeline.",
        },
        "provider": "openai",
        "model_used": "gpt-4o-mini",
        "tokens_used": 300,
    }


@pytest.fixture
def mock_india_email_response():
    """Mock response for India training summary email endpoint."""
    return {
        "is_success": True,
        "subject": "India Training Summary: 2015-01-01 to 2025-12-26",
        "body": "<html><body>India training summary</body></html>",
    }


class TestIndiaTrainingFlow:
    """Test the full India training flow with mocked tasks."""

    @patch("flows.india_weekly_training.send_india_training_email")
    @patch("flows.india_weekly_training.generate_india_training_summary")
    @patch("flows.india_weekly_training.fetch_halal_india_universe")
    @patch("flows.india_weekly_training.train_india_patchtst")
    @patch("flows.india_weekly_training.fetch_nifty_shariah_500_universe")
    def test_india_weekly_training_flow(
        self,
        mock_nifty,
        mock_train,
        mock_india_filtered,
        mock_summary,
        mock_email,
        mock_nifty_shariah_500_response,
        mock_india_training_response,
        mock_halal_india_response,
        mock_india_summary_response,
        mock_india_email_response,
    ):
        """Test full India training flow execution."""
        mock_nifty.return_value = mock_nifty_shariah_500_response
        mock_train.return_value = TrainingResponse(**mock_india_training_response)
        mock_india_filtered.return_value = mock_halal_india_response
        mock_summary.return_value = TrainingSummaryResponse(
            **mock_india_summary_response
        )
        mock_email.return_value = TrainingSummaryEmailResponse(
            **mock_india_email_response
        )

        result = india_weekly_training_flow()

        assert result["nifty_shariah_500"]["total_stocks"] == 210
        assert result["patchtst"]["version"] == "v2026-03-01-india123"
        assert result["patchtst"]["promoted"] is True
        assert result["halal_india"]["stocks"] == 15
        assert result["halal_india"]["selection_method"] == "patchtst_forecast"
        assert result["summary"]["provider"] == "openai"
        assert result["email"]["is_success"] is True
        assert "India" in result["email"]["subject"]

    @patch("flows.india_weekly_training.send_india_training_email")
    @patch("flows.india_weekly_training.generate_india_training_summary")
    @patch("flows.india_weekly_training.fetch_halal_india_universe")
    @patch("flows.india_weekly_training.train_india_patchtst")
    @patch("flows.india_weekly_training.fetch_nifty_shariah_500_universe")
    def test_india_training_task_order(
        self,
        mock_nifty,
        mock_train,
        mock_india_filtered,
        mock_summary,
        mock_email,
        mock_nifty_shariah_500_response,
        mock_india_training_response,
        mock_halal_india_response,
        mock_india_summary_response,
        mock_india_email_response,
    ):
        """Test task execution order matches the expected pipeline."""
        mock_nifty.return_value = mock_nifty_shariah_500_response
        mock_train.return_value = TrainingResponse(**mock_india_training_response)
        mock_india_filtered.return_value = mock_halal_india_response
        mock_summary.return_value = TrainingSummaryResponse(
            **mock_india_summary_response
        )
        mock_email.return_value = TrainingSummaryEmailResponse(
            **mock_india_email_response
        )

        india_weekly_training_flow()

        mock_nifty.assert_called_once()
        mock_train.assert_called_once()
        mock_india_filtered.assert_called_once()
        mock_summary.assert_called_once()
        mock_email.assert_called_once()

    @patch("flows.india_weekly_training.send_india_training_email")
    @patch("flows.india_weekly_training.generate_india_training_summary")
    @patch("flows.india_weekly_training.fetch_halal_india_universe")
    @patch("flows.india_weekly_training.train_india_patchtst")
    @patch("flows.india_weekly_training.fetch_nifty_shariah_500_universe")
    def test_india_training_summary_payload_has_only_patchtst(
        self,
        mock_nifty,
        mock_train,
        mock_india_filtered,
        mock_summary,
        mock_email,
        mock_nifty_shariah_500_response,
        mock_india_training_response,
        mock_halal_india_response,
        mock_india_summary_response,
        mock_india_email_response,
    ):
        """Test that summary task receives only PatchTST (no lstm/ppo/sac)."""
        mock_nifty.return_value = mock_nifty_shariah_500_response
        training_resp = TrainingResponse(**mock_india_training_response)
        mock_train.return_value = training_resp
        mock_india_filtered.return_value = mock_halal_india_response
        mock_summary.return_value = TrainingSummaryResponse(
            **mock_india_summary_response
        )
        mock_email.return_value = TrainingSummaryEmailResponse(
            **mock_india_email_response
        )

        india_weekly_training_flow()

        summary_call = mock_summary.call_args
        assert summary_call.kwargs["patchtst"] == training_resp

    @patch("flows.india_weekly_training.send_india_training_email")
    @patch("flows.india_weekly_training.generate_india_training_summary")
    @patch("flows.india_weekly_training.fetch_halal_india_universe")
    @patch("flows.india_weekly_training.train_india_patchtst")
    @patch("flows.india_weekly_training.fetch_nifty_shariah_500_universe")
    def test_india_training_email_payload_has_patchtst_and_summary(
        self,
        mock_nifty,
        mock_train,
        mock_india_filtered,
        mock_summary,
        mock_email,
        mock_nifty_shariah_500_response,
        mock_india_training_response,
        mock_halal_india_response,
        mock_india_summary_response,
        mock_india_email_response,
    ):
        """Test that email task receives both PatchTST and summary."""
        mock_nifty.return_value = mock_nifty_shariah_500_response
        training_resp = TrainingResponse(**mock_india_training_response)
        mock_train.return_value = training_resp
        mock_india_filtered.return_value = mock_halal_india_response
        summary_resp = TrainingSummaryResponse(**mock_india_summary_response)
        mock_summary.return_value = summary_resp
        mock_email.return_value = TrainingSummaryEmailResponse(
            **mock_india_email_response
        )

        india_weekly_training_flow()

        email_call = mock_email.call_args
        assert email_call.kwargs["patchtst"] == training_resp
        assert email_call.kwargs["summary"] == summary_resp


class TestIndiaTrainingModels:
    """Test that Pydantic models work for India PatchTST training."""

    def test_training_response_with_patchtst_fields(self, mock_india_training_response):
        """Test TrainingResponse works with India PatchTST fields."""
        response = TrainingResponse(**mock_india_training_response)
        assert response.version == "v2026-03-01-india123"
        assert response.promoted is True
        assert response.num_input_channels == 5
        assert response.signals_used == ["ohlcv"]
        assert response.metrics["train_loss"] == 0.01

    def test_training_response_without_lstm_ppo_sac_fields(
        self, mock_india_training_response
    ):
        """Test TrainingResponse works without symbols_used (India has no RL)."""
        response = TrainingResponse(**mock_india_training_response)
        assert response.symbols_used is None


class TestIndiaTrainingFailures:
    """Test failure propagation."""

    @patch("flows.india_weekly_training.send_india_training_email")
    @patch("flows.india_weekly_training.generate_india_training_summary")
    @patch("flows.india_weekly_training.fetch_halal_india_universe")
    @patch("flows.india_weekly_training.train_india_patchtst")
    @patch("flows.india_weekly_training.fetch_nifty_shariah_500_universe")
    def test_universe_failure_stops_flow(
        self,
        mock_nifty,
        mock_train,
        mock_india_filtered,
        mock_summary,
        mock_email,
    ):
        """If NiftyShariah500 fetch fails, training should not run."""
        mock_nifty.side_effect = Exception("NSE API down")

        with pytest.raises(Exception, match="NSE API down"):
            india_weekly_training_flow()

        mock_train.assert_not_called()
        mock_india_filtered.assert_not_called()
        mock_summary.assert_not_called()
        mock_email.assert_not_called()

    @patch("flows.india_weekly_training.send_india_training_email")
    @patch("flows.india_weekly_training.generate_india_training_summary")
    @patch("flows.india_weekly_training.fetch_halal_india_universe")
    @patch("flows.india_weekly_training.train_india_patchtst")
    @patch("flows.india_weekly_training.fetch_nifty_shariah_500_universe")
    def test_training_failure_stops_downstream(
        self,
        mock_nifty,
        mock_train,
        mock_india_filtered,
        mock_summary,
        mock_email,
        mock_nifty_shariah_500_response,
    ):
        """If India PatchTST training fails, filtered/summary/email should not run."""
        mock_nifty.return_value = mock_nifty_shariah_500_response
        mock_train.side_effect = Exception("Training failed: no price data")

        with pytest.raises(Exception, match="Training failed"):
            india_weekly_training_flow()

        mock_nifty.assert_called_once()
        mock_india_filtered.assert_not_called()
        mock_summary.assert_not_called()
        mock_email.assert_not_called()
