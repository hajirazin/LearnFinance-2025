"""Tests for email endpoints."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from brain_api.main import app
from brain_api.routes.email.gmail import GmailConfigError, get_gmail_config

client = TestClient(app)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_training_summary_email_request():
    """Valid request payload for training summary email endpoint."""
    return {
        "lstm": {
            "version": "v2026-01-15-abc123",
            "data_window_start": "2020-01-01",
            "data_window_end": "2025-12-31",
            "metrics": {"mae": 0.025, "rmse": 0.035},
            "promoted": True,
        },
        "patchtst": {
            "version": "v2026-01-15-def456",
            "data_window_start": "2020-01-01",
            "data_window_end": "2025-12-31",
            "metrics": {"mae": 0.020, "rmse": 0.030},
            "promoted": True,
            "num_input_channels": 5,
            "signals_used": ["ohlcv"],
        },
        "ppo": {
            "version": "v2026-01-15-ghi789",
            "data_window_start": "2020-01-01",
            "data_window_end": "2025-12-31",
            "metrics": {"sharpe": 1.5, "max_drawdown": 0.15},
            "promoted": True,
            "symbols_used": ["AAPL", "MSFT", "GOOGL"],
        },
        "sac": {
            "version": "v2026-01-15-jkl012",
            "data_window_start": "2020-01-01",
            "data_window_end": "2025-12-31",
            "metrics": {"sharpe": 1.8, "max_drawdown": 0.12},
            "promoted": False,
            "symbols_used": ["AAPL", "MSFT", "GOOGL"],
        },
        "summary": {
            "para_1_overall": "All models trained successfully with good metrics.",
            "para_2_lstm": "LSTM model shows strong price prediction capability.",
            "para_3_patchtst": "PatchTST leverages OHLCV approach effectively.",
            "para_4_ppo": "PPO allocator demonstrates solid risk-adjusted returns.",
            "para_5_sac": "SAC shows promising results but was not promoted.",
            "para_6_recommendations": "Consider investigating SAC promotion criteria.",
        },
    }


# =============================================================================
# Test Gmail Configuration
# =============================================================================


class TestGmailConfig:
    """Tests for Gmail configuration helper."""

    def test_get_gmail_config_success(self, monkeypatch):
        """Successfully get Gmail config from environment."""
        monkeypatch.setenv("GMAIL_USER", "test@gmail.com")
        monkeypatch.setenv("GMAIL_APP_PASSWORD", "test-password")
        monkeypatch.setenv("TRAINING_EMAIL_TO", "recipient@example.com")
        monkeypatch.setenv("TRAINING_EMAIL_CC", "cc1@example.com, cc2@example.com")

        config = get_gmail_config()

        assert config["user"] == "test@gmail.com"
        assert config["password"] == "test-password"
        assert config["to"] == "recipient@example.com"
        assert config["cc"] == ["cc1@example.com", "cc2@example.com"]

    def test_get_gmail_config_no_cc(self, monkeypatch):
        """Get Gmail config with empty CC."""
        monkeypatch.setenv("GMAIL_USER", "test@gmail.com")
        monkeypatch.setenv("GMAIL_APP_PASSWORD", "test-password")
        monkeypatch.setenv("TRAINING_EMAIL_TO", "recipient@example.com")
        monkeypatch.delenv("TRAINING_EMAIL_CC", raising=False)

        config = get_gmail_config()

        assert config["cc"] == []

    def test_missing_gmail_user_raises(self, monkeypatch):
        """Missing GMAIL_USER raises GmailConfigError."""
        monkeypatch.delenv("GMAIL_USER", raising=False)
        monkeypatch.setenv("GMAIL_APP_PASSWORD", "test-password")
        monkeypatch.setenv("TRAINING_EMAIL_TO", "recipient@example.com")

        with pytest.raises(GmailConfigError, match="GMAIL_USER"):
            get_gmail_config()

    def test_missing_gmail_password_raises(self, monkeypatch):
        """Missing GMAIL_APP_PASSWORD raises GmailConfigError."""
        monkeypatch.setenv("GMAIL_USER", "test@gmail.com")
        monkeypatch.delenv("GMAIL_APP_PASSWORD", raising=False)
        monkeypatch.setenv("TRAINING_EMAIL_TO", "recipient@example.com")

        with pytest.raises(GmailConfigError, match="GMAIL_APP_PASSWORD"):
            get_gmail_config()

    def test_missing_training_email_to_raises(self, monkeypatch):
        """Missing TRAINING_EMAIL_TO raises GmailConfigError."""
        monkeypatch.setenv("GMAIL_USER", "test@gmail.com")
        monkeypatch.setenv("GMAIL_APP_PASSWORD", "test-password")
        monkeypatch.delenv("TRAINING_EMAIL_TO", raising=False)

        with pytest.raises(GmailConfigError, match="TRAINING_EMAIL_TO"):
            get_gmail_config()


# =============================================================================
# Test Email Endpoint
# =============================================================================


class TestTrainingSummaryEmailEndpoint:
    """Tests for POST /email/training-summary endpoint."""

    @patch("brain_api.routes.email.training_summary.send_html_email")
    def test_successful_email_send(
        self,
        mock_send_email,
        mock_training_summary_email_request,
    ):
        """Successful training summary email send."""
        mock_send_email.return_value = True

        response = client.post(
            "/email/training-summary",
            json=mock_training_summary_email_request,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["is_success"] is True
        assert "Training Summary:" in data["subject"]
        assert "2020-01-01" in data["subject"]
        assert "2025-12-31" in data["subject"]
        assert len(data["body"]) > 0  # HTML body is returned
        mock_send_email.assert_called_once()

    @patch("brain_api.routes.email.training_summary.send_html_email")
    def test_email_body_contains_expected_sections(
        self,
        mock_send_email,
        mock_training_summary_email_request,
    ):
        """Email body contains all expected sections."""
        mock_send_email.return_value = True

        response = client.post(
            "/email/training-summary",
            json=mock_training_summary_email_request,
        )

        assert response.status_code == 200
        body = response.json()["body"]

        # Check header
        assert "Weekly Training Summary" in body
        assert "2020-01-01" in body

        # Check AI Analysis section
        assert "AI Analysis" in body
        assert "All models trained successfully" in body

        # Check RL Allocators section
        assert "RL Allocators Comparison" in body
        assert "PPO" in body
        assert "SAC" in body
        assert "v2026-01-15-ghi789" in body  # PPO version
        assert "v2026-01-15-jkl012" in body  # SAC version

        # Check Forecasters section
        assert "Forecasters Comparison" in body
        assert "LSTM" in body
        assert "PatchTST" in body

        # Check footer
        assert "LearnFinance-2025" in body

    @patch("brain_api.routes.email.training_summary.send_html_email")
    def test_gmail_config_error_returns_500(
        self,
        mock_send_email,
        mock_training_summary_email_request,
    ):
        """Gmail configuration error returns 500."""
        mock_send_email.side_effect = GmailConfigError("GMAIL_USER is required")

        response = client.post(
            "/email/training-summary",
            json=mock_training_summary_email_request,
        )

        assert response.status_code == 500
        assert "Gmail configuration error" in response.json()["detail"]

    @patch("brain_api.routes.email.training_summary.send_html_email")
    def test_smtp_error_returns_503(
        self,
        mock_send_email,
        mock_training_summary_email_request,
    ):
        """SMTP send error returns 503."""
        mock_send_email.side_effect = Exception("SMTP connection failed")

        response = client.post(
            "/email/training-summary",
            json=mock_training_summary_email_request,
        )

        assert response.status_code == 503
        assert "Failed to send email" in response.json()["detail"]

    def test_invalid_request_returns_422(self):
        """Invalid request body returns 422."""
        response = client.post(
            "/email/training-summary",
            json={"lstm": "invalid"},  # Should be an object
        )
        assert response.status_code == 422

    def test_missing_required_field_returns_422(self):
        """Missing required field returns 422."""
        response = client.post(
            "/email/training-summary",
            json={
                "lstm": {
                    "version": "v1",
                    "data_window_start": "2020-01-01",
                    "data_window_end": "2025-01-01",
                    "metrics": {},
                    "promoted": True,
                },
                # Missing patchtst, ppo, sac, summary
            },
        )
        assert response.status_code == 422

    @patch("brain_api.routes.email.training_summary.send_html_email")
    def test_empty_summary_still_works(
        self,
        mock_send_email,
        mock_training_summary_email_request,
    ):
        """Email with empty summary paragraphs still sends."""
        mock_send_email.return_value = True
        mock_training_summary_email_request["summary"] = {}

        response = client.post(
            "/email/training-summary",
            json=mock_training_summary_email_request,
        )

        assert response.status_code == 200
        assert response.json()["is_success"] is True

    @patch("brain_api.routes.email.training_summary.send_html_email")
    def test_model_not_promoted_shown_correctly(
        self,
        mock_send_email,
        mock_training_summary_email_request,
    ):
        """Models not promoted are shown correctly in email."""
        mock_send_email.return_value = True
        # SAC is already not promoted in fixture

        response = client.post(
            "/email/training-summary",
            json=mock_training_summary_email_request,
        )

        assert response.status_code == 200
        body = response.json()["body"]
        # The body should contain "No" for SAC promoted status
        # This is rendered as <span style="color: #c62828;">No</span>
        assert "No" in body


# =============================================================================
# Weekly Report Email Tests
# =============================================================================


@pytest.fixture
def mock_weekly_report_email_request():
    """Valid request payload for weekly report email endpoint."""
    return {
        "summary": {
            "para_1_overall_summary": "This week shows bullish momentum.",
            "para_2_sac": "SAC allocator favors tech stocks.",
            "para_3_ppo": "PPO takes a conservative approach.",
            "para_4_hrp_summary": "HRP maintains diversified allocation.",
            "para_5_patchtst_forecast": "PatchTST predicts positive returns.",
            "para_6_lstm_forecast": "LSTM shows bullish signals.",
            "para_7_news_sentiment": "News sentiment is positive.",
            "para_8_fundamentals": "Fundamentals remain strong.",
        },
        "order_results": {
            "ppo": {"orders_submitted": 5, "orders_failed": 0, "skipped": False},
            "sac": {"orders_submitted": 6, "orders_failed": 1, "skipped": False},
            "hrp": {"orders_submitted": 4, "orders_failed": 0, "skipped": False},
        },
        "skipped_algorithms": [],
        "target_week_start": "2026-02-03",
        "target_week_end": "2026-02-07",
        "as_of_date": "2026-02-03",
        "sac": {
            "target_weights": {"AAPL": 0.12, "MSFT": 0.10, "CASH": 0.05},
            "turnover": 0.15,
            "target_week_start": "2026-02-03",
            "target_week_end": "2026-02-07",
            "model_version": "v2026-01-15-sac001",
            "weight_changes": [],
        },
        "ppo": {
            "target_weights": {"AAPL": 0.11, "MSFT": 0.09, "CASH": 0.08},
            "turnover": 0.12,
            "target_week_start": "2026-02-03",
            "target_week_end": "2026-02-07",
            "model_version": "v2026-01-15-ppo001",
            "weight_changes": [],
        },
        "hrp": {
            "percentage_weights": {"AAPL": 10.5, "MSFT": 8.2, "GOOGL": 7.1},
            "symbols_used": 15,
            "symbols_excluded": [],
            "lookback_days": 252,
            "as_of_date": "2026-02-03",
        },
        "lstm": {
            "predictions": [
                {
                    "symbol": "AAPL",
                    "predicted_weekly_return_pct": 2.5,
                    "predicted_volatility": 0.03,
                    "direction": "UP",
                    "has_enough_history": True,
                    "history_days_used": 252,
                    "data_end_date": "2026-02-03",
                    "target_week_start": "2026-02-03",
                    "target_week_end": "2026-02-07",
                },
                {
                    "symbol": "MSFT",
                    "predicted_weekly_return_pct": 1.8,
                    "predicted_volatility": 0.025,
                    "direction": "UP",
                    "has_enough_history": True,
                    "history_days_used": 252,
                    "data_end_date": "2026-02-03",
                    "target_week_start": "2026-02-03",
                    "target_week_end": "2026-02-07",
                },
            ],
            "model_version": "v2026-01-15-lstm001",
            "as_of_date": "2026-02-03",
            "target_week_start": "2026-02-03",
            "target_week_end": "2026-02-07",
        },
        "patchtst": {
            "predictions": [
                {
                    "symbol": "AAPL",
                    "predicted_weekly_return_pct": 2.1,
                    "predicted_volatility": 0.028,
                    "direction": "UP",
                    "has_enough_history": True,
                    "history_days_used": 252,
                    "data_end_date": "2026-02-03",
                    "target_week_start": "2026-02-03",
                    "target_week_end": "2026-02-07",
                    "has_news_data": True,
                    "has_fundamentals_data": True,
                },
            ],
            "model_version": "v2026-01-15-patchtst001",
            "as_of_date": "2026-02-03",
            "target_week_start": "2026-02-03",
            "target_week_end": "2026-02-07",
            "signals_used": ["ohlcv"],
        },
    }


class TestWeeklyReportEmailEndpoint:
    """Tests for POST /email/weekly-report endpoint."""

    @patch("brain_api.routes.email.weekly_report.send_html_email")
    def test_successful_weekly_report_send(
        self,
        mock_send_email,
        mock_weekly_report_email_request,
    ):
        """Successful weekly report email send."""
        mock_send_email.return_value = True

        response = client.post(
            "/email/weekly-report",
            json=mock_weekly_report_email_request,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["is_success"] is True
        assert "Weekly Portfolio Analysis" in data["subject"]
        assert "2026-02-03" in data["subject"]
        assert "2026-02-07" in data["subject"]
        assert len(data["body"]) > 0
        mock_send_email.assert_called_once()

    @patch("brain_api.routes.email.weekly_report.send_html_email")
    def test_weekly_report_with_skipped_algorithms(
        self,
        mock_send_email,
        mock_weekly_report_email_request,
    ):
        """Weekly report with skipped algorithms shows warning."""
        mock_send_email.return_value = True
        mock_weekly_report_email_request["skipped_algorithms"] = ["PPO"]
        mock_weekly_report_email_request["order_results"]["ppo"]["skipped"] = True

        response = client.post(
            "/email/weekly-report",
            json=mock_weekly_report_email_request,
        )

        assert response.status_code == 200
        body = response.json()["body"]
        assert "Skipped Algorithms" in body
        assert "PPO" in body

    @patch("brain_api.routes.email.weekly_report.send_html_email")
    def test_weekly_report_smtp_failure(
        self,
        mock_send_email,
        mock_weekly_report_email_request,
    ):
        """SMTP send error returns 503."""
        mock_send_email.side_effect = Exception("SMTP connection failed")

        response = client.post(
            "/email/weekly-report",
            json=mock_weekly_report_email_request,
        )

        assert response.status_code == 503
        assert "Failed to send email" in response.json()["detail"]

    @patch("brain_api.routes.email.weekly_report.send_html_email")
    def test_weekly_report_gmail_config_error(
        self,
        mock_send_email,
        mock_weekly_report_email_request,
    ):
        """Gmail configuration error returns 500."""
        mock_send_email.side_effect = GmailConfigError("GMAIL_USER is required")

        response = client.post(
            "/email/weekly-report",
            json=mock_weekly_report_email_request,
        )

        assert response.status_code == 500
        assert "Gmail configuration error" in response.json()["detail"]

    @patch("brain_api.routes.email.weekly_report.send_html_email")
    def test_weekly_report_body_contains_expected_sections(
        self,
        mock_send_email,
        mock_weekly_report_email_request,
    ):
        """Email body contains all expected sections."""
        mock_send_email.return_value = True

        response = client.post(
            "/email/weekly-report",
            json=mock_weekly_report_email_request,
        )

        assert response.status_code == 200
        body = response.json()["body"]

        # Check header
        assert "Weekly Portfolio Analysis" in body
        assert "2026-02-03" in body

        # Check Order Execution Summary
        assert "Order Execution Summary" in body

        # Check AI Analysis section
        assert "AI Analysis Summary" in body
        assert "This week shows bullish momentum" in body

        # Check RL Allocators section
        assert "RL Allocations" in body
        assert "SAC" in body
        assert "PPO" in body

        # Check HRP section
        assert "HRP Allocation" in body

        # Check Forecasters section
        assert "Price Forecasts" in body
        assert "LSTM" in body
        assert "PatchTST" in body

        # Check footer
        assert "LearnFinance-2025" in body
