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
def mock_forecasters_email_request():
    """Valid request payload for /email/forecasters-training-summary."""
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
        "summary": {
            "para_1_overall": "Forecasters trained successfully with good metrics.",
            "para_2_lstm": "LSTM model shows strong price prediction capability.",
            "para_3_patchtst": "PatchTST leverages OHLCV approach effectively.",
            "para_4_recommendations": "Slate looks stable for the SAC retrain tomorrow.",
        },
    }


@pytest.fixture
def mock_sac_email_request():
    """Valid request payload for /email/sac-training-summary."""
    return {
        "sac": {
            "version": "v2026-01-15-jkl012",
            "data_window_start": "2020-01-01",
            "data_window_end": "2025-12-31",
            "metrics": {"sharpe": 1.8, "max_drawdown": 0.12},
            "promoted": False,
            "symbols_used": ["AAPL", "MSFT", "GOOGL"],
        },
        "summary": {
            "para_1_overall": "SAC training completed but did not clear the gate.",
            "para_2_metrics": "Sharpe 1.8 and 12% max drawdown are mediocre.",
            "para_3_recommendations": "Investigate SAC promotion criteria.",
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


class TestForecastersTrainingSummaryEmailEndpoint:
    """Tests for POST /email/forecasters-training-summary endpoint."""

    @patch("brain_api.routes.email.training_summary.send_html_email")
    def test_successful_email_send(
        self,
        mock_send_email,
        mock_forecasters_email_request,
    ):
        """Successful forecasters training summary email send."""
        mock_send_email.return_value = True

        response = client.post(
            "/email/forecasters-training-summary",
            json=mock_forecasters_email_request,
        )

        assert response.status_code == 200, response.text
        data = response.json()
        assert data["is_success"] is True
        assert "US Forecasters Training:" in data["subject"]
        assert "2020-01-01" in data["subject"]
        assert "2025-12-31" in data["subject"]
        assert len(data["body"]) > 0
        mock_send_email.assert_called_once()

    @patch("brain_api.routes.email.training_summary.send_html_email")
    def test_email_body_contains_expected_sections(
        self,
        mock_send_email,
        mock_forecasters_email_request,
    ):
        """Email body contains forecasters-only sections (no SAC)."""
        mock_send_email.return_value = True

        response = client.post(
            "/email/forecasters-training-summary",
            json=mock_forecasters_email_request,
        )

        assert response.status_code == 200
        body = response.json()["body"]

        assert "US Forecasters Training Summary" in body
        assert "2020-01-01" in body
        assert "AI Analysis" in body
        assert "Forecasters trained successfully" in body
        assert "Forecasters Comparison" in body
        assert "LSTM" in body
        assert "PatchTST" in body
        assert "v2026-01-15-abc123" in body  # LSTM version
        assert "v2026-01-15-def456" in body  # PatchTST version
        # SAC must not appear in the forecasters email.
        assert "SAC Allocator" not in body
        assert "v2026-01-15-jkl012" not in body
        assert "LearnFinance-2025" in body

    @patch("brain_api.routes.email.training_summary.send_html_email")
    def test_gmail_config_error_returns_500(
        self,
        mock_send_email,
        mock_forecasters_email_request,
    ):
        """Gmail configuration error returns 500."""
        mock_send_email.side_effect = GmailConfigError("GMAIL_USER is required")
        response = client.post(
            "/email/forecasters-training-summary",
            json=mock_forecasters_email_request,
        )
        assert response.status_code == 500
        assert "Gmail configuration error" in response.json()["detail"]

    @patch("brain_api.routes.email.training_summary.send_html_email")
    def test_smtp_error_returns_503(
        self,
        mock_send_email,
        mock_forecasters_email_request,
    ):
        """SMTP send error returns 503."""
        mock_send_email.side_effect = Exception("SMTP connection failed")
        response = client.post(
            "/email/forecasters-training-summary",
            json=mock_forecasters_email_request,
        )
        assert response.status_code == 503
        assert "Failed to send email" in response.json()["detail"]

    def test_invalid_request_returns_422(self):
        """Invalid request body returns 422."""
        response = client.post(
            "/email/forecasters-training-summary",
            json={"lstm": "invalid"},
        )
        assert response.status_code == 422

    def test_missing_patchtst_returns_422(self):
        """Missing patchtst field returns 422."""
        response = client.post(
            "/email/forecasters-training-summary",
            json={
                "lstm": {
                    "version": "v1",
                    "data_window_start": "2020-01-01",
                    "data_window_end": "2025-01-01",
                    "metrics": {},
                    "promoted": True,
                },
                "summary": {},
            },
        )
        assert response.status_code == 422

    @patch("brain_api.routes.email.training_summary.send_html_email")
    def test_empty_summary_still_works(
        self,
        mock_send_email,
        mock_forecasters_email_request,
    ):
        """Email with empty summary paragraphs still sends."""
        mock_send_email.return_value = True
        mock_forecasters_email_request["summary"] = {}
        response = client.post(
            "/email/forecasters-training-summary",
            json=mock_forecasters_email_request,
        )
        assert response.status_code == 200
        assert response.json()["is_success"] is True

    @patch("brain_api.routes.email.training_summary.send_html_email")
    def test_promoted_renders_guardrail_pass_prose(
        self,
        mock_send_email,
        mock_forecasters_email_request,
    ):
        """A promoted forecaster renders the guardrail-pass prose."""
        mock_send_email.return_value = True
        response = client.post(
            "/email/forecasters-training-summary",
            json=mock_forecasters_email_request,
        )
        assert response.status_code == 200
        body = response.json()["body"]
        assert "Passed all artifact health guardrails." in body
        assert "Failed Guardrails" not in body

    @patch("brain_api.routes.email.training_summary.send_html_email")
    def test_not_promoted_renders_failure_reasons(
        self,
        mock_send_email,
        mock_forecasters_email_request,
    ):
        """A non-promoted forecaster renders bulleted failure_reasons."""
        mock_send_email.return_value = True
        mock_forecasters_email_request["lstm"]["promoted"] = False
        mock_forecasters_email_request["lstm"]["failure_reasons"] = [
            "val_loss is not finite",
            "weights.pt missing or empty",
        ]
        response = client.post(
            "/email/forecasters-training-summary",
            json=mock_forecasters_email_request,
        )
        assert response.status_code == 200
        body = response.json()["body"]
        assert "Failed Guardrails" in body
        assert "val_loss is not finite" in body
        assert "weights.pt missing or empty" in body


class TestSACTrainingSummaryEmailEndpoint:
    """Tests for POST /email/sac-training-summary endpoint."""

    @patch("brain_api.routes.email.training_summary.send_html_email")
    def test_successful_email_send(
        self,
        mock_send_email,
        mock_sac_email_request,
    ):
        """Successful SAC training summary email send.

        Default ``universe`` is ``halal_filtered`` (backward-compat for
        the legacy single-bucket workflow), which is rendered into the
        subject as ``US SAC (halal_filtered) Training: ...``.
        """
        mock_send_email.return_value = True

        response = client.post(
            "/email/sac-training-summary",
            json=mock_sac_email_request,
        )

        assert response.status_code == 200, response.text
        data = response.json()
        assert data["is_success"] is True
        assert "US SAC (halal_filtered) Training:" in data["subject"]
        assert "2020-01-01" in data["subject"]
        assert "2025-12-31" in data["subject"]
        assert len(data["body"]) > 0
        mock_send_email.assert_called_once()

    @patch("brain_api.routes.email.training_summary.send_html_email")
    def test_universe_halal_in_subject_and_body(
        self,
        mock_send_email,
        mock_sac_email_request,
    ):
        """Posting ``universe="halal"`` produces a halal-tagged email.

        Two parallel A/B SAC workflows share this endpoint; the
        ``universe`` field is rendered into both the subject and the
        body header so a human inbox reader can immediately tell the
        two reports apart without opening them.
        """
        mock_send_email.return_value = True
        mock_sac_email_request["universe"] = "halal"

        response = client.post(
            "/email/sac-training-summary",
            json=mock_sac_email_request,
        )

        assert response.status_code == 200, response.text
        data = response.json()
        assert "US SAC (halal) Training:" in data["subject"]
        assert "halal_filtered" not in data["subject"]
        assert "sac_halal" in data["body"]

    @patch("brain_api.routes.email.training_summary.send_html_email")
    def test_email_body_contains_expected_sections(
        self,
        mock_send_email,
        mock_sac_email_request,
    ):
        """Email body contains SAC-only sections (no LSTM/PatchTST)."""
        mock_send_email.return_value = True

        response = client.post(
            "/email/sac-training-summary",
            json=mock_sac_email_request,
        )

        assert response.status_code == 200
        body = response.json()["body"]

        assert "US SAC Training Summary" in body
        assert "AI Analysis" in body
        assert "SAC Allocator" in body
        assert "v2026-01-15-jkl012" in body  # SAC version
        # SAC was not promoted -- "No" should appear.
        assert "No" in body
        # Forecasters should not show up in the SAC email.
        assert "Forecasters Comparison" not in body
        assert "v2026-01-15-abc123" not in body  # LSTM version
        assert "v2026-01-15-def456" not in body  # PatchTST version
        assert "LearnFinance-2025" in body

    @patch("brain_api.routes.email.training_summary.send_html_email")
    def test_gmail_config_error_returns_500(
        self,
        mock_send_email,
        mock_sac_email_request,
    ):
        """Gmail configuration error returns 500."""
        mock_send_email.side_effect = GmailConfigError("GMAIL_USER is required")
        response = client.post(
            "/email/sac-training-summary",
            json=mock_sac_email_request,
        )
        assert response.status_code == 500
        assert "Gmail configuration error" in response.json()["detail"]

    @patch("brain_api.routes.email.training_summary.send_html_email")
    def test_smtp_error_returns_503(
        self,
        mock_send_email,
        mock_sac_email_request,
    ):
        """SMTP send error returns 503."""
        mock_send_email.side_effect = Exception("SMTP connection failed")
        response = client.post(
            "/email/sac-training-summary",
            json=mock_sac_email_request,
        )
        assert response.status_code == 503
        assert "Failed to send email" in response.json()["detail"]

    def test_missing_sac_returns_422(self):
        """Empty body fails validation (sac is required)."""
        response = client.post("/email/sac-training-summary", json={})
        assert response.status_code == 422

    @patch("brain_api.routes.email.training_summary.send_html_email")
    def test_empty_summary_still_works(
        self,
        mock_send_email,
        mock_sac_email_request,
    ):
        """Email with empty summary paragraphs still sends."""
        mock_send_email.return_value = True
        mock_sac_email_request["summary"] = {}
        response = client.post(
            "/email/sac-training-summary",
            json=mock_sac_email_request,
        )
        assert response.status_code == 200
        assert response.json()["is_success"] is True

    @patch("brain_api.routes.email.training_summary.send_html_email")
    def test_promoted_renders_guardrail_pass_prose(
        self,
        mock_send_email,
        mock_sac_email_request,
    ):
        """A promoted SAC run renders the guardrail-pass prose."""
        mock_send_email.return_value = True
        mock_sac_email_request["sac"]["promoted"] = True
        mock_sac_email_request["sac"]["failure_reasons"] = []
        response = client.post(
            "/email/sac-training-summary",
            json=mock_sac_email_request,
        )
        assert response.status_code == 200
        body = response.json()["body"]
        assert "Passed all SAC artifact health guardrails." in body
        assert "Failed Guardrails" not in body

    @patch("brain_api.routes.email.training_summary.send_html_email")
    def test_not_promoted_renders_failure_reasons(
        self,
        mock_send_email,
        mock_sac_email_request,
    ):
        """A non-promoted SAC run renders bulleted failure_reasons."""
        mock_send_email.return_value = True
        mock_sac_email_request["sac"]["promoted"] = False
        mock_sac_email_request["sac"]["failure_reasons"] = [
            "eval_cagr 0.10 below floor 0.12",
            "actor.pt missing or empty",
        ]
        response = client.post(
            "/email/sac-training-summary",
            json=mock_sac_email_request,
        )
        assert response.status_code == 200
        body = response.json()["body"]
        assert "Failed Guardrails" in body
        assert "eval_cagr 0.10 below floor 0.12" in body
        assert "actor.pt missing or empty" in body


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
            "para_3_patchtst_forecast": "PatchTST predicts positive returns.",
            "para_4_news_sentiment": "News sentiment is positive.",
        },
        "order_results": {
            "sac": {"orders_submitted": 6, "orders_failed": 1, "skipped": False},
        },
        "skipped_algorithms": [],
        "target_week_start": "2026-02-03",
        "target_week_end": "2026-02-07",
        "as_of_date": "2026-02-03",
        "universe": "halal_filtered",
        "sac": {
            "target_weights": {"AAPL": 0.12, "MSFT": 0.10, "CASH": 0.05},
            "turnover": 0.15,
            "target_week_start": "2026-02-03",
            "target_week_end": "2026-02-07",
            "model_version": "v2026-01-15-sac001",
            "weight_changes": [],
            "asset_eligibility": {"AAPL": True, "MSFT": True},
            "regime_posterior": [0.7, 0.2, 0.1],
            "sac_schema_version": 3,
            "architecture": "masked_attention",
        },
        "patchtst": {
            "predictions": [
                {
                    "symbol": "AAPL",
                    "predicted_weekly_return_pct": 2.1,
                    "direction": "UP",
                    "has_enough_history": True,
                    "history_days_used": 252,
                    "data_end_date": "2026-02-03",
                    "target_week_start": "2026-02-03",
                    "target_week_end": "2026-02-07",
                },
            ],
            "model_version": "v2026-01-15-patchtst001",
            "as_of_date": "2026-02-03",
            "target_week_start": "2026-02-03",
            "target_week_end": "2026-02-07",
            "signals_used": ["ohlcv"],
        },
    }


# =============================================================================
# India Alpha-HRP Report Email Tests
# =============================================================================


@pytest.fixture
def mock_india_weekly_report_email_request():
    """Valid request payload for India Alpha-HRP report email endpoint.

    Mirrors the shape of the US Alpha-HRP request fixture (Stage 1
    top-30 + sticky stats + Stage 2 HRP). India does not trade through
    Alpaca so no ``order_results`` / ``skipped`` fields are sent --
    those defaults to ``None`` / ``False`` on the shared
    :class:`AlphaHRPEmailRequest` base.
    """
    return {
        "summary": {
            "para_1_market_outlook": "Top 30 PatchTST forecasts cluster around IT services.",
            "para_2_selection_rationale": "Sticky kept 12 NSE names; three new high-rank entrants.",
            "para_3_final_allocation": "HRP weights RELIANCE.NS=7.0%, TCS.NS=6.8%.",
            "para_4_risk_observations": "Watch INR/USD risk and small-cap NSE liquidity.",
            "para_5_stage_transition_insight": "TCS.NS jumped from alpha rank 12 to HRP weight rank 1 due to low correlation with NSE IT peers.",
        },
        "stage1_top_scores": [
            {"symbol": f"NSE{i:03d}.NS", "score": 5.0 - 0.1 * i, "rank": i + 1}
            for i in range(20)
        ],
        "model_version": "v2026-04-26-india",
        "predicted_count": 200,
        "requested_count": 210,
        "selected_symbols": [f"NSE{i:03d}.NS" for i in range(15)],
        "kept_count": 12,
        "fillers_count": 3,
        "evicted_from_previous": {"OLD1.NS": "rank_out_of_hold"},
        "previous_year_week_used": "202617",
        "stage2": {
            "percentage_weights": {f"NSE{i:03d}.NS": 100.0 / 15 for i in range(15)},
            "symbols_used": 15,
            "symbols_excluded": [],
            "lookback_days": 252,
            "as_of_date": "2026-04-28",
        },
        "universe": "halal_india_alpha",
        "top_n": 15,
        "hold_threshold": 20,
        "target_week_start": "2026-03-02",
        "target_week_end": "2026-03-06",
        "as_of_date": "2026-03-02",
    }


class TestIndiaAlphaHRPReportEmailEndpoint:
    """Tests for POST /email/india-alpha-hrp-report endpoint."""

    @patch("brain_api.routes.email.weekly_report.send_html_email")
    def test_successful_india_report_send(
        self,
        mock_send_email,
        mock_india_weekly_report_email_request,
    ):
        """Successful India Alpha-HRP report email send."""
        mock_send_email.return_value = True

        response = client.post(
            "/email/india-alpha-hrp-report",
            json=mock_india_weekly_report_email_request,
        )

        assert response.status_code == 200, response.text
        data = response.json()
        assert data["is_success"] is True
        assert "India Alpha-HRP Portfolio Analysis" in data["subject"]
        assert "2026-03-02" in data["subject"]
        assert "2026-03-06" in data["subject"]
        assert len(data["body"]) > 0
        mock_send_email.assert_called_once()

    @patch("brain_api.routes.email.weekly_report.send_html_email")
    def test_india_report_body_contains_alpha_hrp_sections(
        self,
        mock_send_email,
        mock_india_weekly_report_email_request,
    ):
        """India email body contains the same Stage 1 + sticky + Stage 2 sections US has."""
        mock_send_email.return_value = True

        response = client.post(
            "/email/india-alpha-hrp-report",
            json=mock_india_weekly_report_email_request,
        )

        assert response.status_code == 200
        body = response.json()["body"]

        assert "India Alpha-HRP Portfolio Analysis (NSE)" in body
        assert "AI Analysis Summary" in body
        # 4-paragraph schema rendered.
        assert "Market Outlook" in body
        assert "Selection Rationale" in body
        # Stage 1 alpha screen with top-30 context.
        assert "Stage 1: Alpha Screen" in body
        # Sticky stats present.
        assert "Rank-band Sticky Selection" in body
        # Stage 2 HRP allocation table.
        assert "Stage 2: HRP Allocation" in body
        # Selected symbols rendered (first one from fixture).
        assert "NSE000.NS" in body
        # Sticky kept count from fixture.
        assert "12" in body  # kept_count
        # NSE Paper-only banner instead of Alpaca.
        assert "NSE Paper-only Reporting" in body

    @patch("brain_api.routes.email.weekly_report.send_html_email")
    def test_india_report_body_does_not_contain_us_only_sections(
        self,
        mock_send_email,
        mock_india_weekly_report_email_request,
    ):
        """India email body does NOT contain US-only blocks."""
        mock_send_email.return_value = True

        response = client.post(
            "/email/india-alpha-hrp-report",
            json=mock_india_weekly_report_email_request,
        )

        assert response.status_code == 200
        body = response.json()["body"]

        # India does not trade -> no Alpaca order execution heading.
        assert "Alpaca Order Execution" not in body
        # India does not have an open-orders skip path.
        assert "Run Skipped" not in body

    @patch("brain_api.routes.email.weekly_report.send_html_email")
    def test_india_report_smtp_failure(
        self,
        mock_send_email,
        mock_india_weekly_report_email_request,
    ):
        """SMTP send error returns 503."""
        mock_send_email.side_effect = Exception("SMTP connection failed")

        response = client.post(
            "/email/india-alpha-hrp-report",
            json=mock_india_weekly_report_email_request,
        )

        assert response.status_code == 503
        assert "Failed to send email" in response.json()["detail"]

    @patch("brain_api.routes.email.weekly_report.send_html_email")
    def test_india_report_gmail_config_error(
        self,
        mock_send_email,
        mock_india_weekly_report_email_request,
    ):
        """Gmail configuration error returns 500."""
        mock_send_email.side_effect = GmailConfigError("GMAIL_USER is required")

        response = client.post(
            "/email/india-alpha-hrp-report",
            json=mock_india_weekly_report_email_request,
        )

        assert response.status_code == 500
        assert "Gmail configuration error" in response.json()["detail"]

    def test_india_report_missing_required_fields_returns_422(self):
        """Empty body fails Pydantic validation."""
        response = client.post(
            "/email/india-alpha-hrp-report",
            json={},
        )
        assert response.status_code == 422

    def test_india_report_missing_stage2_returns_422(
        self, mock_india_weekly_report_email_request
    ):
        """Missing stage2 field returns 422."""
        payload = dict(mock_india_weekly_report_email_request)
        del payload["stage2"]
        response = client.post("/email/india-alpha-hrp-report", json=payload)
        assert response.status_code == 422

    @patch("brain_api.routes.email.weekly_report.send_html_email")
    def test_india_report_stage2_renders_stop_loss_from_paper_allocation(
        self,
        mock_send_email,
        mock_india_weekly_report_email_request,
    ):
        """Stage 2 Stop Loss column renders atr14 and atr_unavailable rows."""
        mock_send_email.return_value = True
        payload = dict(mock_india_weekly_report_email_request)
        payload["paper_allocation"] = {
            "details": [
                {
                    "symbol": "NSE000.NS",
                    "weight_pct": 100.0 / 15,
                    "price": 100.0,
                    "whole_shares": 66,
                    "trade_value": 6600.0,
                    "stop_loss_price": 94.0,
                    "stop_loss_distance_pct": 0.06,
                    "stop_loss_reason": "atr14",
                },
                {
                    "symbol": "NSE001.NS",
                    "weight_pct": 100.0 / 15,
                    "price": 200.0,
                    "whole_shares": 33,
                    "trade_value": 6600.0,
                    "stop_loss_price": None,
                    "stop_loss_distance_pct": None,
                    "stop_loss_reason": "atr_unavailable",
                },
            ],
            "total_nav": 100_000.0,
            "prices_used": {"NSE000.NS": 100.0, "NSE001.NS": 200.0},
            "total_allocated_pct": 200.0 / 15,
        }

        response = client.post("/email/india-alpha-hrp-report", json=payload)

        assert response.status_code == 200, response.text
        body = response.json()["body"]
        assert "Stop Loss" in body
        assert "94.00" in body
        assert "(-6.0%)" in body
        assert "n/a (no ATR)" in body
        assert "ATR(14)" in body


class TestSACWeeklyReportEmailEndpoint:
    """Tests for POST /email/sac-weekly-report endpoint."""

    @patch("brain_api.routes.email.weekly_report.send_html_email")
    def test_successful_weekly_report_send(
        self,
        mock_send_email,
        mock_weekly_report_email_request,
    ):
        """Successful SAC weekly report email send for halal_filtered."""
        mock_send_email.return_value = True

        response = client.post(
            "/email/sac-weekly-report",
            json=mock_weekly_report_email_request,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["is_success"] is True
        # Subject is now universe-tagged (mandatory) so the two A/B
        # SAC weekly emails self-identify in the inbox.
        assert "US SAC (halal_filtered) Weekly Portfolio Analysis" in data["subject"]
        assert "2026-02-03" in data["subject"]
        assert "2026-02-07" in data["subject"]
        assert len(data["body"]) > 0
        mock_send_email.assert_called_once()

    @patch("brain_api.routes.email.weekly_report.send_html_email")
    def test_weekly_report_universe_halal_renders_in_subject_and_body(
        self,
        mock_send_email,
        mock_weekly_report_email_request,
    ):
        """universe='halal' must reach both the subject and the email header."""
        mock_send_email.return_value = True
        payload = dict(mock_weekly_report_email_request)
        payload["universe"] = "halal"

        response = client.post("/email/sac-weekly-report", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert "US SAC (halal) Weekly Portfolio Analysis" in data["subject"]
        # The HTML email body header also renders the universe so the
        # human reader can tell the two A/B reports apart even before
        # reading the subject.
        assert "US SAC (halal) Weekly Portfolio Analysis" in data["body"]

    def test_weekly_report_missing_universe_returns_422(
        self, mock_weekly_report_email_request
    ):
        """Universe is mandatory: 422 when omitted (AGENTS.md no-default)."""
        payload = dict(mock_weekly_report_email_request)
        payload.pop("universe")

        response = client.post("/email/sac-weekly-report", json=payload)

        assert response.status_code == 422
        body = response.json()
        assert any("universe" in str(loc) for loc in body.get("detail", []))

    @patch("brain_api.routes.email.weekly_report.send_html_email")
    def test_weekly_report_with_skipped_algorithms(
        self,
        mock_send_email,
        mock_weekly_report_email_request,
    ):
        """SAC weekly report with skipped algorithms shows warning."""
        mock_send_email.return_value = True
        mock_weekly_report_email_request["skipped_algorithms"] = ["SAC"]
        mock_weekly_report_email_request["order_results"]["sac"]["skipped"] = True

        response = client.post(
            "/email/sac-weekly-report",
            json=mock_weekly_report_email_request,
        )

        assert response.status_code == 200
        body = response.json()["body"]
        assert "Skipped Algorithms" in body
        assert "SAC" in body

    @patch("brain_api.routes.email.weekly_report.send_html_email")
    def test_weekly_report_smtp_failure(
        self,
        mock_send_email,
        mock_weekly_report_email_request,
    ):
        """SMTP send error returns 503."""
        mock_send_email.side_effect = Exception("SMTP connection failed")

        response = client.post(
            "/email/sac-weekly-report",
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
            "/email/sac-weekly-report",
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
        """Email body contains all expected SAC-only sections."""
        mock_send_email.return_value = True

        response = client.post(
            "/email/sac-weekly-report",
            json=mock_weekly_report_email_request,
        )

        assert response.status_code == 200
        body = response.json()["body"]

        # Check header (universe-tagged form)
        assert "US SAC (halal_filtered) Weekly Portfolio Analysis" in body
        assert "2026-02-03" in body

        # Check Order Execution Summary
        assert "Order Execution Summary" in body

        # Check AI Analysis section
        assert "AI Analysis Summary" in body
        assert "This week shows bullish momentum" in body

        # Check SAC Allocation section
        assert "SAC Allocation" in body
        assert "SAC" in body

        # Check Forecasters section
        assert "Price Forecasts" in body
        assert "LSTM" not in body
        assert "PatchTST" in body

        # Check footer
        assert "LearnFinance-2025" in body

    @patch("brain_api.routes.email.weekly_report.send_html_email")
    def test_weekly_report_with_per_order_detail_table(
        self,
        mock_send_email,
        mock_weekly_report_email_request,
    ):
        """Detailed order table renders when ``orders`` list is populated."""
        mock_send_email.return_value = True
        mock_weekly_report_email_request["order_results"]["sac"]["orders"] = [
            {
                "symbol": "AAPL",
                "side": "buy",
                "qty": 12.0,
                "current_price": 200.0,
                "trade_value": 2400.0,
                "stop_loss_price": 188.0,
                "stop_loss_distance_pct": 0.06,
                "stop_loss_reason": "atr14",
                "client_order_id": "paper:2026-02-03:attempt-1:AAPL:buy",
                "submission_status": "submitted",
            },
        ]
        response = client.post(
            "/email/sac-weekly-report",
            json=mock_weekly_report_email_request,
        )
        assert response.status_code == 200
        body = response.json()["body"]
        assert "Order Execution Detail" in body
        assert "AAPL" in body
        assert "$188.00" in body

    @patch("brain_api.routes.email.weekly_report.send_html_email")
    def test_weekly_report_with_prior_allocation_block(
        self,
        mock_send_email,
        mock_weekly_report_email_request,
    ):
        """Prior allocation block renders the live broker label for US SAC."""
        mock_send_email.return_value = True
        mock_weekly_report_email_request["prior_allocation"] = {
            "weights": {"AAPL": 0.10, "MSFT": 0.08, "CASH": 0.82},
            "source_label": "live Alpaca account: sac",
            "as_of": "2026-01-27",
        }
        response = client.post(
            "/email/sac-weekly-report",
            json=mock_weekly_report_email_request,
        )
        assert response.status_code == 200
        body = response.json()["body"]
        assert "Going Into This Week" in body
        assert "live Alpaca account: sac" in body


# =============================================================================
# US Double HRP Report Email Tests
# =============================================================================


@pytest.fixture
def mock_us_double_hrp_email_request():
    """Valid request payload for /email/us-double-hrp-report."""
    return {
        "summary": {
            "para_1_screening_overview": "HRP screened 410 halal_new stocks.",
            "para_2_selection_rationale": "Top 15 are tech-heavy.",
            "para_3_final_allocation": "Stage 2 distributes evenly.",
            "para_4_risk_observations": "Watch sector concentration.",
            "para_5_stage_transition_insight": "NVDA jumped from Stage 1 weight rank 12 to Stage 2 rank 1 due to low correlation in the chosen 15.",
        },
        "stage1": {
            "percentage_weights": {f"S{i:03d}": 0.5 for i in range(20)},
            "symbols_used": 20,
            "symbols_excluded": [],
            "lookback_days": 756,
            "as_of_date": "2026-02-23",
        },
        "stage2": {
            "percentage_weights": {f"S{i:03d}": 100.0 / 15 for i in range(15)},
            "symbols_used": 15,
            "symbols_excluded": [],
            "lookback_days": 252,
            "as_of_date": "2026-02-23",
        },
        "universe": "halal_new",
        "top_n": 15,
        "target_week_start": "2026-02-23",
        "target_week_end": "2026-02-27",
        "as_of_date": "2026-02-23",
        "order_results": {
            "orders_submitted": 14,
            "orders_failed": 1,
            "skipped": False,
        },
        "skipped": False,
        "kept_count": 12,
        "fillers_count": 3,
        "previous_year_week_used": "202608",
        "stickiness_threshold_pp": 1.0,
    }


class TestUSDoubleHRPReportEmailEndpoint:
    """Tests for POST /email/us-double-hrp-report endpoint."""

    @patch("brain_api.routes.email.weekly_report.send_html_email")
    def test_happy_path_with_orders(
        self,
        mock_send_email,
        mock_us_double_hrp_email_request,
    ):
        mock_send_email.return_value = True
        response = client.post(
            "/email/us-double-hrp-report",
            json=mock_us_double_hrp_email_request,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_success"] is True
        assert "US Double HRP Portfolio Analysis" in data["subject"]
        assert "2026-02-23" in data["subject"]

        body = data["body"]
        assert "AI Analysis Summary" in body
        assert "Stage 1: Screening" in body
        assert "Stage 2: Final Allocation" in body
        assert "Alpaca Order Execution" in body
        assert "Sticky Selection" in body
        assert "halal_new" in body
        assert "14" in body
        assert "Run Skipped" not in body

    @patch("brain_api.routes.email.weekly_report.send_html_email")
    def test_skipped_path_hides_orders(
        self,
        mock_send_email,
        mock_us_double_hrp_email_request,
    ):
        mock_send_email.return_value = True
        mock_us_double_hrp_email_request["skipped"] = True
        response = client.post(
            "/email/us-double-hrp-report",
            json=mock_us_double_hrp_email_request,
        )
        assert response.status_code == 200
        data = response.json()
        assert "US Double HRP Skipped" in data["subject"]
        body = data["body"]
        assert "Run Skipped" in body
        assert "AI Analysis Summary" not in body
        assert "Alpaca Order Execution" not in body

    @patch("brain_api.routes.email.weekly_report.send_html_email")
    def test_no_order_results_renders(
        self,
        mock_send_email,
        mock_us_double_hrp_email_request,
    ):
        mock_send_email.return_value = True
        mock_us_double_hrp_email_request["order_results"] = None
        response = client.post(
            "/email/us-double-hrp-report",
            json=mock_us_double_hrp_email_request,
        )
        assert response.status_code == 200
        body = response.json()["body"]
        assert "Stage 2: Final Allocation" in body
        assert "Alpaca Order Execution" not in body

    @patch("brain_api.routes.email.weekly_report.send_html_email")
    def test_smtp_failure_returns_503(
        self,
        mock_send_email,
        mock_us_double_hrp_email_request,
    ):
        mock_send_email.side_effect = Exception("SMTP down")
        response = client.post(
            "/email/us-double-hrp-report",
            json=mock_us_double_hrp_email_request,
        )
        assert response.status_code == 503
        assert "Failed to send email" in response.json()["detail"]

    @patch("brain_api.routes.email.weekly_report.send_html_email")
    def test_gmail_config_error_returns_500(
        self,
        mock_send_email,
        mock_us_double_hrp_email_request,
    ):
        mock_send_email.side_effect = GmailConfigError("GMAIL_USER is required")
        response = client.post(
            "/email/us-double-hrp-report",
            json=mock_us_double_hrp_email_request,
        )
        assert response.status_code == 500

    def test_missing_required_field_returns_422(self):
        response = client.post(
            "/email/us-double-hrp-report",
            json={
                "summary": {"para_1": "x"},
                "universe": "halal_new",
                "top_n": 15,
                "target_week_start": "2026-02-23",
                "target_week_end": "2026-02-27",
                "as_of_date": "2026-02-23",
            },
        )
        assert response.status_code == 422


# =============================================================================
# India Double HRP Report Email Tests
# =============================================================================


@pytest.fixture
def mock_india_double_hrp_email_request():
    """Valid request payload for /email/india-double-hrp-report.

    Mirrors the US fixture's shape because both endpoints share the
    DoubleHRPEmailRequest base. India omits ``order_results`` /
    ``skipped`` -- they are US-only fields on USDoubleHRPEmailRequest.
    """
    return {
        "summary": {
            "para_1_screening_overview": "HRP screened 210 NSE Shariah stocks.",
            "para_2_selection_rationale": "Top 15 lean towards IT services.",
            "para_3_final_allocation": "Stage 2 spreads weight broadly.",
            "para_4_risk_observations": "Watch sector concentration.",
            "para_5_stage_transition_insight": "ONGC.NS jumped from Stage 1 weight rank 12 to Stage 2 rank 1 because the 252-day covariance over the chosen 15 isolated it from energy peers.",
        },
        "stage1": {
            "percentage_weights": {f"S{i:03d}.NS": 0.5 for i in range(20)},
            "symbols_used": 20,
            "symbols_excluded": [],
            "lookback_days": 756,
            "as_of_date": "2026-02-23",
        },
        "stage2": {
            "percentage_weights": {f"S{i:03d}.NS": 100.0 / 15 for i in range(15)},
            "symbols_used": 15,
            "symbols_excluded": [],
            "lookback_days": 252,
            "as_of_date": "2026-02-23",
        },
        "universe": "halal_india_double_hrp",
        "top_n": 15,
        "target_week_start": "2026-02-23",
        "target_week_end": "2026-02-27",
        "as_of_date": "2026-02-23",
        "kept_count": 12,
        "fillers_count": 3,
        "previous_year_week_used": "202608",
        "stickiness_threshold_pp": 1.0,
    }


class TestIndiaDoubleHRPReportEmailEndpoint:
    """Tests for POST /email/india-double-hrp-report endpoint.

    Cross-checks shared-base parity with US Double HRP -- both should
    render the same Stage 1 + Sticky + Stage 2 sections from
    ``double_hrp_email_base.html.j2``. India-specific differences:

    * No "Alpaca Order Execution" block (paper-only, no broker).
    * No "Run Skipped" block (no open-orders gate).
    * Footer says "Paper-only, no broker" instead of "Alpaca Paper
      Trading".
    """

    @patch("brain_api.routes.email.weekly_report.send_html_email")
    def test_happy_path_renders_shared_sections(
        self,
        mock_send_email,
        mock_india_double_hrp_email_request,
    ):
        mock_send_email.return_value = True
        response = client.post(
            "/email/india-double-hrp-report",
            json=mock_india_double_hrp_email_request,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_success"] is True
        assert "India Double HRP Portfolio Analysis" in data["subject"]

        body = data["body"]
        # Math-aware sections come from the shared base, so they MUST
        # appear identically across markets.
        assert "AI Analysis Summary" in body
        assert "Stage 1: Screening" in body
        assert "Stage 2: Final Allocation" in body
        assert "Weight-band Sticky Selection" in body
        # Universe label is rendered verbatim -- partition string is
        # acceptable here, mirrors India Alpha-HRP convention.
        assert "halal_india_double_hrp" in body
        # Sticky stats from the request must round-trip into the email.
        assert ">12<" in body or "<strong>12</strong>" in body
        assert ">3<" in body or "<strong>3</strong>" in body
        assert "202608" in body

    @patch("brain_api.routes.email.weekly_report.send_html_email")
    def test_no_alpaca_or_skipped_blocks_render(
        self,
        mock_send_email,
        mock_india_double_hrp_email_request,
    ):
        mock_send_email.return_value = True
        response = client.post(
            "/email/india-double-hrp-report",
            json=mock_india_double_hrp_email_request,
        )
        body = response.json()["body"]
        # India does not trade -- the order-execution and skipped
        # blocks must be absent; otherwise the email lies about Alpaca.
        assert "Alpaca Order Execution" not in body
        assert "Run Skipped" not in body
        # Footer signals paper-only NSE India, not Alpaca.
        assert "Paper-only, no broker" in body
        assert "Alpaca Paper Trading" not in body

    @patch("brain_api.routes.email.weekly_report.send_html_email")
    def test_stage1_renders_top_25_with_greying_when_universe_has_25_plus(
        self,
        mock_send_email,
        mock_india_double_hrp_email_request,
    ):
        # Bump the Stage 1 universe to >25 so the table renders the full
        # top-25 context -- proves India inherits the same greying pass
        # the US email gets.
        mock_send_email.return_value = True
        big_weights = {f"S{i:03d}.NS": 1.0 for i in range(30)}
        mock_india_double_hrp_email_request["stage1"]["percentage_weights"] = (
            big_weights
        )
        mock_india_double_hrp_email_request["stage1"]["symbols_used"] = 30
        response = client.post(
            "/email/india-double-hrp-report",
            json=mock_india_double_hrp_email_request,
        )
        body = response.json()["body"]
        # All top-25 symbols should be referenced in Stage 1 rows.
        for i in range(25):
            assert f"S{i:03d}.NS" in body
        # The 26th symbol must NOT appear (rendering caps at 25).
        assert "S025.NS" not in body

    def test_missing_required_field_returns_422(self):
        response = client.post(
            "/email/india-double-hrp-report",
            json={
                "summary": {"para_1_screening_overview": "x"},
                "universe": "halal_india_double_hrp",
                "top_n": 15,
                "target_week_start": "2026-02-23",
                "target_week_end": "2026-02-27",
                "as_of_date": "2026-02-23",
            },
        )
        assert response.status_code == 422

    @patch("brain_api.routes.email.weekly_report.send_html_email")
    def test_india_with_prior_allocation_db_label(
        self,
        mock_send_email,
        mock_india_double_hrp_email_request,
    ):
        """India sources prior allocation from DB (paper-only, no broker)."""
        mock_send_email.return_value = True
        mock_india_double_hrp_email_request["prior_allocation"] = {
            "weights": {"S001.NS": 0.07, "S002.NS": 0.06, "CASH": 0.0},
            "source_label": "recorded last week (202608)",
            "as_of": "202608",
        }
        response = client.post(
            "/email/india-double-hrp-report",
            json=mock_india_double_hrp_email_request,
        )
        assert response.status_code == 200
        body = response.json()["body"]
        # India never gets an order detail table -- the partial is gated
        # on ``order_results.orders`` and India never sets order_results.
        assert "Order Execution Detail" not in body
        # Prior allocation block IS visible with the DB label.
        assert "Going Into This Week" in body
        assert "recorded last week" in body
        assert "S001.NS" in body

    @patch("brain_api.routes.email.weekly_report.send_html_email")
    def test_india_double_hrp_stage2_renders_stop_loss_from_paper_allocation(
        self,
        mock_send_email,
        mock_india_double_hrp_email_request,
    ):
        """Stage 2 Stop Loss column renders atr14 from paper_allocation."""
        mock_send_email.return_value = True
        mock_india_double_hrp_email_request["paper_allocation"] = {
            "details": [
                {
                    "symbol": "S000.NS",
                    "weight_pct": 100.0 / 15,
                    "price": 100.0,
                    "whole_shares": 66,
                    "trade_value": 6600.0,
                    "stop_loss_price": 94.0,
                    "stop_loss_distance_pct": 0.06,
                    "stop_loss_reason": "atr14",
                },
            ],
            "total_nav": 100_000.0,
            "prices_used": {"S000.NS": 100.0},
            "total_allocated_pct": 100.0 / 15,
        }
        response = client.post(
            "/email/india-double-hrp-report",
            json=mock_india_double_hrp_email_request,
        )
        assert response.status_code == 200
        body = response.json()["body"]
        assert "Stop Loss" in body
        assert "94.00" in body
        assert "(-6.0%)" in body
        assert "ATR(14)" in body
        # Still no US order-execution table -- stop lives on Stage 2.
        assert "Order Execution Detail" not in body
