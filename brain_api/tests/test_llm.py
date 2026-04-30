"""Tests for LLM endpoints."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from brain_api.main import app
from brain_api.routes.llm.providers import (
    LLMProvider,
    LLMResponse,
    OllamaProvider,
    OpenAIProvider,
    get_llm_provider,
    parse_json_response,
)

client = TestClient(app)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_training_summary_request():
    """Valid request payload for training summary endpoint."""
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
        "sac": {
            "version": "v2026-01-15-jkl012",
            "data_window_start": "2020-01-01",
            "data_window_end": "2025-12-31",
            "metrics": {"sharpe": 1.8, "max_drawdown": 0.12},
            "promoted": False,
            "symbols_used": ["AAPL", "MSFT", "GOOGL"],
        },
    }


@pytest.fixture
def mock_llm_json_response():
    """Mock JSON response from LLM."""
    return {
        "para_1_overall": "All models trained successfully with good metrics.",
        "para_2_lstm": "LSTM model shows strong price prediction capability.",
        "para_3_patchtst": "PatchTST leverages OHLCV approach effectively.",
        "para_4_sac": "SAC shows promising results but was not promoted.",
        "para_5_recommendations": "Consider investigating SAC promotion criteria.",
    }


# =============================================================================
# Test Provider Functions
# =============================================================================


class TestParseJsonResponse:
    """Tests for parse_json_response helper."""

    def test_parse_valid_json(self):
        """Parse valid JSON string."""
        result = parse_json_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_json_with_markdown_code_block(self):
        """Parse JSON wrapped in markdown code block."""
        content = '```json\n{"key": "value"}\n```'
        result = parse_json_response(content)
        assert result == {"key": "value"}

    def test_parse_json_with_generic_code_block(self):
        """Parse JSON wrapped in generic code block."""
        content = '```\n{"key": "value"}\n```'
        result = parse_json_response(content)
        assert result == {"key": "value"}

    def test_parse_invalid_json_raises(self):
        """Invalid JSON raises ValueError."""
        with pytest.raises(ValueError, match="Failed to parse JSON"):
            parse_json_response("not valid json")


class TestGetLLMProvider:
    """Tests for get_llm_provider factory."""

    def test_default_is_openai(self, monkeypatch):
        """Default provider is OpenAI when LLM_PROVIDER not set."""
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        provider = get_llm_provider()
        assert isinstance(provider, OpenAIProvider)
        assert provider.name == "openai"

    def test_openai_provider(self, monkeypatch):
        """LLM_PROVIDER=openai returns OpenAIProvider."""
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        provider = get_llm_provider()
        assert isinstance(provider, OpenAIProvider)

    def test_ollama_provider(self, monkeypatch):
        """LLM_PROVIDER=ollama returns OllamaProvider."""
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        provider = get_llm_provider()
        assert isinstance(provider, OllamaProvider)
        assert provider.name == "ollama"

    def test_unknown_provider_raises(self, monkeypatch):
        """Unknown LLM_PROVIDER raises ValueError."""
        monkeypatch.setenv("LLM_PROVIDER", "unknown")
        with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
            get_llm_provider()

    def test_openai_missing_api_key_raises(self, monkeypatch):
        """OpenAI provider without API key raises ValueError."""
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            get_llm_provider()


class TestOpenAIProvider:
    """Tests for OpenAIProvider."""

    def test_init_with_env_vars(self, monkeypatch):
        """Provider initializes from environment variables."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4")
        provider = OpenAIProvider()
        assert provider.model == "gpt-4"
        assert provider.name == "openai"

    def test_init_with_explicit_params(self, monkeypatch):
        """Provider accepts explicit parameters."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        provider = OpenAIProvider(
            api_key="explicit-key",
            model="gpt-3.5-turbo",
            temperature=0.5,
        )
        assert provider.model == "gpt-3.5-turbo"
        assert provider.temperature == 0.5

    @patch("brain_api.routes.llm.providers.OpenAI")
    def test_generate_calls_openai(self, mock_openai_class, monkeypatch):
        """Generate method calls OpenAI API correctly."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Setup mock
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"result": "test"}'))
        ]
        mock_response.usage = MagicMock(total_tokens=100)
        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIProvider()
        result = provider.generate("Test prompt")

        assert result.content == '{"result": "test"}'
        assert result.tokens_used == 100
        mock_client.chat.completions.create.assert_called_once()


class TestOllamaProvider:
    """Tests for OllamaProvider."""

    def test_init_with_defaults(self, monkeypatch):
        """Provider initializes with default values."""
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("OLLAMA_MODEL", raising=False)
        provider = OllamaProvider()
        assert provider.base_url == "http://localhost:11434"
        assert provider.model == "llama3.2"
        assert provider.name == "ollama"

    def test_init_with_env_vars(self, monkeypatch):
        """Provider initializes from environment variables."""
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://custom:8080")
        monkeypatch.setenv("OLLAMA_MODEL", "mistral")
        provider = OllamaProvider()
        assert provider.base_url == "http://custom:8080"
        assert provider.model == "mistral"


# =============================================================================
# Test API Endpoint
# =============================================================================


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""

    def __init__(
        self, name: str, response: LLMResponse, error: Exception | None = None
    ):
        self._name = name
        self._response = response
        self._error = error

    @property
    def name(self) -> str:
        return self._name

    def generate(self, prompt: str) -> LLMResponse:
        if self._error:
            raise self._error
        return self._response


class TestTrainingSummaryEndpoint:
    """Tests for POST /llm/training-summary endpoint."""

    def test_successful_summary_generation(
        self,
        mock_training_summary_request,
        mock_llm_json_response,
    ):
        """Successful training summary generation."""
        import json

        mock_provider = MockLLMProvider(
            name="openai",
            response=LLMResponse(
                content=json.dumps(mock_llm_json_response),
                model="gpt-4o-mini",
                tokens_used=500,
            ),
        )

        # Override dependency
        app.dependency_overrides[get_llm_provider] = lambda: mock_provider

        try:
            response = client.post(
                "/llm/training-summary",
                json=mock_training_summary_request,
            )

            assert response.status_code == 200
            data = response.json()
            assert "summary" in data
            assert data["provider"] == "openai"
            assert data["model_used"] == "gpt-4o-mini"
            assert data["tokens_used"] == 500
        finally:
            app.dependency_overrides.clear()

    def test_ollama_provider_response(
        self,
        mock_training_summary_request,
        mock_llm_json_response,
    ):
        """Training summary with OLLAMA provider."""
        import json

        mock_provider = MockLLMProvider(
            name="ollama",
            response=LLMResponse(
                content=json.dumps(mock_llm_json_response),
                model="llama3.2",
                tokens_used=None,
            ),
        )

        app.dependency_overrides[get_llm_provider] = lambda: mock_provider

        try:
            response = client.post(
                "/llm/training-summary",
                json=mock_training_summary_request,
            )

            assert response.status_code == 200
            data = response.json()
            assert data["provider"] == "ollama"
            assert data["model_used"] == "llama3.2"
            assert data["tokens_used"] is None
        finally:
            app.dependency_overrides.clear()

    def test_llm_service_unavailable(
        self,
        mock_training_summary_request,
    ):
        """LLM service failure returns 503."""
        mock_provider = MockLLMProvider(
            name="openai",
            response=LLMResponse(content="", model="", tokens_used=None),
            error=Exception("API connection failed"),
        )

        app.dependency_overrides[get_llm_provider] = lambda: mock_provider

        try:
            response = client.post(
                "/llm/training-summary",
                json=mock_training_summary_request,
            )

            assert response.status_code == 503
            assert "LLM service unavailable" in response.json()["detail"]
        finally:
            app.dependency_overrides.clear()

    def test_invalid_json_response_fallback(
        self,
        mock_training_summary_request,
    ):
        """Invalid JSON from LLM returns fallback summary."""
        mock_provider = MockLLMProvider(
            name="openai",
            response=LLMResponse(
                content="This is not valid JSON",
                model="gpt-4o-mini",
                tokens_used=100,
            ),
        )

        app.dependency_overrides[get_llm_provider] = lambda: mock_provider

        try:
            response = client.post(
                "/llm/training-summary",
                json=mock_training_summary_request,
            )

            assert response.status_code == 200
            data = response.json()
            assert "para_1_overall" in data["summary"]
            assert "Unable to generate AI summary" in data["summary"]["para_1_overall"]
        finally:
            app.dependency_overrides.clear()

    def test_invalid_request_returns_422(self):
        """Invalid request body returns 422."""
        # Override dependency to avoid API key check (validation happens before dependency)
        mock_provider = MockLLMProvider(
            name="openai",
            response=LLMResponse(content="{}", model="test", tokens_used=0),
        )
        app.dependency_overrides[get_llm_provider] = lambda: mock_provider

        try:
            response = client.post(
                "/llm/training-summary",
                json={"lstm": "invalid"},  # Should be an object
            )
            assert response.status_code == 422
        finally:
            app.dependency_overrides.clear()

    def test_missing_required_field_returns_422(self):
        """Missing required field returns 422."""
        # Override dependency to avoid API key check (validation happens before dependency)
        mock_provider = MockLLMProvider(
            name="openai",
            response=LLMResponse(content="{}", model="test", tokens_used=0),
        )
        app.dependency_overrides[get_llm_provider] = lambda: mock_provider

        try:
            response = client.post(
                "/llm/training-summary",
                json={
                    "lstm": {
                        "version": "v1",
                        "data_window_start": "2020-01-01",
                        "data_window_end": "2025-01-01",
                        "metrics": {},
                        "promoted": True,
                    },
                    # Missing patchtst, sac
                },
            )
            assert response.status_code == 422
        finally:
            app.dependency_overrides.clear()


# =============================================================================
# Weekly Summary Endpoint Tests
# =============================================================================


@pytest.fixture
def mock_weekly_summary_request():
    """Valid request payload for weekly summary endpoint."""
    return {
        "lstm": {
            "predictions": [
                {
                    "symbol": "AAPL",
                    "predicted_weekly_return_pct": 2.5,
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
                    "direction": "UP",
                    "has_enough_history": True,
                    "history_days_used": 252,
                    "data_end_date": "2026-02-03",
                    "target_week_start": "2026-02-03",
                    "target_week_end": "2026-02-07",
                },
            ],
            "model_version": "v2026-01-15-abc123",
            "as_of_date": "2026-02-03",
            "target_week_start": "2026-02-03",
            "target_week_end": "2026-02-07",
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
            "model_version": "v2026-01-15-def456",
            "as_of_date": "2026-02-03",
            "target_week_start": "2026-02-03",
            "target_week_end": "2026-02-07",
            "signals_used": ["ohlcv"],
        },
        "news": {
            "run_id": "paper:2026-02-03",
            "attempt": 1,
            "as_of_date": "2026-02-03",
            "from_cache": False,
            "per_symbol": [
                {
                    "symbol": "AAPL",
                    "article_count_fetched": 10,
                    "article_count_used": 5,
                    "sentiment_score": 0.65,
                    "insufficient_news": False,
                    "top_k_articles": [],
                },
            ],
        },
        "fundamentals": {
            "as_of_date": "2026-02-03",
            "per_symbol": [
                {
                    "symbol": "AAPL",
                    "ratios": {
                        "symbol": "AAPL",
                        "as_of_date": "2026-02-03",
                        "gross_margin": 0.43,
                        "operating_margin": 0.30,
                        "net_margin": 0.25,
                        "current_ratio": 1.05,
                        "debt_to_equity": 1.5,
                    },
                    "error": None,
                },
            ],
        },
        "sac": {
            "target_weights": {"AAPL": 0.12, "MSFT": 0.10, "CASH": 0.05},
            "turnover": 0.15,
            "target_week_start": "2026-02-03",
            "target_week_end": "2026-02-07",
            "model_version": "v2026-01-15-sac001",
            "weight_changes": [],
        },
    }


@pytest.fixture
def mock_weekly_llm_json_response():
    """Mock JSON response from LLM for SAC weekly summary."""
    return {
        "para_1_overall_summary": "This week shows bullish momentum across tech stocks.",
        "para_2_sac": "SAC allocator favors AAPL and MSFT with moderate turnover.",
        "para_3_patchtst_forecast": "PatchTST predicts positive returns for tech sector.",
        "para_4_lstm_forecast": "LSTM shows strong bullish signals for AAPL.",
        "para_5_news_sentiment": "News sentiment is generally positive for holdings.",
        "para_6_fundamentals": "Fundamentals remain strong with solid margins.",
    }


# =============================================================================
# India Alpha-HRP Summary Endpoint Tests
# =============================================================================


@pytest.fixture
def mock_india_weekly_summary_request():
    """Valid request payload for India Alpha-HRP summary endpoint (HRP only)."""
    return {
        "hrp": {
            "percentage_weights": {
                "RELIANCE.NS": 12.3,
                "TCS.NS": 10.1,
                "INFY.NS": 9.5,
                "HDFCBANK.NS": 8.7,
                "BAJFINANCE.NS": 7.2,
                "WIPRO.NS": 6.8,
                "MARUTI.NS": 6.4,
                "NESTLEIND.NS": 5.9,
                "TITAN.NS": 5.5,
                "DABUR.NS": 5.1,
                "COLPAL.NS": 4.8,
                "HINDUNILVR.NS": 4.5,
                "GODREJCP.NS": 4.3,
                "MARICO.NS": 4.6,
                "PIDILITIND.NS": 4.3,
            },
            "symbols_used": 15,
            "symbols_excluded": ["ADANIENT.NS"],
            "lookback_days": 252,
            "as_of_date": "2026-03-02",
        },
        "universe": "halal_india",
    }


@pytest.fixture
def mock_india_llm_json_response():
    """Mock JSON response from LLM for India summary."""
    return {
        "para_1_portfolio_overview": "HRP allocated across 15 NSE stocks with moderate concentration.",
        "para_2_concentration_analysis": "Top 3 holdings (RELIANCE, TCS, INFY) hold 31.9% combined.",
        "para_3_risk_observations": "IT sector is overweight; watch for INR/USD currency risk.",
    }


class TestIndiaAlphaHRPSummaryEndpoint:
    """Tests for POST /llm/india-alpha-hrp-summary endpoint."""

    def test_successful_india_summary_generation(
        self,
        mock_india_weekly_summary_request,
        mock_india_llm_json_response,
    ):
        """Successful India Alpha-HRP summary generation with HRP data."""
        import json

        mock_provider = MockLLMProvider(
            name="openai",
            response=LLMResponse(
                content=json.dumps(mock_india_llm_json_response),
                model="gpt-4o-mini",
                tokens_used=350,
            ),
        )

        app.dependency_overrides[get_llm_provider] = lambda: mock_provider

        try:
            response = client.post(
                "/llm/india-alpha-hrp-summary",
                json=mock_india_weekly_summary_request,
            )

            assert response.status_code == 200
            data = response.json()
            assert "summary" in data
            assert data["provider"] == "openai"
            assert data["model_used"] == "gpt-4o-mini"
            assert data["tokens_used"] == 350
            assert "para_1_portfolio_overview" in data["summary"]
            assert "para_2_concentration_analysis" in data["summary"]
            assert "para_3_risk_observations" in data["summary"]
        finally:
            app.dependency_overrides.clear()

    def test_india_summary_llm_failure(
        self,
        mock_india_weekly_summary_request,
    ):
        """LLM service failure returns 503."""
        mock_provider = MockLLMProvider(
            name="openai",
            response=LLMResponse(content="", model="", tokens_used=None),
            error=Exception("API connection failed"),
        )

        app.dependency_overrides[get_llm_provider] = lambda: mock_provider

        try:
            response = client.post(
                "/llm/india-alpha-hrp-summary",
                json=mock_india_weekly_summary_request,
            )

            assert response.status_code == 503
            assert "LLM service unavailable" in response.json()["detail"]
        finally:
            app.dependency_overrides.clear()

    def test_india_summary_json_parse_error(
        self,
        mock_india_weekly_summary_request,
    ):
        """Invalid JSON from LLM returns fallback summary."""
        mock_provider = MockLLMProvider(
            name="openai",
            response=LLMResponse(
                content="This is not valid JSON at all",
                model="gpt-4o-mini",
                tokens_used=100,
            ),
        )

        app.dependency_overrides[get_llm_provider] = lambda: mock_provider

        try:
            response = client.post(
                "/llm/india-alpha-hrp-summary",
                json=mock_india_weekly_summary_request,
            )

            assert response.status_code == 200
            data = response.json()
            assert "para_1_portfolio_overview" in data["summary"]
            assert (
                "Unable to generate AI summary"
                in data["summary"]["para_1_portfolio_overview"]
            )
        finally:
            app.dependency_overrides.clear()

    def test_india_summary_missing_hrp_returns_422(self):
        """Missing HRP field returns 422."""
        mock_provider = MockLLMProvider(
            name="openai",
            response=LLMResponse(content="{}", model="test", tokens_used=0),
        )
        app.dependency_overrides[get_llm_provider] = lambda: mock_provider

        try:
            response = client.post(
                "/llm/india-alpha-hrp-summary",
                json={},
            )
            assert response.status_code == 422
        finally:
            app.dependency_overrides.clear()

    def test_india_summary_invalid_hrp_returns_422(self):
        """Invalid HRP structure returns 422."""
        mock_provider = MockLLMProvider(
            name="openai",
            response=LLMResponse(content="{}", model="test", tokens_used=0),
        )
        app.dependency_overrides[get_llm_provider] = lambda: mock_provider

        try:
            response = client.post(
                "/llm/india-alpha-hrp-summary",
                json={"hrp": "invalid"},
            )
            assert response.status_code == 422
        finally:
            app.dependency_overrides.clear()


class TestSACWeeklySummaryEndpoint:
    """Tests for POST /llm/sac-weekly-summary endpoint."""

    def test_successful_weekly_summary_generation(
        self,
        mock_weekly_summary_request,
        mock_weekly_llm_json_response,
    ):
        """Successful weekly summary generation."""
        import json

        mock_provider = MockLLMProvider(
            name="openai",
            response=LLMResponse(
                content=json.dumps(mock_weekly_llm_json_response),
                model="gpt-4o-mini",
                tokens_used=800,
            ),
        )

        app.dependency_overrides[get_llm_provider] = lambda: mock_provider

        try:
            response = client.post(
                "/llm/sac-weekly-summary",
                json=mock_weekly_summary_request,
            )

            assert response.status_code == 200
            data = response.json()
            assert "summary" in data
            assert data["provider"] == "openai"
            assert data["model_used"] == "gpt-4o-mini"
            assert data["tokens_used"] == 800
            assert "para_1_overall_summary" in data["summary"]
        finally:
            app.dependency_overrides.clear()

    def test_weekly_summary_llm_failure(
        self,
        mock_weekly_summary_request,
    ):
        """LLM service failure returns 503."""
        mock_provider = MockLLMProvider(
            name="openai",
            response=LLMResponse(content="", model="", tokens_used=None),
            error=Exception("API connection failed"),
        )

        app.dependency_overrides[get_llm_provider] = lambda: mock_provider

        try:
            response = client.post(
                "/llm/sac-weekly-summary",
                json=mock_weekly_summary_request,
            )

            assert response.status_code == 503
            assert "LLM service unavailable" in response.json()["detail"]
        finally:
            app.dependency_overrides.clear()

    def test_weekly_summary_json_parse_error(
        self,
        mock_weekly_summary_request,
    ):
        """Invalid JSON from LLM returns fallback summary."""
        mock_provider = MockLLMProvider(
            name="openai",
            response=LLMResponse(
                content="This is not valid JSON at all",
                model="gpt-4o-mini",
                tokens_used=100,
            ),
        )

        app.dependency_overrides[get_llm_provider] = lambda: mock_provider

        try:
            response = client.post(
                "/llm/sac-weekly-summary",
                json=mock_weekly_summary_request,
            )

            assert response.status_code == 200
            data = response.json()
            assert "para_1_overall_summary" in data["summary"]
            assert (
                "Unable to generate AI summary"
                in data["summary"]["para_1_overall_summary"]
            )
        finally:
            app.dependency_overrides.clear()


# =============================================================================
# US Double HRP Summary Endpoint Tests
# =============================================================================


@pytest.fixture
def mock_us_double_hrp_request():
    """Valid request payload for /llm/us-double-hrp-summary."""
    return {
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
    }


@pytest.fixture
def mock_us_double_hrp_llm_response():
    """Mock JSON response from LLM for US Double HRP summary."""
    return {
        "para_1_screening_overview": "HRP screened 410 halal_new stocks over 756 days.",
        "para_2_selection_rationale": "Top 15 are tech-heavy with low correlation.",
        "para_3_final_allocation": "Stage 2 distributes evenly with NVDA at 7.5%.",
        "para_4_risk_observations": "Watch sector concentration in semis.",
    }


class TestUSDoubleHRPSummaryEndpoint:
    """Tests for POST /llm/us-double-hrp-summary endpoint."""

    def test_happy_path(
        self, mock_us_double_hrp_request, mock_us_double_hrp_llm_response
    ):
        import json

        mock_provider = MockLLMProvider(
            name="openai",
            response=LLMResponse(
                content=json.dumps(mock_us_double_hrp_llm_response),
                model="gpt-4o-mini",
                tokens_used=400,
            ),
        )
        app.dependency_overrides[get_llm_provider] = lambda: mock_provider
        try:
            response = client.post(
                "/llm/us-double-hrp-summary",
                json=mock_us_double_hrp_request,
            )
            assert response.status_code == 200, response.text
            data = response.json()
            assert "para_1_screening_overview" in data["summary"]
            assert "para_2_selection_rationale" in data["summary"]
            assert data["provider"] == "openai"
            assert data["model_used"] == "gpt-4o-mini"
            assert data["tokens_used"] == 400
        finally:
            app.dependency_overrides.clear()

    def test_missing_required_field_returns_422(self):
        mock_provider = MockLLMProvider(
            name="openai",
            response=LLMResponse(content="{}", model="test", tokens_used=0),
        )
        app.dependency_overrides[get_llm_provider] = lambda: mock_provider
        try:
            # Missing stage2
            response = client.post(
                "/llm/us-double-hrp-summary",
                json={
                    "stage1": {
                        "percentage_weights": {"AAPL": 1.0},
                        "symbols_used": 1,
                        "symbols_excluded": [],
                        "lookback_days": 756,
                        "as_of_date": "2026-02-23",
                    },
                    "universe": "halal_new",
                    "top_n": 15,
                },
            )
            assert response.status_code == 422
        finally:
            app.dependency_overrides.clear()

    def test_json_parse_fallback(self, mock_us_double_hrp_request):
        mock_provider = MockLLMProvider(
            name="openai",
            response=LLMResponse(
                content="not json", model="gpt-4o-mini", tokens_used=50
            ),
        )
        app.dependency_overrides[get_llm_provider] = lambda: mock_provider
        try:
            response = client.post(
                "/llm/us-double-hrp-summary",
                json=mock_us_double_hrp_request,
            )
            assert response.status_code == 200
            data = response.json()
            assert "para_1_screening_overview" in data["summary"]
            assert (
                "Unable to generate AI summary"
                in data["summary"]["para_1_screening_overview"]
            )
        finally:
            app.dependency_overrides.clear()

    def test_llm_failure_returns_503(self, mock_us_double_hrp_request):
        mock_provider = MockLLMProvider(
            name="openai",
            response=LLMResponse(content="", model="", tokens_used=None),
            error=Exception("LLM down"),
        )
        app.dependency_overrides[get_llm_provider] = lambda: mock_provider
        try:
            response = client.post(
                "/llm/us-double-hrp-summary",
                json=mock_us_double_hrp_request,
            )
            assert response.status_code == 503
            assert "LLM service unavailable" in response.json()["detail"]
        finally:
            app.dependency_overrides.clear()
