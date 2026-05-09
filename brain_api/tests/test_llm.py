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
def mock_forecasters_summary_request():
    """Valid request payload for /llm/forecasters-training-summary."""
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
    }


@pytest.fixture
def mock_sac_summary_request():
    """Valid request payload for /llm/sac-training-summary."""
    return {
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
def mock_forecasters_llm_json_response():
    """Mock JSON response from LLM for the forecasters summary endpoint."""
    return {
        "para_1_overall": "Both forecasters trained successfully with good metrics.",
        "para_2_lstm": "LSTM model shows strong price prediction capability.",
        "para_3_patchtst": "PatchTST leverages OHLCV approach effectively.",
        "para_4_recommendations": "Slate looks stable for the SAC retrain tomorrow.",
    }


@pytest.fixture
def mock_sac_llm_json_response():
    """Mock JSON response from LLM for the SAC summary endpoint."""
    return {
        "para_1_overall": "SAC training completed but did not clear the promotion gate.",
        "para_2_metrics": "Sharpe 1.8 and 12% max drawdown are mediocre this week.",
        "para_3_recommendations": "Investigate SAC promotion criteria before next retrain.",
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


class TestForecastersTrainingSummaryEndpoint:
    """Tests for POST /llm/forecasters-training-summary endpoint."""

    def test_successful_summary_generation(
        self,
        mock_forecasters_summary_request,
        mock_forecasters_llm_json_response,
    ):
        """Successful forecasters summary generation."""
        import json

        mock_provider = MockLLMProvider(
            name="openai",
            response=LLMResponse(
                content=json.dumps(mock_forecasters_llm_json_response),
                model="gpt-5-mini",
                tokens_used=500,
            ),
        )
        app.dependency_overrides[get_llm_provider] = lambda: mock_provider
        try:
            response = client.post(
                "/llm/forecasters-training-summary",
                json=mock_forecasters_summary_request,
            )
            assert response.status_code == 200, response.text
            data = response.json()
            assert "summary" in data
            assert data["provider"] == "openai"
            assert data["model_used"] == "gpt-5-mini"
            assert data["tokens_used"] == 500
            assert "para_1_overall" in data["summary"]
            assert "para_2_lstm" in data["summary"]
            assert "para_3_patchtst" in data["summary"]
        finally:
            app.dependency_overrides.clear()

    def test_ollama_provider_response(
        self,
        mock_forecasters_summary_request,
        mock_forecasters_llm_json_response,
    ):
        """Forecasters summary with OLLAMA provider."""
        import json

        mock_provider = MockLLMProvider(
            name="ollama",
            response=LLMResponse(
                content=json.dumps(mock_forecasters_llm_json_response),
                model="llama3.2",
                tokens_used=None,
            ),
        )
        app.dependency_overrides[get_llm_provider] = lambda: mock_provider
        try:
            response = client.post(
                "/llm/forecasters-training-summary",
                json=mock_forecasters_summary_request,
            )
            assert response.status_code == 200
            data = response.json()
            assert data["provider"] == "ollama"
            assert data["model_used"] == "llama3.2"
            assert data["tokens_used"] is None
        finally:
            app.dependency_overrides.clear()

    def test_llm_service_unavailable(self, mock_forecasters_summary_request):
        """LLM service failure returns 503."""
        mock_provider = MockLLMProvider(
            name="openai",
            response=LLMResponse(content="", model="", tokens_used=None),
            error=Exception("API connection failed"),
        )
        app.dependency_overrides[get_llm_provider] = lambda: mock_provider
        try:
            response = client.post(
                "/llm/forecasters-training-summary",
                json=mock_forecasters_summary_request,
            )
            assert response.status_code == 503
            assert "LLM service unavailable" in response.json()["detail"]
        finally:
            app.dependency_overrides.clear()

    def test_invalid_json_response_fallback(self, mock_forecasters_summary_request):
        """Invalid JSON from LLM returns the para_1_overall fallback stub."""
        mock_provider = MockLLMProvider(
            name="openai",
            response=LLMResponse(
                content="This is not valid JSON",
                model="gpt-5-mini",
                tokens_used=100,
            ),
        )
        app.dependency_overrides[get_llm_provider] = lambda: mock_provider
        try:
            response = client.post(
                "/llm/forecasters-training-summary",
                json=mock_forecasters_summary_request,
            )
            assert response.status_code == 200
            data = response.json()
            assert "para_1_overall" in data["summary"]
            assert "Unable to generate AI summary" in data["summary"]["para_1_overall"]
        finally:
            app.dependency_overrides.clear()

    def test_invalid_request_returns_422(self):
        """Invalid request body returns 422."""
        mock_provider = MockLLMProvider(
            name="openai",
            response=LLMResponse(content="{}", model="test", tokens_used=0),
        )
        app.dependency_overrides[get_llm_provider] = lambda: mock_provider
        try:
            response = client.post(
                "/llm/forecasters-training-summary",
                json={"lstm": "invalid"},
            )
            assert response.status_code == 422
        finally:
            app.dependency_overrides.clear()

    def test_missing_patchtst_returns_422(self):
        """Missing patchtst field returns 422 (lstm + patchtst both required)."""
        mock_provider = MockLLMProvider(
            name="openai",
            response=LLMResponse(content="{}", model="test", tokens_used=0),
        )
        app.dependency_overrides[get_llm_provider] = lambda: mock_provider
        try:
            response = client.post(
                "/llm/forecasters-training-summary",
                json={
                    "lstm": {
                        "version": "v1",
                        "data_window_start": "2020-01-01",
                        "data_window_end": "2025-01-01",
                        "metrics": {},
                        "promoted": True,
                    },
                },
            )
            assert response.status_code == 422
        finally:
            app.dependency_overrides.clear()

    def test_sac_field_is_rejected(self, mock_forecasters_summary_request):
        """SAC training data must NOT be sent to the forecasters endpoint.

        We don't enforce extra=forbid at the model layer, so sending an
        extra ``sac`` field still returns 200 (it's silently dropped),
        but a sloppy caller that omits the required ``patchtst`` while
        sending ``sac`` instead must still get 422. This guards against
        accidentally wiring the SAC payload through the wrong activity.
        """
        import json

        mock_provider = MockLLMProvider(
            name="openai",
            response=LLMResponse(
                content=json.dumps({"para_1_overall": "ok"}),
                model="gpt-5-mini",
                tokens_used=10,
            ),
        )
        app.dependency_overrides[get_llm_provider] = lambda: mock_provider
        try:
            payload = {
                "lstm": mock_forecasters_summary_request["lstm"],
                "sac": {
                    "version": "v",
                    "data_window_start": "2020-01-01",
                    "data_window_end": "2025-01-01",
                    "metrics": {},
                    "promoted": True,
                },
            }
            response = client.post(
                "/llm/forecasters-training-summary",
                json=payload,
            )
            assert response.status_code == 422
        finally:
            app.dependency_overrides.clear()


class TestSACTrainingSummaryEndpoint:
    """Tests for POST /llm/sac-training-summary endpoint."""

    def test_successful_summary_generation(
        self,
        mock_sac_summary_request,
        mock_sac_llm_json_response,
    ):
        """Successful SAC summary generation."""
        import json

        mock_provider = MockLLMProvider(
            name="openai",
            response=LLMResponse(
                content=json.dumps(mock_sac_llm_json_response),
                model="gpt-5-mini",
                tokens_used=300,
            ),
        )
        app.dependency_overrides[get_llm_provider] = lambda: mock_provider
        try:
            response = client.post(
                "/llm/sac-training-summary",
                json=mock_sac_summary_request,
            )
            assert response.status_code == 200, response.text
            data = response.json()
            assert "summary" in data
            assert data["provider"] == "openai"
            assert data["tokens_used"] == 300
            assert "para_1_overall" in data["summary"]
            assert "para_2_metrics" in data["summary"]
            assert "para_3_recommendations" in data["summary"]
        finally:
            app.dependency_overrides.clear()

    def test_llm_service_unavailable(self, mock_sac_summary_request):
        """LLM service failure returns 503."""
        mock_provider = MockLLMProvider(
            name="openai",
            response=LLMResponse(content="", model="", tokens_used=None),
            error=Exception("LLM down"),
        )
        app.dependency_overrides[get_llm_provider] = lambda: mock_provider
        try:
            response = client.post(
                "/llm/sac-training-summary",
                json=mock_sac_summary_request,
            )
            assert response.status_code == 503
            assert "LLM service unavailable" in response.json()["detail"]
        finally:
            app.dependency_overrides.clear()

    def test_invalid_json_response_fallback(self, mock_sac_summary_request):
        """Invalid JSON from LLM returns fallback under para_1_overall."""
        mock_provider = MockLLMProvider(
            name="openai",
            response=LLMResponse(
                content="not json", model="gpt-5-mini", tokens_used=42
            ),
        )
        app.dependency_overrides[get_llm_provider] = lambda: mock_provider
        try:
            response = client.post(
                "/llm/sac-training-summary",
                json=mock_sac_summary_request,
            )
            assert response.status_code == 200
            data = response.json()
            assert "Unable to generate AI summary" in data["summary"]["para_1_overall"]
        finally:
            app.dependency_overrides.clear()

    def test_missing_sac_returns_422(self):
        """Empty body fails validation (sac is required)."""
        mock_provider = MockLLMProvider(
            name="openai",
            response=LLMResponse(content="{}", model="test", tokens_used=0),
        )
        app.dependency_overrides[get_llm_provider] = lambda: mock_provider
        try:
            response = client.post("/llm/sac-training-summary", json={})
            assert response.status_code == 422
        finally:
            app.dependency_overrides.clear()

    def test_default_universe_is_halal_filtered(
        self,
        mock_sac_summary_request,
        mock_sac_llm_json_response,
    ):
        """Backward compatibility: omitting ``universe`` defaults to halal_filtered.

        The legacy SAC workflow predates the parallel A/B halal bucket
        and posts payloads without the ``universe`` field. The endpoint
        must accept those payloads and render the existing
        halal_filtered prompt branch.
        """
        import json

        captured_prompts: list[str] = []

        class CapturingProvider(MockLLMProvider):
            def generate(self, prompt: str) -> LLMResponse:
                captured_prompts.append(prompt)
                return super().generate(prompt)

        mock_provider = CapturingProvider(
            name="openai",
            response=LLMResponse(
                content=json.dumps(mock_sac_llm_json_response),
                model="gpt-5-mini",
                tokens_used=300,
            ),
        )
        app.dependency_overrides[get_llm_provider] = lambda: mock_provider
        try:
            response = client.post(
                "/llm/sac-training-summary",
                json=mock_sac_summary_request,
            )
            assert response.status_code == 200, response.text
            assert len(captured_prompts) == 1
            assert "halal_filtered" in captured_prompts[0]
        finally:
            app.dependency_overrides.clear()

    def test_universe_halal_renders_in_prompt(
        self,
        mock_sac_summary_request,
        mock_sac_llm_json_response,
    ):
        """``universe="halal"`` must reach the prompt template.

        Two parallel A/B SAC workflows hit this endpoint with different
        ``universe`` values; the rendered prompt must identify the
        bucket so the LLM-generated summary is unambiguous.
        """
        import json

        captured_prompts: list[str] = []

        class CapturingProvider(MockLLMProvider):
            def generate(self, prompt: str) -> LLMResponse:
                captured_prompts.append(prompt)
                return super().generate(prompt)

        mock_provider = CapturingProvider(
            name="openai",
            response=LLMResponse(
                content=json.dumps(mock_sac_llm_json_response),
                model="gpt-5-mini",
                tokens_used=300,
            ),
        )
        app.dependency_overrides[get_llm_provider] = lambda: mock_provider
        try:
            payload = {**mock_sac_summary_request, "universe": "halal"}
            response = client.post("/llm/sac-training-summary", json=payload)
            assert response.status_code == 200, response.text
            assert len(captured_prompts) == 1
            assert "Universe: ``halal``" in captured_prompts[0]
            assert "yfinance ETF top-holdings" in captured_prompts[0]
        finally:
            app.dependency_overrides.clear()

    def test_promoted_renders_guardrail_pass_prose(
        self,
        mock_sac_summary_request,
        mock_sac_llm_json_response,
    ):
        """Promoted SAC run renders guardrail-pass copy in the prompt."""
        import json

        captured_prompts: list[str] = []

        class CapturingProvider(MockLLMProvider):
            def generate(self, prompt: str) -> LLMResponse:
                captured_prompts.append(prompt)
                return super().generate(prompt)

        mock_provider = CapturingProvider(
            name="openai",
            response=LLMResponse(
                content=json.dumps(mock_sac_llm_json_response),
                model="gpt-5-mini",
                tokens_used=300,
            ),
        )
        app.dependency_overrides[get_llm_provider] = lambda: mock_provider
        try:
            payload = dict(mock_sac_summary_request)
            payload["sac"] = {**payload["sac"], "promoted": True, "failure_reasons": []}
            response = client.post("/llm/sac-training-summary", json=payload)
            assert response.status_code == 200, response.text
            assert "Passed all SAC artifact health guardrails" in captured_prompts[0]
        finally:
            app.dependency_overrides.clear()

    def test_failure_reasons_render_in_prompt(
        self,
        mock_sac_summary_request,
        mock_sac_llm_json_response,
    ):
        """Non-promoted SAC run sends failure_reasons into the prompt.

        The guardrail copy is what stops the LLM from inventing
        prior-comparison narratives -- this test guards that contract.
        """
        import json

        captured_prompts: list[str] = []

        class CapturingProvider(MockLLMProvider):
            def generate(self, prompt: str) -> LLMResponse:
                captured_prompts.append(prompt)
                return super().generate(prompt)

        mock_provider = CapturingProvider(
            name="openai",
            response=LLMResponse(
                content=json.dumps(mock_sac_llm_json_response),
                model="gpt-5-mini",
                tokens_used=300,
            ),
        )
        app.dependency_overrides[get_llm_provider] = lambda: mock_provider
        try:
            payload = dict(mock_sac_summary_request)
            payload["sac"] = {
                **payload["sac"],
                "promoted": False,
                "failure_reasons": [
                    "eval_cagr 0.10 below floor 0.12",
                    "actor.pt missing or empty",
                ],
            }
            response = client.post("/llm/sac-training-summary", json=payload)
            assert response.status_code == 200, response.text
            prompt = captured_prompts[0]
            assert "NOT promoted" in prompt
            assert "eval_cagr 0.10 below floor 0.12" in prompt
            assert "actor.pt missing or empty" in prompt
            assert "guardrail-based" in prompt

        finally:
            app.dependency_overrides.clear()


class TestForecastersTrainingSummaryGuardrails:
    """Integration tests that verify forecaster prompt-rendering of guardrails."""

    def test_promoted_renders_guardrail_pass_prose(
        self,
        mock_forecasters_summary_request,
        mock_forecasters_llm_json_response,
    ):
        """Promoted forecaster runs render guardrail-pass copy in the prompt."""
        import json

        captured_prompts: list[str] = []

        class CapturingProvider(MockLLMProvider):
            def generate(self, prompt: str) -> LLMResponse:
                captured_prompts.append(prompt)
                return super().generate(prompt)

        mock_provider = CapturingProvider(
            name="openai",
            response=LLMResponse(
                content=json.dumps(mock_forecasters_llm_json_response),
                model="gpt-5-mini",
                tokens_used=300,
            ),
        )
        app.dependency_overrides[get_llm_provider] = lambda: mock_provider
        try:
            response = client.post(
                "/llm/forecasters-training-summary",
                json=mock_forecasters_summary_request,
            )
            assert response.status_code == 200, response.text
            prompt = captured_prompts[0]
            # Two passes, one per model.
            assert prompt.count("Passed all artifact health guardrails") == 2
            assert "guardrail-based" in prompt
        finally:
            app.dependency_overrides.clear()

    def test_failure_reasons_render_in_prompt(
        self,
        mock_forecasters_summary_request,
        mock_forecasters_llm_json_response,
    ):
        """Non-promoted forecaster sends failure_reasons into the prompt."""
        import json

        captured_prompts: list[str] = []

        class CapturingProvider(MockLLMProvider):
            def generate(self, prompt: str) -> LLMResponse:
                captured_prompts.append(prompt)
                return super().generate(prompt)

        mock_provider = CapturingProvider(
            name="openai",
            response=LLMResponse(
                content=json.dumps(mock_forecasters_llm_json_response),
                model="gpt-5-mini",
                tokens_used=300,
            ),
        )
        app.dependency_overrides[get_llm_provider] = lambda: mock_provider
        try:
            payload = dict(mock_forecasters_summary_request)
            payload["lstm"] = {
                **payload["lstm"],
                "promoted": False,
                "failure_reasons": ["val_loss is not finite"],
            }
            response = client.post(
                "/llm/forecasters-training-summary",
                json=payload,
            )
            assert response.status_code == 200, response.text
            prompt = captured_prompts[0]
            assert "NOT promoted" in prompt
            assert "val_loss is not finite" in prompt
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
        "universe": "halal_filtered",
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
    """Valid request payload for India Alpha-HRP summary endpoint.

    Mirrors the US Alpha-HRP fixture in tests/test_llm_us_alpha_hrp.py
    with NSE symbols. The shape is the unified
    :class:`AlphaHRPSummaryRequest` -- both markets share the same DTO
    post-parity, discriminated by the ``universe`` field.
    """
    return {
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
        "hold_threshold": 30,
    }


@pytest.fixture
def mock_india_llm_json_response():
    """Mock JSON response from LLM for India summary (4-paragraph schema)."""
    return {
        "para_1_market_outlook": "Top 25 PatchTST forecasts cluster around IT services and pharma.",
        "para_2_selection_rationale": "Sticky kept 12 NSE names; three new high-rank entrants.",
        "para_3_final_allocation": "HRP weights RELIANCE.NS=7.0%, TCS.NS=6.8%.",
        "para_4_risk_observations": "Watch INR/USD risk and small-cap NSE liquidity.",
        "para_5_stage_transition_insight": "TCS.NS jumped from alpha rank 12 to HRP weight rank 1 due to low correlation with IT peers in the basket.",
    }


class TestIndiaAlphaHRPSummaryEndpoint:
    """Tests for POST /llm/india-alpha-hrp-summary endpoint."""

    def test_successful_india_summary_generation(
        self,
        mock_india_weekly_summary_request,
        mock_india_llm_json_response,
    ):
        """Successful India Alpha-HRP summary generation."""
        import json

        mock_provider = MockLLMProvider(
            name="openai",
            response=LLMResponse(
                content=json.dumps(mock_india_llm_json_response),
                model="gpt-5-mini",
                tokens_used=350,
            ),
        )

        app.dependency_overrides[get_llm_provider] = lambda: mock_provider

        try:
            response = client.post(
                "/llm/india-alpha-hrp-summary",
                json=mock_india_weekly_summary_request,
            )

            assert response.status_code == 200, response.text
            data = response.json()
            assert "summary" in data
            assert data["provider"] == "openai"
            assert data["model_used"] == "gpt-5-mini"
            assert data["tokens_used"] == 350
            assert "para_1_market_outlook" in data["summary"]
            assert "para_2_selection_rationale" in data["summary"]
            assert "para_3_final_allocation" in data["summary"]
            assert "para_4_risk_observations" in data["summary"]
            assert "para_5_stage_transition_insight" in data["summary"]
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
        """Invalid JSON from LLM returns the para_1_market_outlook fallback stub."""
        mock_provider = MockLLMProvider(
            name="openai",
            response=LLMResponse(
                content="This is not valid JSON at all",
                model="gpt-5-mini",
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
            # The fallback uses para_1_market_outlook (matches US schema).
            assert "para_1_market_outlook" in data["summary"]
            assert (
                "Unable to generate AI summary"
                in data["summary"]["para_1_market_outlook"]
            )
        finally:
            app.dependency_overrides.clear()

    def test_india_summary_missing_required_fields_returns_422(self):
        """Empty request body fails Pydantic validation (no required fields)."""
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

    def test_india_summary_invalid_stage2_returns_422(self):
        """Invalid stage2 structure fails Pydantic validation."""
        mock_provider = MockLLMProvider(
            name="openai",
            response=LLMResponse(content="{}", model="test", tokens_used=0),
        )
        app.dependency_overrides[get_llm_provider] = lambda: mock_provider

        try:
            response = client.post(
                "/llm/india-alpha-hrp-summary",
                json={
                    "stage1_top_scores": [],
                    "model_version": "v",
                    "predicted_count": 0,
                    "requested_count": 0,
                    "selected_symbols": [],
                    "stage2": "not-an-hrp-allocation",
                    "universe": "halal_india_alpha",
                    "top_n": 15,
                    "hold_threshold": 30,
                },
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
                model="gpt-5-mini",
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
            assert data["model_used"] == "gpt-5-mini"
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
                model="gpt-5-mini",
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

    def test_weekly_summary_missing_universe_returns_422(
        self,
        mock_weekly_summary_request,
    ):
        """Universe is mandatory: 422 when omitted (AGENTS.md no-default)."""
        payload = dict(mock_weekly_summary_request)
        payload.pop("universe")

        response = client.post("/llm/sac-weekly-summary", json=payload)

        assert response.status_code == 422
        # Pydantic surfaces the missing-field error against the body.
        body = response.json()
        assert any("universe" in str(loc) for loc in body.get("detail", []))

    def test_weekly_summary_universe_halal_renders_in_prompt(
        self,
        mock_weekly_summary_request,
        mock_weekly_llm_json_response,
    ):
        """universe='halal' must reach the prompt template, not silently default."""
        import json

        captured_prompts = []

        class CapturingProvider(MockLLMProvider):
            def generate(self, prompt: str) -> LLMResponse:
                captured_prompts.append(prompt)
                return super().generate(prompt)

        mock_provider = CapturingProvider(
            name="openai",
            response=LLMResponse(
                content=json.dumps(mock_weekly_llm_json_response),
                model="gpt-5-mini",
                tokens_used=400,
            ),
        )

        payload = dict(mock_weekly_summary_request)
        payload["universe"] = "halal"

        app.dependency_overrides[get_llm_provider] = lambda: mock_provider
        try:
            response = client.post("/llm/sac-weekly-summary", json=payload)
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 200
        assert len(captured_prompts) == 1
        # The prompt template renders {{ universe }} as **<universe>**
        # in the A/B header so the LLM cannot conflate the two parallel
        # runs. (Both bucket names appear in the documentation list,
        # so we anchor on the bolded marker.)
        assert "**halal**" in captured_prompts[0]
        assert "**halal_filtered**" not in captured_prompts[0]


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
        "para_5_stage_transition_insight": "NVDA moved from Stage 1 rank 12 to Stage 2 rank 1 because the 252-day covariance over the chosen 15 isolated it from semi peers.",
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
                model="gpt-5-mini",
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
            assert data["model_used"] == "gpt-5-mini"
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
                content="not json", model="gpt-5-mini", tokens_used=50
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
