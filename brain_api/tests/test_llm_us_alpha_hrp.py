"""Tests for POST /llm/us-alpha-hrp-summary."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from brain_api.main import app
from brain_api.routes.llm.providers import (
    LLMProvider,
    LLMResponse,
    get_llm_provider,
)

client = TestClient(app)


class MockLLMProvider(LLMProvider):
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


@pytest.fixture
def alpha_request_body():
    """Valid request payload for /llm/us-alpha-hrp-summary."""
    return {
        "stage1_top_scores": [
            {"symbol": f"S{i:03d}", "score": 5.0 - 0.1 * i, "rank": i + 1}
            for i in range(20)
        ],
        "model_version": "v2026-04-26-abc",
        "predicted_count": 380,
        "requested_count": 410,
        "selected_symbols": [f"S{i:03d}" for i in range(15)],
        "kept_count": 12,
        "fillers_count": 3,
        "evicted_from_previous": {"OLD1": "rank_out_of_hold"},
        "previous_year_week_used": "202617",
        "stage2": {
            "percentage_weights": {f"S{i:03d}": 100.0 / 15 for i in range(15)},
            "symbols_used": 15,
            "symbols_excluded": [],
            "lookback_days": 252,
            "as_of_date": "2026-04-28",
        },
        "universe": "halal_new",
        "top_n": 15,
        "hold_threshold": 30,
    }


@pytest.fixture
def llm_summary_payload():
    return {
        "para_1_market_outlook": "Top 25 PatchTST forecasts cluster around tech.",
        "para_2_selection_rationale": "Sticky kept 12, three new high-rank entrants.",
        "para_3_final_allocation": "HRP weights AAA=8.5%, BBB=8.2%.",
        "para_4_risk_observations": "Watch K_hold=30 across regime shifts.",
    }


class TestUSAlphaHRPSummaryEndpoint:
    def test_happy_path_returns_summary(self, alpha_request_body, llm_summary_payload):
        provider = MockLLMProvider(
            name="openai",
            response=LLMResponse(
                content=json.dumps(llm_summary_payload),
                model="gpt-4o-mini",
                tokens_used=420,
            ),
        )
        app.dependency_overrides[get_llm_provider] = lambda: provider
        try:
            response = client.post(
                "/llm/us-alpha-hrp-summary",
                json=alpha_request_body,
            )
            assert response.status_code == 200, response.text
            data = response.json()
            assert "para_1_market_outlook" in data["summary"]
            assert data["provider"] == "openai"
            assert data["model_used"] == "gpt-4o-mini"
            assert data["tokens_used"] == 420
        finally:
            app.dependency_overrides.clear()

    def test_missing_required_field_returns_422(self):
        provider = MockLLMProvider(
            name="openai",
            response=LLMResponse(content="{}", model="t", tokens_used=0),
        )
        app.dependency_overrides[get_llm_provider] = lambda: provider
        try:
            # Missing stage2.
            response = client.post(
                "/llm/us-alpha-hrp-summary",
                json={
                    "stage1_top_scores": [],
                    "model_version": "v",
                    "predicted_count": 0,
                    "requested_count": 0,
                    "selected_symbols": [],
                    "universe": "halal_new",
                    "top_n": 15,
                    "hold_threshold": 30,
                },
            )
            assert response.status_code == 422
        finally:
            app.dependency_overrides.clear()

    def test_json_parse_fallback(self, alpha_request_body):
        # LLM returns non-JSON; endpoint must still respond 200 with a
        # graceful para_1 fallback message (matches existing pattern).
        provider = MockLLMProvider(
            name="openai",
            response=LLMResponse(
                content="not json", model="gpt-4o-mini", tokens_used=50
            ),
        )
        app.dependency_overrides[get_llm_provider] = lambda: provider
        try:
            response = client.post(
                "/llm/us-alpha-hrp-summary",
                json=alpha_request_body,
            )
            assert response.status_code == 200, response.text
            summary = response.json()["summary"]
            assert "para_1_market_outlook" in summary
            assert "Unable to generate AI summary" in summary["para_1_market_outlook"]
        finally:
            app.dependency_overrides.clear()

    def test_llm_failure_returns_503(self, alpha_request_body):
        provider = MockLLMProvider(
            name="openai",
            response=LLMResponse(content="", model="", tokens_used=None),
            error=Exception("LLM down"),
        )
        app.dependency_overrides[get_llm_provider] = lambda: provider
        try:
            response = client.post(
                "/llm/us-alpha-hrp-summary",
                json=alpha_request_body,
            )
            assert response.status_code == 503
            assert "LLM service unavailable" in response.json()["detail"]
        finally:
            app.dependency_overrides.clear()

    def test_skip_path_with_empty_selection(self, llm_summary_payload):
        # Skip path: selected_symbols empty + zero stage2 weights. The
        # endpoint should still render and the LLM still receive the
        # prompt. (The Temporal workflow uses this on the open-orders
        # gate path before sending the email.)
        provider = MockLLMProvider(
            name="openai",
            response=LLMResponse(
                content=json.dumps(llm_summary_payload),
                model="gpt-4o-mini",
                tokens_used=120,
            ),
        )
        app.dependency_overrides[get_llm_provider] = lambda: provider
        try:
            response = client.post(
                "/llm/us-alpha-hrp-summary",
                json={
                    "stage1_top_scores": [],
                    "model_version": "skipped",
                    "predicted_count": 0,
                    "requested_count": 0,
                    "selected_symbols": [],
                    "kept_count": 0,
                    "fillers_count": 0,
                    "evicted_from_previous": {},
                    "previous_year_week_used": None,
                    "stage2": {
                        "percentage_weights": {},
                        "symbols_used": 0,
                        "symbols_excluded": [],
                        "lookback_days": 0,
                        "as_of_date": "2026-04-28",
                    },
                    "universe": "halal_new",
                    "top_n": 15,
                    "hold_threshold": 30,
                },
            )
            assert response.status_code == 200, response.text
        finally:
            app.dependency_overrides.clear()
