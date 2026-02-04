"""LLM provider abstraction supporting OpenAI and OLLAMA."""

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx
from openai import OpenAI

logger = logging.getLogger(__name__)


# =============================================================================
# Response Model
# =============================================================================


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    content: str
    model: str
    tokens_used: int | None  # None for providers that don't report tokens


# =============================================================================
# Provider Interface
# =============================================================================


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g., 'openai', 'ollama')."""
        ...

    @abstractmethod
    def generate(self, prompt: str) -> LLMResponse:
        """Generate a response from the LLM."""
        ...


# =============================================================================
# OpenAI Provider
# =============================================================================


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider using the official SDK."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2500,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=self.api_key)

    @property
    def name(self) -> str:
        return "openai"

    def generate(self, prompt: str) -> LLMResponse:
        """Generate a response using OpenAI API."""
        logger.info(f"Calling OpenAI API with model={self.model}")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        content = response.choices[0].message.content or ""
        tokens_used = response.usage.total_tokens if response.usage else None

        logger.info(f"OpenAI response received, tokens_used={tokens_used}")

        return LLMResponse(
            content=content,
            model=self.model,
            tokens_used=tokens_used,
        )


# =============================================================================
# OLLAMA Provider
# =============================================================================


class OllamaProvider(LLMProvider):
    """OLLAMA LLM provider using REST API."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.3,
    ):
        self.base_url = base_url or os.environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.2")
        self.temperature = temperature

    @property
    def name(self) -> str:
        return "ollama"

    def generate(self, prompt: str) -> LLMResponse:
        """Generate a response using OLLAMA REST API."""
        logger.info(f"Calling OLLAMA API at {self.base_url} with model={self.model}")

        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
        }

        with httpx.Client(timeout=300.0) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

        content = data.get("response", "")

        # OLLAMA doesn't provide token counts in the same way
        # but we can estimate from eval_count if available
        tokens_used = data.get("eval_count")

        logger.info(f"OLLAMA response received, eval_count={tokens_used}")

        return LLMResponse(
            content=content,
            model=self.model,
            tokens_used=tokens_used,
        )


# =============================================================================
# Provider Factory
# =============================================================================


def get_llm_provider() -> LLMProvider:
    """Get the configured LLM provider based on environment variables.

    Environment variables:
        LLM_PROVIDER: "openai" (default) or "ollama"

    Returns:
        Configured LLM provider instance.

    Raises:
        ValueError: If LLM_PROVIDER is not recognized.
    """
    provider_name = os.environ.get("LLM_PROVIDER", "openai").lower()

    if provider_name == "openai":
        return OpenAIProvider()
    elif provider_name == "ollama":
        return OllamaProvider()
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: {provider_name}. Must be 'openai' or 'ollama'."
        )


def parse_json_response(content: str) -> dict:
    """Parse JSON from LLM response, handling markdown code blocks.

    Args:
        content: Raw LLM response content.

    Returns:
        Parsed JSON as a dictionary.

    Raises:
        ValueError: If JSON parsing fails.
    """
    # Strip markdown code blocks if present
    text = content.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from LLM response: {e}")
        logger.error(f"Raw content: {content[:500]}")
        raise ValueError(f"Failed to parse JSON from LLM response: {e}") from e
