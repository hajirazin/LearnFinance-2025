"""Models for LLM endpoints."""

from pydantic import BaseModel


class TrainingSummaryResponse(BaseModel):
    """Response model for POST /llm/training-summary.

    Works with the brain_api /llm/training-summary endpoint.
    """

    summary: dict[str, str]  # Paragraph fields from LLM
    provider: str  # "openai" or "ollama"
    model_used: str  # e.g., "gpt-4o-mini" or "llama3.2"
    tokens_used: int | None  # Total tokens (None for OLLAMA)
