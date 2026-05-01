"""Models for LLM endpoints."""

from pydantic import BaseModel


class TrainingSummaryResponse(BaseModel):
    """Response model for the split training-summary LLM endpoints.

    Works with both ``/llm/forecasters-training-summary`` and
    ``/llm/sac-training-summary`` (and is shape-compatible with
    ``/llm/india-training-summary``). Returned by the matching
    activities in ``temporal/activities/training.py``.
    """

    summary: dict[str, str]  # Paragraph fields from LLM
    provider: str  # "openai" or "ollama"
    model_used: str  # e.g., "gpt-4o-mini" or "llama3.2"
    tokens_used: int | None  # Total tokens (None for OLLAMA)
