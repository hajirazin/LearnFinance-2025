"""Models for model metadata endpoints."""

from pydantic import BaseModel


class ActiveSymbolsResponse(BaseModel):
    """Response from GET /models/active-symbols endpoint."""

    symbols: list[str]
    source_model: str
    model_version: str
