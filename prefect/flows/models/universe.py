"""Models for universe endpoints."""

from pydantic import BaseModel


class HalalUniverseResponse(BaseModel):
    """Response from GET /universe/halal endpoint."""

    symbols: list[str]
    count: int
    source: str
