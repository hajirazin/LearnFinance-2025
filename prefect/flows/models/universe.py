"""Models for universe endpoints."""

from pydantic import BaseModel


class HalalStock(BaseModel):
    """A single stock in the halal universe."""

    symbol: str
    name: str
    max_weight: float
    sources: list[str]


class HalalUniverseResponse(BaseModel):
    """Response from GET /universe/halal endpoint."""

    stocks: list[HalalStock]
    etfs_used: list[str]
    total_stocks: int
    fetched_at: str

    @property
    def symbols(self) -> list[str]:
        """Get just the list of symbols."""
        return [stock.symbol for stock in self.stocks]
