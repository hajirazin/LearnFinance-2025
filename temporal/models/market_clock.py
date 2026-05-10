"""Model for the GET /alpaca/clock endpoint.

Mirrors :class:`brain_api.routes.alpaca.MarketClockResponse` so the
sell-wait-buy helper can sleep until the next NYSE market open before
polling sell-order status. The clock is account-agnostic market data
(the brain_api route authenticates with the generic
``ALPACA_API_KEY`` / ``ALPACA_API_SECRET`` pair, not per-account
trading creds), so this Pydantic shape carries no account field.

``next_open`` and ``next_close`` are ISO-8601 strings (with timezone)
straight from Alpaca; callers parse them with
``datetime.fromisoformat`` inside the workflow.
"""

from pydantic import BaseModel


class MarketClockResponse(BaseModel):
    """Alpaca market clock state."""

    timestamp: str
    is_open: bool
    next_open: str
    next_close: str
