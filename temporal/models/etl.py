"""Temporal-facing result for the asynchronous sentiment-gap refresh."""

from pydantic import BaseModel


class SentimentGapFillResponse(BaseModel):
    """Published result returned by ``run_sentiment_gap_fill``."""

    rows_added: int
    remaining_gaps: int
    gaps_pre_api_date: int
    duration_seconds: float
    hf_url: str
    published: bool = True
