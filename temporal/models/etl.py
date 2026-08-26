"""Temporal-facing result for asynchronous news backfill.

The parse-only DTO lives in ``models.news`` so Temporal never
aggregates news events. This module re-exports it for callers that
historically imported the gap-fill response from ``models.etl``.
"""

from models.news import NewsBackfillResponse

__all__ = ["NewsBackfillResponse"]
