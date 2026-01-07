"""Fundamentals module for Alpha Vantage data fetching and analysis.

This module provides tools for fetching and analyzing fundamental financial data
from Alpha Vantage, including income statements and balance sheets.

Storage strategy:
- Raw JSON files: Store complete API responses as source of truth
  Location: data/raw/fundamentals/{symbol}/income_statement.json
            data/raw/fundamentals/{symbol}/balance_sheet.json
- SQLite index: Track what's been fetched and when for quick lookups
  Location: data/cache/fundamentals.db

This approach ensures:
1. No data loss if parsing/schema changes
2. Efficient lookups without re-parsing all files
3. Rate limit awareness (25/day free tier)
"""

# Models
from brain_api.core.fundamentals.models import (
    FetchRecord,
    FundamentalRatios,
    FundamentalsResult,
    QuarterlyStatement,
)

# Index
from brain_api.core.fundamentals.index import FundamentalsIndex

# Client
from brain_api.core.fundamentals.client import (
    AlphaVantageClient,
    RealAlphaVantageClient,
)

# Storage
from brain_api.core.fundamentals.storage import (
    get_fundamentals_dir,
    load_raw_response,
    save_raw_response,
)

# Parser
from brain_api.core.fundamentals.parser import (
    compute_ratios,
    get_statement_as_of,
    parse_quarterly_statements,
)

# Fetcher
from brain_api.core.fundamentals.fetcher import FundamentalsFetcher

__all__ = [
    # Models
    "QuarterlyStatement",
    "FundamentalRatios",
    "FetchRecord",
    "FundamentalsResult",
    # Index
    "FundamentalsIndex",
    # Client
    "AlphaVantageClient",
    "RealAlphaVantageClient",
    # Storage
    "get_fundamentals_dir",
    "save_raw_response",
    "load_raw_response",
    # Parser
    "parse_quarterly_statements",
    "get_statement_as_of",
    "compute_ratios",
    # Fetcher
    "FundamentalsFetcher",
]


