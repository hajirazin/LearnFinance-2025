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
# Client
from brain_api.core.fundamentals.client import (
    AlphaVantageClient,
    RealAlphaVantageClient,
)

# Fetcher
from brain_api.core.fundamentals.fetcher import FundamentalsFetcher

# Index
from brain_api.core.fundamentals.index import FundamentalsIndex

# Loader (shared by all consumers)
from brain_api.core.fundamentals.loader import (
    get_default_data_path,
    load_historical_fundamentals_from_cache,
)
from brain_api.core.fundamentals.models import (
    FetchRecord,
    FundamentalRatios,
    FundamentalsResult,
    QuarterlyStatement,
)

# Parser
from brain_api.core.fundamentals.parser import (
    compute_ratios,
    get_statement_as_of,
    parse_quarterly_statements,
)

# Storage
from brain_api.core.fundamentals.storage import (
    get_fundamentals_dir,
    load_raw_response,
    save_raw_response,
)

__all__ = [
    # Client
    "AlphaVantageClient",
    "FetchRecord",
    "FundamentalRatios",
    # Fetcher
    "FundamentalsFetcher",
    # Index
    "FundamentalsIndex",
    "FundamentalsResult",
    # Models
    "QuarterlyStatement",
    "RealAlphaVantageClient",
    "compute_ratios",
    # Loader (shared by all consumers)
    "get_default_data_path",
    # Storage
    "get_fundamentals_dir",
    "get_statement_as_of",
    "load_historical_fundamentals_from_cache",
    "load_raw_response",
    # Parser
    "parse_quarterly_statements",
    "save_raw_response",
]
