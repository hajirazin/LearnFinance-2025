"""Fundamentals module: SEC-first historical statements + AV fallback.

Historical refresh uses SEC CompanyFacts for SEC-eligible US names (CIK +
majority recent 10-K/10-Q), Alpha Vantage + SEC enrichment for non-eligible
names, and yfinance only on the inference (current) path.

Storage strategy:
- Raw JSON files: Store complete provider responses as source of truth
  Location: data/raw/fundamentals/{symbol}/income_statement.json
            data/raw/fundamentals/{symbol}/balance_sheet.json
- SQLite index: Track what's been fetched and when for quick lookups
  Location: data/cache/fundamentals.db

Freshness is filing-head based (cheap SEC submissions check), not fetch-today.
AV daily_limit is observability only — no local pre-gate.
"""

# Models
# Client
from brain_api.core.fundamentals.client import (
    AlphaVantageClient,
    AlphaVantageProviderError,
    RealAlphaVantageClient,
)

# Fetcher
from brain_api.core.fundamentals.fetcher import (
    FundamentalsConfigurationError,
    FundamentalsFetcher,
    FundamentalsProviderError,
    cached_fundamentals_require_sec_enrichment,
)

# Index
from brain_api.core.fundamentals.index import FundamentalsIndex

# Loader (shared by all consumers)
from brain_api.core.fundamentals.loader import (
    FundamentalsCacheError,
    get_default_data_path,
    load_historical_fundamentals_from_cache,
    load_point_in_time_fundamentals,
)
from brain_api.core.fundamentals.models import (
    FetchRecord,
    FundamentalRatios,
    FundamentalsResult,
    PointInTimeFundamental,
    QuarterlyStatement,
)

# Parser
from brain_api.core.fundamentals.parser import (
    compute_ratios,
    get_statement_as_of,
    parse_quarterly_statements,
)
from brain_api.core.fundamentals.refresh_policy import (
    RefreshAction,
    decide_refresh_action,
)
from brain_api.core.fundamentals.sec_eligibility import (
    EligibilityResult,
    FilingHead,
    SECEligibilityClient,
)
from brain_api.core.fundamentals.sec_statements import (
    SECStatementError,
    build_statement_payloads_from_companyfacts,
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
    "AlphaVantageProviderError",
    "EligibilityResult",
    "FetchRecord",
    "FilingHead",
    "FundamentalRatios",
    "FundamentalsCacheError",
    # Fetcher
    "FundamentalsConfigurationError",
    "FundamentalsFetcher",
    # Index
    "FundamentalsIndex",
    "FundamentalsProviderError",
    "FundamentalsResult",
    "PointInTimeFundamental",
    # Models
    "QuarterlyStatement",
    "RealAlphaVantageClient",
    "RefreshAction",
    "SECEligibilityClient",
    "SECStatementError",
    "build_statement_payloads_from_companyfacts",
    "cached_fundamentals_require_sec_enrichment",
    "compute_ratios",
    "decide_refresh_action",
    # Loader (shared by all consumers)
    "get_default_data_path",
    # Storage
    "get_fundamentals_dir",
    "get_statement_as_of",
    "load_historical_fundamentals_from_cache",
    "load_point_in_time_fundamentals",
    "load_raw_response",
    # Parser
    "parse_quarterly_statements",
    "save_raw_response",
]
