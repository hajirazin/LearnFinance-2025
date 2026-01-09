"""Backward compatibility re-export.

The fundamentals module has been split into the brain_api.core.fundamentals package.
This module re-exports for backward compatibility.
"""

from brain_api.core.fundamentals import (
    AlphaVantageClient,
    FetchRecord,
    FundamentalRatios,
    FundamentalsFetcher,
    FundamentalsIndex,
    FundamentalsResult,
    QuarterlyStatement,
    RealAlphaVantageClient,
    compute_ratios,
    get_fundamentals_dir,
    get_statement_as_of,
    load_raw_response,
    parse_quarterly_statements,
    save_raw_response,
)

__all__ = [
    "AlphaVantageClient",
    "FetchRecord",
    "FundamentalRatios",
    "FundamentalsFetcher",
    "FundamentalsIndex",
    "FundamentalsResult",
    "QuarterlyStatement",
    "RealAlphaVantageClient",
    "compute_ratios",
    "get_fundamentals_dir",
    "get_statement_as_of",
    "load_raw_response",
    "parse_quarterly_statements",
    "save_raw_response",
]
