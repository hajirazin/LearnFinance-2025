"""Data freshness utilities for training.

Ensures training data is up-to-date before training begins by:
1. Filling news sentiment gaps in the parquet file
2. Refreshing fundamentals that haven't been fetched today
"""

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from brain_api.core.fundamentals.fetcher import (
    FundamentalsFetcher,
    cached_fundamentals_require_sec_enrichment,
)
from brain_api.core.fundamentals.index import FundamentalsIndex
from brain_api.etl.gap_fill import GapFillResult, fill_sentiment_gaps

logger = logging.getLogger(__name__)


def get_default_data_path() -> Path:
    """Get the default data path for brain_api."""
    return Path(__file__).parent.parent.parent / "data"


@dataclass
class FundamentalsRefreshResult:
    """Result of fundamentals refresh operation.

    Shared by:
    - PUT /signals/fundamentals/historical endpoint
    - ensure_fresh_training_data() before training
    """

    refreshed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # Already fetched today
    failed: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    api_status: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataFreshnessResult:
    """Result of data freshness check."""

    sentiment_gaps_filled: int = 0
    sentiment_gaps_remaining: int = 0  # Pre-2015 gaps that can't be filled
    fundamentals_refreshed: list[str] = field(default_factory=list)
    fundamentals_skipped_today: list[str] = field(default_factory=list)
    fundamentals_failed: list[str] = field(default_factory=list)
    fundamentals_errors: dict[str, str] = field(default_factory=dict)
    duration_seconds: float = 0.0


def get_symbols_not_fetched_today(
    symbols: list[str],
    cache_dir: Path,
) -> list[str]:
    """Find symbols whose fundamentals were NOT fetched today.

    Args:
        symbols: List of symbols to check
        cache_dir: Directory containing fundamentals.db

    Returns:
        List of symbols that need to be fetched
    """
    index = FundamentalsIndex(cache_dir)
    not_fetched_today: list[str] = []
    today = date.today()

    try:
        for symbol in symbols:
            record = index.get_fetch_record(symbol, "income_statement")
            if record is None:
                not_fetched_today.append(symbol)
            else:
                # Parse the ISO timestamp to get the date
                # Format: 2026-01-05T07:47:12.286617+00:00
                fetched_date_str = record.fetched_at.split("T")[0]
                fetched_date = date.fromisoformat(fetched_date_str)
                if fetched_date < today:
                    not_fetched_today.append(symbol)
                # else: fetched today, skip
    finally:
        index.close()

    return not_fetched_today


def refresh_stale_fundamentals(
    symbols: list[str],
    base_path: Path | None = None,
) -> FundamentalsRefreshResult:
    """Refresh fundamentals for symbols not fetched today.

    SHARED BY:
    - PUT /signals/fundamentals/historical endpoint
    - ensure_fresh_training_data() before training

    Only fetches symbols that haven't been fetched today.
    Uses Alpha Vantage API (requires ALPHA_VANTAGE_API_KEY env var).

    Args:
        symbols: List of symbols to potentially refresh
        base_path: Base path for fundamentals data (defaults to brain_api/data/)

    Returns:
        FundamentalsRefreshResult with refresh statistics
    """
    if base_path is None:
        base_path = get_default_data_path()

    cache_dir = base_path / "cache"
    result = FundamentalsRefreshResult()

    # A legacy cache with unresolved filing provenance must be enriched even
    # when its Alpha Vantage index row says it was fetched today.
    stale_symbols = get_symbols_not_fetched_today(symbols, cache_dir)
    enrichment_symbols = []
    for symbol in symbols:
        try:
            if cached_fundamentals_require_sec_enrichment(base_path, symbol):
                enrichment_symbols.append(symbol)
        except Exception as exc:
            result.failed.append(symbol)
            result.errors[symbol] = f"Fundamentals cache inspection failed: {exc}"
    symbols_to_fetch = [
        symbol
        for symbol in symbols
        if symbol in set(stale_symbols) | set(enrichment_symbols)
    ]
    result.skipped = [symbol for symbol in symbols if symbol not in symbols_to_fetch]

    if not symbols_to_fetch:
        logger.info(
            f"[RefreshFundamentals] All {len(symbols)} symbols already fetched today"
        )
        # Get API status even if nothing to fetch
        api_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
        if api_key:
            index = FundamentalsIndex(cache_dir)
            try:
                calls_today = index.get_api_calls_today()
                result.api_status = {
                    "calls_today": calls_today,
                    "daily_limit": 25,
                    "remaining": max(0, 25 - calls_today),
                }
            finally:
                index.close()
        return result

    logger.info(
        f"[RefreshFundamentals] Refreshing {len(symbols_to_fetch)} symbols: {symbols_to_fetch}"
    )

    sec_user_agent = os.environ.get("SEC_USER_AGENT", "").strip()
    if not sec_user_agent:
        logger.error(
            "[RefreshFundamentals] SEC_USER_AGENT not set; refusing unchecked "
            "fundamentals refresh"
        )
        for symbol in symbols_to_fetch:
            if symbol not in result.failed:
                result.failed.append(symbol)
            result.errors.setdefault(
                symbol, "SEC_USER_AGENT is required for point-in-time filing enrichment"
            )
        return result

    # Get Alpha Vantage API key. Cache-only SEC enrichment does not consume AV
    # quota, but every stale symbol requires a fresh AV response.
    api_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
    stale_without_key = [symbol for symbol in stale_symbols if not api_key]
    if stale_without_key:
        logger.warning(
            "[RefreshFundamentals] ALPHA_VANTAGE_API_KEY not set; stale symbols "
            f"cannot refresh: {stale_without_key}"
        )
        for symbol in stale_without_key:
            if symbol not in result.failed:
                result.failed.append(symbol)
            result.errors.setdefault(
                symbol,
                "ALPHA_VANTAGE_API_KEY is required to refresh stale fundamentals",
            )

    if all(symbol in result.failed for symbol in symbols_to_fetch):
        return result

    fetcher = FundamentalsFetcher(
        api_key=api_key or "cache-enrichment-only",
        base_path=base_path,
        cache_dir=cache_dir,
    )

    try:
        # Fetch each symbol, continue on failure
        for symbol in symbols_to_fetch:
            if symbol in result.failed:
                continue
            try:
                fetcher.fetch_symbol(
                    symbol,
                    force_refresh=symbol in stale_symbols,
                )
                result.refreshed.append(symbol)
                logger.info(f"[RefreshFundamentals] Refreshed {symbol}")
            except Exception as e:
                logger.warning(f"[RefreshFundamentals] Failed to fetch {symbol}: {e}")
                result.failed.append(symbol)
                result.errors[symbol] = str(e)

        # Get API status
        result.api_status = fetcher.get_api_status()
    finally:
        fetcher.close()

    logger.info(
        f"[RefreshFundamentals] Complete - refreshed: {len(result.refreshed)}, "
        f"skipped: {len(result.skipped)}, failed: {len(result.failed)}"
    )

    return result


def ensure_fresh_training_data(
    universe: str,
    symbols: list[str],
    start_date: date,
    end_date: date,
    parquet_path: Path | None = None,
    fundamentals_base_path: Path | None = None,
) -> DataFreshnessResult:
    """Ensure training data is fresh before training.

    1. Fills news sentiment gaps (2015+ via Alpaca API)
    2. Refreshes fundamentals not fetched today

    Called automatically by training endpoints.

    Args:
        universe: Registered ETL universe string -- forwarded to
            ``fill_sentiment_gaps`` so the gap fill scopes its symbol
            slate to the same universe the caller is training on.
        symbols: List of symbols to ensure data for
        start_date: Training window start date
        end_date: Training window end date
        parquet_path: Path to daily_sentiment.parquet (defaults to brain_api/data/output/)
        fundamentals_base_path: Base path for fundamentals data (defaults to brain_api/data/)

    Returns:
        DataFreshnessResult with statistics on what was refreshed
    """
    start_time = time.time()
    result = DataFreshnessResult()

    # Set default paths
    if parquet_path is None:
        parquet_path = get_default_data_path() / "output" / "daily_sentiment.parquet"
    if fundamentals_base_path is None:
        fundamentals_base_path = get_default_data_path()

    logger.info(
        f"[DataFreshness] Ensuring fresh data for {len(symbols)} symbols, "
        f"window {start_date} to {end_date}"
    )

    # ==========================================================================
    # Phase 1: Fill news sentiment gaps
    # ==========================================================================
    logger.info("[DataFreshness] Phase 1: Checking news sentiment gaps...")

    try:
        if parquet_path.exists():
            gap_result: GapFillResult = fill_sentiment_gaps(
                universe=universe,
                start_date=start_date,
                end_date=end_date,
                parquet_path=parquet_path,
                local_only=True,  # Don't upload to HuggingFace during training
            )

            if gap_result.success:
                result.sentiment_gaps_filled = gap_result.progress.rows_added
                result.sentiment_gaps_remaining = gap_result.progress.gaps_pre_api_date
                logger.info(
                    f"[DataFreshness] Sentiment gaps filled: {result.sentiment_gaps_filled}, "
                    f"remaining (pre-2015): {result.sentiment_gaps_remaining}"
                )
            else:
                logger.warning(
                    f"[DataFreshness] Sentiment gap fill failed: {gap_result.progress.error}"
                )
        else:
            logger.warning(
                f"[DataFreshness] Parquet file not found: {parquet_path}. "
                "Skipping sentiment gap fill."
            )
    except Exception as e:
        logger.warning(f"[DataFreshness] Sentiment gap fill failed: {e}")

    # ==========================================================================
    # Phase 2: Refresh fundamentals not fetched today
    # ==========================================================================
    logger.info("[DataFreshness] Phase 2: Checking fundamentals freshness...")

    try:
        # Use shared refresh_stale_fundamentals function
        fund_result = refresh_stale_fundamentals(symbols, fundamentals_base_path)
        result.fundamentals_refreshed = fund_result.refreshed
        result.fundamentals_skipped_today = fund_result.skipped
        result.fundamentals_failed = fund_result.failed
        result.fundamentals_errors = fund_result.errors
    except Exception as e:
        logger.warning(f"[DataFreshness] Fundamentals refresh failed: {e}")
        result.fundamentals_failed = list(symbols)
        result.fundamentals_errors = {symbol: str(e) for symbol in symbols}

    # ==========================================================================
    # Done
    # ==========================================================================
    result.duration_seconds = time.time() - start_time

    logger.info(
        f"[DataFreshness] Complete in {result.duration_seconds:.1f}s - "
        f"sentiment: {result.sentiment_gaps_filled} filled, "
        f"fundamentals: {len(result.fundamentals_refreshed)} refreshed, "
        f"{len(result.fundamentals_failed)} failed"
    )

    return result
