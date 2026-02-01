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

from brain_api.core.fundamentals.fetcher import FundamentalsFetcher
from brain_api.core.fundamentals.index import FundamentalsIndex
from brain_api.etl.gap_fill import GapFillResult, fill_sentiment_gaps

logger = logging.getLogger(__name__)


def get_default_data_path() -> Path:
    """Get the default data path for brain_api."""
    return Path(__file__).parent.parent.parent / "data"


@dataclass
class DataFreshnessResult:
    """Result of data freshness check."""

    sentiment_gaps_filled: int = 0
    sentiment_gaps_remaining: int = 0  # Pre-2015 gaps that can't be filled
    fundamentals_refreshed: list[str] = field(default_factory=list)
    fundamentals_skipped_today: list[str] = field(default_factory=list)
    fundamentals_failed: list[str] = field(default_factory=list)
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


def ensure_fresh_training_data(
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

    cache_dir = fundamentals_base_path / "cache"

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
        # Find symbols that need to be refreshed
        symbols_to_fetch = get_symbols_not_fetched_today(symbols, cache_dir)
        result.fundamentals_skipped_today = [
            s for s in symbols if s not in symbols_to_fetch
        ]

        if not symbols_to_fetch:
            logger.info(
                f"[DataFreshness] All {len(symbols)} symbols have fresh fundamentals"
            )
        else:
            logger.info(
                f"[DataFreshness] Refreshing fundamentals for {len(symbols_to_fetch)} symbols: "
                f"{symbols_to_fetch}"
            )

            # Get Alpha Vantage API key
            api_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
            if not api_key:
                logger.warning(
                    "[DataFreshness] ALPHA_VANTAGE_API_KEY not set, skipping fundamentals refresh"
                )
            else:
                fetcher = FundamentalsFetcher(
                    api_key=api_key,
                    base_path=fundamentals_base_path / "raw" / "fundamentals",
                    cache_dir=cache_dir,
                )

                try:
                    # Fetch each symbol, continue on failure
                    for symbol in symbols_to_fetch:
                        try:
                            fetcher.fetch_symbol(symbol)
                            result.fundamentals_refreshed.append(symbol)
                            logger.info(
                                f"[DataFreshness] Refreshed fundamentals for {symbol}"
                            )
                        except Exception as e:
                            logger.warning(
                                f"[DataFreshness] Failed to fetch fundamentals for {symbol}: {e}"
                            )
                            result.fundamentals_failed.append(symbol)
                            # Continue to next symbol, don't break
                finally:
                    fetcher.close()

    except Exception as e:
        logger.warning(f"[DataFreshness] Fundamentals refresh failed: {e}")

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
