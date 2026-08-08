"""Data freshness utilities for training.

Ensures training data is up-to-date before training begins by:
1. Filling news sentiment gaps in the parquet file
2. Refreshing fundamentals via SEC-first filing-head policy
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from brain_api.core.fundamentals.fetcher import (
    FundamentalsFetcher,
    has_usable_cached_quarters,
)
from brain_api.core.fundamentals.refresh_policy import (
    RefreshAction,
    SymbolCacheState,
    order_av_pull_queue,
)
from brain_api.etl.gap_fill import GapFillResult, fill_sentiment_gaps

logger = logging.getLogger(__name__)


def get_default_data_path() -> Path:
    """Get the default data path for brain_api."""
    return Path(__file__).parent.parent.parent / "data"


@dataclass
class FundamentalsRefreshResult:
    """Result of fundamentals refresh operation."""

    refreshed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    pending_new_filing: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    api_status: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataFreshnessResult:
    """Result of data freshness check."""

    sentiment_gaps_filled: int = 0
    sentiment_gaps_remaining: int = 0
    fundamentals_refreshed: list[str] = field(default_factory=list)
    fundamentals_skipped_today: list[str] = field(default_factory=list)
    fundamentals_failed: list[str] = field(default_factory=list)
    fundamentals_errors: dict[str, str] = field(default_factory=dict)
    duration_seconds: float = 0.0


def refresh_stale_fundamentals(
    symbols: list[str],
    base_path: Path | None = None,
    *,
    force_refresh: bool = False,
) -> FundamentalsRefreshResult:
    """Refresh fundamentals using filing-head freshness (not fetch-today).

    SHARED BY:
    - PUT /signals/fundamentals/historical endpoint
    - ensure_fresh_training_data() before training
    """
    if base_path is None:
        base_path = get_default_data_path()

    cache_dir = base_path / "cache"
    result = FundamentalsRefreshResult()

    sec_user_agent = os.environ.get("SEC_USER_AGENT", "").strip()
    if not sec_user_agent:
        logger.error(
            "[RefreshFundamentals] SEC_USER_AGENT not set; refusing unchecked "
            "fundamentals refresh"
        )
        for symbol in symbols:
            result.failed.append(symbol)
            result.errors[symbol] = (
                "SEC_USER_AGENT is required for point-in-time filing enrichment"
            )
        return result

    api_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
    fetcher = FundamentalsFetcher(
        api_key=api_key or "cache-enrichment-only",
        base_path=base_path,
        cache_dir=cache_dir,
    )

    try:
        pull_sec: list[str] = []
        pull_av_states: list[tuple[str, SymbolCacheState]] = []
        enrich_only: list[str] = []
        forced: set[str] = set(symbols) if force_refresh else set()

        for symbol in symbols:
            try:
                action = fetcher.decide_action_for_symbol(
                    symbol, force_refresh=force_refresh or symbol in forced
                )
            except Exception as exc:
                result.failed.append(symbol)
                result.errors[symbol] = str(exc)
                continue

            if action == RefreshAction.SKIP:
                result.skipped.append(symbol)
                continue
            if action == RefreshAction.ENRICH_ONLY:
                enrich_only.append(symbol)
                continue

            # PULL — classify SEC vs AV for ordering
            eligibility = None
            if fetcher.eligibility_client is not None:
                try:
                    eligibility = fetcher.eligibility_client.classify(symbol)
                except Exception as exc:
                    result.failed.append(symbol)
                    result.errors[symbol] = str(exc)
                    continue
            if eligibility is not None and eligibility.sec_eligible:
                pull_sec.append(symbol)
            else:
                state = (
                    SymbolCacheState.MISSING
                    if not has_usable_cached_quarters(base_path, symbol)
                    else SymbolCacheState.FILING_STALE
                )
                pull_av_states.append((symbol, state))

        for symbol in enrich_only:
            if symbol in result.failed:
                continue
            try:
                fetcher.fetch_symbol(symbol, force_refresh=False)
                if symbol.upper() in fetcher._pending_new_filing:
                    result.pending_new_filing.append(symbol)
                else:
                    result.refreshed.append(symbol)
                logger.info(f"[RefreshFundamentals] Enriched {symbol}")
            except Exception as e:
                logger.warning(f"[RefreshFundamentals] Failed to enrich {symbol}: {e}")
                result.failed.append(symbol)
                result.errors[symbol] = str(e)

        for symbol in pull_sec:
            if symbol in result.failed or symbol in result.refreshed:
                continue
            try:
                fetcher.fetch_symbol(symbol, force_refresh=force_refresh)
                if symbol.upper() in fetcher._pending_new_filing:
                    result.pending_new_filing.append(symbol)
                    result.skipped.append(symbol)
                else:
                    result.refreshed.append(symbol)
                logger.info(f"[RefreshFundamentals] Refreshed SEC {symbol}")
            except Exception as e:
                logger.warning(f"[RefreshFundamentals] Failed SEC {symbol}: {e}")
                result.failed.append(symbol)
                result.errors[symbol] = str(e)

        av_queue = order_av_pull_queue(pull_av_states, forced=forced)
        for symbol in av_queue:
            if symbol in result.failed:
                continue
            if not api_key:
                result.failed.append(symbol)
                result.errors[symbol] = (
                    "ALPHA_VANTAGE_API_KEY is required to refresh stale fundamentals"
                )
                continue
            try:
                fetcher.fetch_symbol(symbol, force_refresh=force_refresh)
                if symbol.upper() in fetcher._pending_new_filing:
                    result.pending_new_filing.append(symbol)
                    if symbol not in result.skipped:
                        result.skipped.append(symbol)
                else:
                    result.refreshed.append(symbol)
                logger.info(f"[RefreshFundamentals] Refreshed AV {symbol}")
            except Exception as e:
                logger.warning(f"[RefreshFundamentals] Failed AV {symbol}: {e}")
                result.failed.append(symbol)
                result.errors[symbol] = str(e)

        result.api_status = fetcher.get_api_status()
    finally:
        fetcher.close()

    # Deduplicate skipped/refreshed
    result.skipped = [s for s in result.skipped if s not in result.refreshed]

    logger.info(
        f"[RefreshFundamentals] Complete - refreshed: {len(result.refreshed)}, "
        f"skipped: {len(result.skipped)}, failed: {len(result.failed)}, "
        f"pending: {len(result.pending_new_filing)}"
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
    """Ensure training data is fresh before training."""
    start_time = time.time()
    result = DataFreshnessResult()

    if parquet_path is None:
        parquet_path = get_default_data_path() / "output" / "daily_sentiment.parquet"
    if fundamentals_base_path is None:
        fundamentals_base_path = get_default_data_path()

    logger.info(
        f"[DataFreshness] Ensuring fresh data for {len(symbols)} symbols, "
        f"window {start_date} to {end_date}"
    )

    logger.info("[DataFreshness] Phase 1: Checking news sentiment gaps...")
    try:
        if parquet_path.exists():
            gap_result: GapFillResult = fill_sentiment_gaps(
                universe=universe,
                start_date=start_date,
                end_date=end_date,
                parquet_path=parquet_path,
                local_only=True,
            )
            if gap_result.success:
                result.sentiment_gaps_filled = gap_result.progress.rows_added
                result.sentiment_gaps_remaining = gap_result.progress.gaps_pre_api_date
            else:
                logger.warning(
                    f"[DataFreshness] Sentiment gap fill failed: "
                    f"{gap_result.progress.error}"
                )
        else:
            logger.warning(
                f"[DataFreshness] Parquet file not found: {parquet_path}. "
                "Skipping sentiment gap fill."
            )
    except Exception as e:
        logger.warning(f"[DataFreshness] Sentiment gap fill failed: {e}")

    logger.info("[DataFreshness] Phase 2: Checking fundamentals freshness...")
    try:
        fund_result = refresh_stale_fundamentals(symbols, fundamentals_base_path)
        result.fundamentals_refreshed = fund_result.refreshed
        result.fundamentals_skipped_today = fund_result.skipped
        result.fundamentals_failed = fund_result.failed
        result.fundamentals_errors = fund_result.errors
    except Exception as e:
        logger.warning(f"[DataFreshness] Fundamentals refresh failed: {e}")
        result.fundamentals_failed = list(symbols)
        result.fundamentals_errors = {symbol: str(e) for symbol in symbols}

    result.duration_seconds = time.time() - start_time
    logger.info(
        f"[DataFreshness] Complete in {result.duration_seconds:.1f}s - "
        f"sentiment: {result.sentiment_gaps_filled} filled, "
        f"fundamentals: {len(result.fundamentals_refreshed)} refreshed, "
        f"{len(result.fundamentals_failed)} failed"
    )
    return result
