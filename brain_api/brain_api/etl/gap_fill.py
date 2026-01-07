"""Gap filling for sentiment data.

Orchestrates filling missing sentiment data by:
1. Fetching news from Alpaca API
2. Scoring with FinBERT
3. Appending results to the parquet file
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from brain_api.core.config import SENTIMENT_BACKFILL_MAX_API_CALLS
from brain_api.core.finbert import FinBERTScorer, SentimentScore
from brain_api.core.news_api.alpaca import ALPACA_EARLIEST_DATE, AlpacaNewsClient
from brain_api.etl.gap_detection import categorize_gaps, find_gaps, get_gap_statistics
from brain_api.etl.parquet_writer import OUTPUT_SCHEMA
from brain_api.universe.halal import get_halal_symbols

logger = logging.getLogger(__name__)


@dataclass
class GapFillProgress:
    """Progress tracking for gap fill operation."""

    total_gaps: int = 0
    gaps_fillable: int = 0
    gaps_pre_api_date: int = 0
    api_calls_made: int = 0
    api_calls_limit: int = SENTIMENT_BACKFILL_MAX_API_CALLS
    articles_fetched: int = 0
    articles_scored: int = 0
    rows_added: int = 0
    remaining_gaps: int = 0
    status: str = "pending"
    error: str | None = None
    current_phase: str = "initializing"


@dataclass
class GapFillResult:
    """Result of gap fill operation."""

    success: bool
    progress: GapFillProgress
    statistics: dict[str, Any] = field(default_factory=dict)
    parquet_updated: bool = False


def _group_gaps_by_date(
    gaps: list[tuple[date, str]],
) -> dict[date, list[str]]:
    """Group gaps by date for efficient API calls.

    Args:
        gaps: List of (date, symbol) tuples

    Returns:
        Dict mapping date -> list of symbols with gaps on that date
    """
    date_to_symbols: dict[date, list[str]] = {}
    for gap_date, symbol in gaps:
        if gap_date not in date_to_symbols:
            date_to_symbols[gap_date] = []
        date_to_symbols[gap_date].append(symbol)
    return date_to_symbols


def _aggregate_daily_sentiment(
    articles_with_scores: list[tuple[datetime, str, SentimentScore]],
) -> list[dict]:
    """Aggregate article scores into daily sentiment per symbol.

    Args:
        articles_with_scores: List of (article_datetime, symbol, score) tuples

    Returns:
        List of dicts ready for parquet output
    """
    # Group by (date, symbol)
    aggregations: dict[tuple[date, str], list[SentimentScore]] = {}

    for article_dt, symbol, score in articles_with_scores:
        article_date = article_dt.date() if isinstance(article_dt, datetime) else article_dt
        key = (article_date, symbol)
        if key not in aggregations:
            aggregations[key] = []
        aggregations[key].append(score)

    # Compute aggregated sentiment for each (date, symbol)
    results = []
    for (agg_date, symbol), scores in aggregations.items():
        if not scores:
            continue

        # Confidence-weighted average score
        total_weight = sum(s.confidence for s in scores)
        if total_weight > 0:
            weighted_score = sum(s.score * s.confidence for s in scores) / total_weight
        else:
            weighted_score = sum(s.score for s in scores) / len(scores)

        results.append({
            "date": agg_date,
            "symbol": symbol,
            "sentiment_score": round(weighted_score, 4),
            "article_count": len(scores),
            "avg_confidence": round(sum(s.confidence for s in scores) / len(scores), 4),
            "p_pos_avg": round(sum(s.p_pos for s in scores) / len(scores), 4),
            "p_neg_avg": round(sum(s.p_neg for s in scores) / len(scores), 4),
            "total_articles": len(scores),
        })

    return results


def _append_to_parquet(
    new_rows: list[dict],
    parquet_path: Path,
) -> int:
    """Append new rows to the parquet file.

    Args:
        new_rows: List of row dicts to append
        parquet_path: Path to the parquet file

    Returns:
        Number of rows added
    """
    if not new_rows:
        return 0

    # Create DataFrame from new rows
    new_df = pd.DataFrame(new_rows)
    new_df["date"] = pd.to_datetime(new_df["date"]).dt.date

    # Read existing data if file exists
    if parquet_path.exists():
        existing_df = pd.read_parquet(parquet_path)
        # Convert date column if needed
        if existing_df["date"].dtype == "object":
            existing_df["date"] = pd.to_datetime(existing_df["date"]).dt.date

        # Merge: existing + new, deduplicate by (date, symbol)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=["date", "symbol"], keep="last")
    else:
        combined_df = new_df

    # Sort by date and symbol
    combined_df = combined_df.sort_values(["date", "symbol"]).reset_index(drop=True)

    # Write back to parquet
    table = pa.Table.from_pandas(combined_df, schema=OUTPUT_SCHEMA)
    pq.write_table(table, parquet_path, compression="snappy")

    return len(new_rows)


def fill_sentiment_gaps(
    start_date: date,
    end_date: date,
    parquet_path: Path,
    progress_callback: Callable[[GapFillProgress], None] | None = None,
) -> GapFillResult:
    """Fill missing sentiment data in the parquet file.

    Args:
        start_date: Earliest date to check for gaps
        end_date: Latest date to check for gaps
        parquet_path: Path to daily_sentiment.parquet
        progress_callback: Optional callback for progress updates

    Returns:
        GapFillResult with statistics and success status
    """
    progress = GapFillProgress()
    progress.api_calls_limit = SENTIMENT_BACKFILL_MAX_API_CALLS

    def update_progress():
        if progress_callback:
            progress_callback(progress)

    try:
        # Phase 1: Get halal symbols
        progress.current_phase = "getting_symbols"
        update_progress()

        symbols = get_halal_symbols()
        if not symbols:
            progress.error = "No halal symbols found"
            progress.status = "failed"
            return GapFillResult(success=False, progress=progress)

        logger.info(f"Found {len(symbols)} halal symbols")

        # Phase 2: Detect gaps
        progress.current_phase = "detecting_gaps"
        update_progress()

        all_gaps = find_gaps(symbols, start_date, end_date, parquet_path)
        progress.total_gaps = len(all_gaps)

        fillable_gaps, unfillable_gaps = categorize_gaps(all_gaps, ALPACA_EARLIEST_DATE)
        progress.gaps_fillable = len(fillable_gaps)
        progress.gaps_pre_api_date = len(unfillable_gaps)

        logger.info(
            f"Found {len(all_gaps)} gaps: "
            f"{len(fillable_gaps)} fillable, {len(unfillable_gaps)} pre-2015"
        )

        if not fillable_gaps:
            progress.status = "completed"
            progress.current_phase = "done"
            progress.remaining_gaps = 0
            update_progress()

            statistics = get_gap_statistics(
                symbols, start_date, end_date, parquet_path, ALPACA_EARLIEST_DATE
            )
            return GapFillResult(
                success=True,
                progress=progress,
                statistics=statistics,
                parquet_updated=False,
            )

        # Phase 3: Fetch news from Alpaca and score
        progress.current_phase = "fetching_and_scoring"
        update_progress()

        alpaca_client = AlpacaNewsClient()
        scorer = FinBERTScorer()

        # Group gaps by date for efficient API calls
        date_to_symbols = _group_gaps_by_date(fillable_gaps)

        # Sort dates in reverse chronological order (most recent first)
        sorted_dates = sorted(date_to_symbols.keys(), reverse=True)

        articles_with_scores: list[tuple[datetime, str, SentimentScore]] = []

        for gap_date in sorted_dates:
            if progress.api_calls_made >= progress.api_calls_limit:
                logger.info(f"API call limit reached ({progress.api_calls_limit})")
                break

            gap_symbols = date_to_symbols[gap_date]

            # Fetch news for this date and symbols
            articles = alpaca_client.fetch_news_for_date(
                symbols=gap_symbols,
                target_date=gap_date,
                limit=50,
            )
            progress.api_calls_made = alpaca_client.call_count
            progress.articles_fetched += len(articles)

            update_progress()

            if not articles:
                continue

            # Score articles with FinBERT
            texts = [
                f"{a.headline} {a.summary}".strip() if a.summary else a.headline
                for a in articles
            ]
            scores = scorer.score_batch(texts)
            progress.articles_scored += len(scores)

            # Map articles to their symbols and dates
            for article, score in zip(articles, scores):
                # Each article may be relevant to multiple symbols
                for symbol in article.symbols:
                    if symbol in gap_symbols:
                        articles_with_scores.append(
                            (article.created_at, symbol, score)
                        )

            update_progress()

        # Phase 4: Aggregate and write to parquet
        progress.current_phase = "writing_parquet"
        update_progress()

        new_rows = _aggregate_daily_sentiment(articles_with_scores)
        rows_added = _append_to_parquet(new_rows, parquet_path)
        progress.rows_added = rows_added

        # Calculate remaining gaps
        filled_date_symbols = {(row["date"], row["symbol"]) for row in new_rows}
        remaining = [g for g in fillable_gaps if g not in filled_date_symbols]
        progress.remaining_gaps = len(remaining)

        progress.status = "completed"
        progress.current_phase = "done"
        update_progress()

        statistics = get_gap_statistics(
            symbols, start_date, end_date, parquet_path, ALPACA_EARLIEST_DATE
        )

        return GapFillResult(
            success=True,
            progress=progress,
            statistics=statistics,
            parquet_updated=rows_added > 0,
        )

    except Exception as e:
        logger.exception("Gap fill failed")
        progress.status = "failed"
        progress.error = str(e)
        update_progress()
        return GapFillResult(success=False, progress=progress)

