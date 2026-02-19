"""Gap filling for sentiment data.

Orchestrates filling missing sentiment data by:
1. Fetching news from Alpaca API
2. Scoring with FinBERT
3. Appending results to the parquet file
"""

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from brain_api.core.config import UniverseType, get_etl_universe
from brain_api.core.finbert import FinBERTScorer, SentimentScore
from brain_api.core.news_api.alpaca import ALPACA_EARLIEST_DATE, AlpacaNewsClient
from brain_api.etl.gap_detection import categorize_gaps, find_gaps, get_gap_statistics
from brain_api.etl.parquet_writer import OUTPUT_SCHEMA
from brain_api.universe import (
    get_halal_filtered_symbols,
    get_halal_new_symbols,
    get_halal_symbols,
    get_sp500_symbols,
)

logger = logging.getLogger(__name__)

# Checkpoint interval: save to parquet every N API calls
CHECKPOINT_INTERVAL = 100


@dataclass
class GapFillProgress:
    """Progress tracking for gap fill operation."""

    total_gaps: int = 0
    gaps_fillable: int = 0
    gaps_pre_api_date: int = 0
    api_calls_made: int = 0
    articles_fetched: int = 0
    articles_scored: int = 0
    rows_added: int = 0
    remaining_gaps: int = 0
    checkpoints_saved: int = 0
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
    hf_url: str | None = None


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
        article_date = (
            article_dt.date() if isinstance(article_dt, datetime) else article_dt
        )
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

        results.append(
            {
                "date": agg_date,
                "symbol": symbol,
                "sentiment_score": round(weighted_score, 4),
                "article_count": len(scores),
                "avg_confidence": round(
                    sum(s.confidence for s in scores) / len(scores), 4
                ),
                "p_pos_avg": round(sum(s.p_pos for s in scores) / len(scores), 4),
                "p_neg_avg": round(sum(s.p_neg for s in scores) / len(scores), 4),
                "total_articles": len(scores),
            }
        )

    return results


def _create_zero_article_rows(
    checked_gaps: list[tuple[date, str]],
) -> list[dict]:
    """Create parquet rows for gaps that were checked but had no articles.

    Args:
        checked_gaps: List of (date, symbol) tuples that were checked

    Returns:
        List of dicts with article_count=0 ready for parquet output
    """
    return [
        {
            "date": gap_date,
            "symbol": symbol,
            "sentiment_score": 0.0,
            "article_count": 0,
            "avg_confidence": 0.0,
            "p_pos_avg": 0.0,
            "p_neg_avg": 0.0,
            "total_articles": 0,
        }
        for gap_date, symbol in checked_gaps
    ]


def append_to_parquet(
    new_rows: list[dict],
    parquet_path: Path,
) -> int:
    """Append new rows to the parquet file with merge and deduplication.

    This function is used by both the main ETL pipeline and gap fill to
    ensure parquet writes are incremental rather than overwriting.

    Args:
        new_rows: List of row dicts to append
        parquet_path: Path to the parquet file

    Returns:
        Number of rows added
    """
    if not new_rows:
        logger.info("No new rows to append to parquet")
        return 0

    # Create DataFrame from new rows
    new_df = pd.DataFrame(new_rows)
    new_df["date"] = pd.to_datetime(new_df["date"]).dt.date

    # Read existing data if file exists
    if parquet_path.exists():
        existing_df = pd.read_parquet(parquet_path)
        existing_count = len(existing_df)
        # Convert date column if needed
        if existing_df["date"].dtype == "object":
            existing_df["date"] = pd.to_datetime(existing_df["date"]).dt.date

        # Merge: existing + new, deduplicate by (date, symbol)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(
            subset=["date", "symbol"], keep="last"
        )
        logger.info(
            f"Merging {len(new_rows)} new rows with {existing_count:,} existing rows"
        )
    else:
        combined_df = new_df
        logger.info(f"Creating new parquet file with {len(new_rows)} rows")

    # Sort by date and symbol
    combined_df = combined_df.sort_values(["date", "symbol"]).reset_index(drop=True)

    # Write back to parquet
    table = pa.Table.from_pandas(combined_df, schema=OUTPUT_SCHEMA)
    pq.write_table(table, parquet_path, compression="snappy")
    logger.info(f"Wrote {len(combined_df):,} total rows to {parquet_path}")

    return len(new_rows)


def fill_sentiment_gaps(
    start_date: date,
    end_date: date,
    parquet_path: Path,
    progress_callback: Callable[[GapFillProgress], None] | None = None,
    local_only: bool = False,
    shutdown_event: threading.Event | None = None,
) -> GapFillResult:
    """Fill missing sentiment data in the parquet file.

    Args:
        start_date: Earliest date to check for gaps
        end_date: Latest date to check for gaps
        parquet_path: Path to daily_sentiment.parquet
        progress_callback: Optional callback for progress updates
        local_only: If True, skip HuggingFace upload
        shutdown_event: If set, the function saves a checkpoint and stops early.

    Returns:
        GapFillResult with statistics, success status, and optional HF URL
    """
    progress = GapFillProgress()

    def update_progress():
        if progress_callback:
            progress_callback(progress)

    try:
        # Phase 1: Get universe symbols
        universe_type = get_etl_universe()
        logger.info("Phase 1: Getting %s symbols", universe_type.value)
        progress.current_phase = "getting_symbols"
        update_progress()

        if universe_type == UniverseType.HALAL:
            symbols = get_halal_symbols()
        elif universe_type == UniverseType.HALAL_NEW:
            symbols = get_halal_new_symbols()
        elif universe_type == UniverseType.HALAL_FILTERED:
            symbols = get_halal_filtered_symbols()
        elif universe_type == UniverseType.SP500:
            symbols = get_sp500_symbols()
        else:
            raise ValueError(f"Unknown universe type: {universe_type}")

        if not symbols:
            logger.error("No %s symbols found", universe_type.value)
            progress.error = f"No {universe_type.value} symbols found"
            progress.status = "failed"
            return GapFillResult(success=False, progress=progress)

        logger.info(
            "Found %d %s symbols: %s...",
            len(symbols),
            universe_type.value,
            symbols[:5],
        )

        # Phase 2: Detect gaps
        logger.info("Phase 2: Detecting gaps in parquet")
        progress.current_phase = "detecting_gaps"
        update_progress()

        all_gaps = find_gaps(symbols, start_date, end_date, parquet_path)
        progress.total_gaps = len(all_gaps)

        fillable_gaps, unfillable_gaps = categorize_gaps(all_gaps, ALPACA_EARLIEST_DATE)
        progress.gaps_fillable = len(fillable_gaps)
        progress.gaps_pre_api_date = len(unfillable_gaps)

        logger.info(
            f"Gap analysis: {len(all_gaps):,} total gaps, "
            f"{len(fillable_gaps):,} fillable (2015+), "
            f"{len(unfillable_gaps):,} unfillable (pre-2015)"
        )

        if not fillable_gaps:
            logger.info("No fillable gaps found (all gaps are pre-2015)")
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
        logger.info("Phase 3: Fetching news from Alpaca and scoring with FinBERT")
        progress.current_phase = "fetching_and_scoring"
        update_progress()

        alpaca_client = AlpacaNewsClient()
        scorer = FinBERTScorer()

        # Group gaps by date for efficient API calls
        date_to_symbols = _group_gaps_by_date(fillable_gaps)
        logger.info(f"Grouped gaps into {len(date_to_symbols)} unique dates")

        # Sort dates in reverse chronological order (most recent first)
        sorted_dates = sorted(date_to_symbols.keys(), reverse=True)
        if sorted_dates:
            logger.info(
                f"Processing dates from {sorted_dates[0]} to {sorted_dates[-1]}"
            )

        articles_with_scores: list[tuple[datetime, str, SentimentScore]] = []
        checked_gaps_no_articles: list[tuple[date, str]] = []
        today = date.today()
        last_checkpoint_calls = 0

        for gap_date in sorted_dates:
            if shutdown_event and shutdown_event.is_set():
                logger.warning("Shutdown requested, saving checkpoint and stopping")
                progress.current_phase = "shutdown_checkpoint"
                update_progress()
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
                # Record gaps we checked but found no articles (except today)
                if gap_date != today:
                    for symbol in gap_symbols:
                        checked_gaps_no_articles.append((gap_date, symbol))
            else:
                # Score articles with FinBERT
                texts = [
                    f"{a.headline} {a.summary}".strip() if a.summary else a.headline
                    for a in articles
                ]
                scores = scorer.score_batch(texts)
                progress.articles_scored += len(scores)

                # Track which gap symbols got matched by articles
                matched_symbols: set[str] = set()

                # Map articles to their symbols and dates
                for article, score in zip(articles, scores, strict=False):
                    # Each article may be relevant to multiple symbols
                    for symbol in article.symbols:
                        if symbol in gap_symbols:
                            articles_with_scores.append(
                                (article.created_at, symbol, score)
                            )
                            matched_symbols.add(symbol)

                # Record unmatched gap symbols as zero-article (except today)
                # This handles the case where Alpaca returns articles but none
                # match the gap symbols we're looking for
                if gap_date != today:
                    for symbol in gap_symbols:
                        if symbol not in matched_symbols:
                            checked_gaps_no_articles.append((gap_date, symbol))

                update_progress()

            # Checkpoint: save to parquet every CHECKPOINT_INTERVAL API calls
            # (runs regardless of whether articles were found)
            calls_since_checkpoint = progress.api_calls_made - last_checkpoint_calls
            if calls_since_checkpoint >= CHECKPOINT_INTERVAL:
                logger.info(
                    f"Checkpoint at {progress.api_calls_made} API calls - "
                    f"saving to parquet"
                )
                progress.current_phase = "checkpoint_saving"
                update_progress()

                # Aggregate and write accumulated data
                checkpoint_rows = _aggregate_daily_sentiment(articles_with_scores)
                checkpoint_zero_rows = _create_zero_article_rows(
                    checked_gaps_no_articles
                )
                all_checkpoint_rows = checkpoint_rows + checkpoint_zero_rows

                if all_checkpoint_rows:
                    rows_added = append_to_parquet(all_checkpoint_rows, parquet_path)
                    progress.rows_added += rows_added
                    logger.info(
                        f"Checkpoint saved {rows_added} rows "
                        f"(total: {progress.rows_added})"
                    )

                # Clear buffers (data is now in parquet)
                articles_with_scores.clear()
                checked_gaps_no_articles.clear()

                progress.checkpoints_saved += 1
                last_checkpoint_calls = progress.api_calls_made
                progress.current_phase = "fetching_and_scoring"
                update_progress()

        # Detect if we were interrupted by shutdown
        was_cancelled = shutdown_event is not None and shutdown_event.is_set()

        # Phase 4: Aggregate and write remaining data to parquet
        phase_label = "shutdown_checkpoint" if was_cancelled else "writing_parquet"
        logger.info(f"Phase 4: Writing remaining data to parquet ({phase_label})")
        progress.current_phase = phase_label
        update_progress()

        # Write any remaining data not yet checkpointed
        rows_added = 0
        if articles_with_scores or checked_gaps_no_articles:
            logger.info(
                f"Aggregating {len(articles_with_scores)} remaining "
                f"article-symbol-score entries"
            )
            new_rows = _aggregate_daily_sentiment(articles_with_scores)
            zero_rows = _create_zero_article_rows(checked_gaps_no_articles)
            all_new_rows = new_rows + zero_rows

            logger.info(
                f"Writing {len(new_rows)} rows with articles + "
                f"{len(zero_rows)} zero-article rows"
            )
            rows_added = append_to_parquet(all_new_rows, parquet_path)
            progress.rows_added += rows_added
        else:
            logger.info("No remaining data to write (all saved in checkpoints)")

        # Calculate remaining gaps by re-checking the parquet file
        updated_gaps = find_gaps(symbols, start_date, end_date, parquet_path)
        updated_fillable, _ = categorize_gaps(updated_gaps, ALPACA_EARLIEST_DATE)
        progress.remaining_gaps = len(updated_fillable)

        if was_cancelled:
            progress.status = "cancelled"
            progress.current_phase = "cancelled"
            logger.warning(
                f"Gap fill cancelled by shutdown: {progress.rows_added} rows saved, "
                f"{progress.remaining_gaps:,} gaps still remaining"
            )
        else:
            progress.status = "completed"
            progress.current_phase = "done"
            logger.info(
                f"Gap fill completed: {rows_added} rows added, "
                f"{progress.api_calls_made} API calls made, "
                f"{progress.remaining_gaps:,} gaps remaining"
            )
        update_progress()

        statistics = get_gap_statistics(
            symbols, start_date, end_date, parquet_path, ALPACA_EARLIEST_DATE
        )

        # Upload to HuggingFace if configured and not local_only
        hf_url = None
        if not local_only and rows_added > 0:
            hf_url = _upload_to_huggingface(parquet_path)

        return GapFillResult(
            success=True,
            progress=progress,
            statistics=statistics,
            parquet_updated=rows_added > 0,
            hf_url=hf_url,
        )

    except Exception as e:
        logger.exception("Gap fill failed")
        progress.status = "failed"
        progress.error = str(e)
        update_progress()
        return GapFillResult(success=False, progress=progress)


def _upload_to_huggingface(parquet_path: Path) -> str | None:
    """Upload parquet file to HuggingFace.

    Args:
        parquet_path: Path to the parquet file

    Returns:
        URL of the uploaded file, or None if upload failed or not configured
    """
    from brain_api.etl.config import get_hf_news_sentiment_repo

    hf_repo = get_hf_news_sentiment_repo()
    if not hf_repo:
        logger.info("HuggingFace upload skipped (HF_NEWS_SENTIMENT_REPO not set)")
        return None

    try:
        from huggingface_hub import HfApi

        logger.info(f"Uploading to HuggingFace: {hf_repo}")
        api = HfApi()

        # Create repo if it doesn't exist
        try:
            api.repo_info(repo_id=hf_repo, repo_type="dataset")
        except Exception:
            logger.info(f"Creating HuggingFace repository: {hf_repo}")
            api.create_repo(repo_id=hf_repo, repo_type="dataset", exist_ok=True)

        # Upload parquet file
        api.upload_file(
            path_or_fileobj=str(parquet_path),
            path_in_repo="daily_sentiment.parquet",
            repo_id=hf_repo,
            repo_type="dataset",
        )

        hf_url = f"https://huggingface.co/datasets/{hf_repo}"
        logger.info(f"Uploaded to {hf_url}")
        return hf_url

    except Exception as e:
        logger.error(f"HuggingFace upload failed: {e}")
        return None
