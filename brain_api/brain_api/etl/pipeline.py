"""Main ETL pipeline for news sentiment processing.

Moved and adapted from news_sentiment_etl/main.py.
This module provides the run_pipeline function that can be called
from the API endpoint or CLI.
"""

import json
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from rich.console import Console
from tqdm import tqdm

from brain_api.core.finbert import FinBERTScorer, compute_text_hash
from brain_api.core.sentiment_cache import SentimentCache
from brain_api.etl.config import ETLConfig, get_hf_news_sentiment_repo
from brain_api.etl.dataset import batch_articles, stream_articles
from brain_api.etl.gap_fill import append_to_parquet
from brain_api.etl.parquet_writer import ParquetWriter, read_parquet_stats
from brain_api.etl.symbol_filter import UniverseFilter

console = Console()

# Estimated total articles in dataset (for progress %)
ESTIMATED_TOTAL_ARTICLES = 57_000_000
# How often to print detailed stats
DETAILED_STATS_INTERVAL = 50  # Every N batches


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def format_number(n: int) -> str:
    """Format large numbers with K/M suffix."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def run_pipeline(
    config: ETLConfig,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    shutdown_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Run the full ETL pipeline.

    Args:
        config: ETL configuration
        progress_callback: Optional callback for progress updates (for API jobs)
        shutdown_event: If set, the pipeline saves progress and stops early.

    Returns:
        Dict with pipeline results and statistics
    """
    start_time = time.time()
    stats: dict[str, Any] = {
        "started_at": datetime.now(UTC).isoformat(),
        "config": {
            "dataset": config.dataset_name,
            "batch_size": config.batch_size,
            "sentiment_threshold": config.sentiment_threshold,
            "universe": config.universe.value,
            "max_articles": config.max_articles,
        },
    }

    def update_progress(progress: dict[str, Any]) -> None:
        """Update progress via callback if provided."""
        if progress_callback:
            progress_callback(progress)

    # Initialize components
    console.print("[bold blue]🚀 Initializing pipeline...[/]")

    # Show configuration
    console.print(f"  Dataset: [cyan]{config.dataset_name}[/]")
    console.print(f"  Output: [cyan]{config.output_dir}[/]")
    console.print(f"  Batch size: [cyan]{config.batch_size}[/]")
    console.print(f"  Sentiment threshold: [cyan]{config.sentiment_threshold}[/]")
    if config.max_articles:
        console.print(f"  Max articles: [yellow]{config.max_articles:,}[/] (limited)")
    console.print()

    # Universe filter
    universe_name = config.universe.value
    console.print(f"  Fetching [cyan]{universe_name}[/] universe...")
    universe = UniverseFilter.from_universe_type(config.universe)
    console.print(f"  Found [green]{universe.symbol_count}[/] {universe_name} symbols")
    stats["universe"] = {
        "type": universe_name,
        "symbol_count": universe.symbol_count,
        "fetched_at": universe.fetched_at,
    }

    # Sentiment cache
    cache = SentimentCache(config.cache_dir)
    cache.log_status()
    initial_cache_size = cache.stats.total_entries

    # FinBERT scorer (lazy-loaded on first use, with cache)
    console.print("  FinBERT model will load on first use")
    # Reset singleton to ensure fresh instance with cache
    FinBERTScorer.reset()
    scorer = FinBERTScorer(cache=cache, use_gpu=config.use_gpu)
    model_loaded = False

    # Writer (aggregation now happens via SQL in cache)
    writer = ParquetWriter(config.output_dir)

    # Checkpoint setup
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = config.checkpoint_dir / "progress.json"

    # Stats tracking
    total_articles = 0
    articles_with_symbols = 0
    articles_after_filter = 0
    batches_processed = 0
    symbols_seen: set[str] = set()
    dates_seen: set[str] = set()
    last_stats_time = time.time()
    last_stats_articles = 0

    # Cache tracking
    total_cache_hits = 0
    total_new_scores = 0

    console.print("\n[bold blue]Processing articles...[/]")
    console.print(f"  Batch size: {config.batch_size}")
    if config.max_articles:
        console.print(
            f"  Target: [yellow]{config.max_articles:,}[/] NEW articles to score"
        )
    console.print()

    # Stream and process articles with DuckDB pre-filtering
    try:
        # Pass universe symbols to DuckDB for SQL-level filtering
        universe_symbols = universe.symbols

        # Get cached hashes to skip already-processed articles
        cached_hashes = cache.get_all_cached_hashes()

        article_stream = stream_articles(
            config,
            halal_symbols=universe_symbols,
            cached_hashes=cached_hashes,
        )

        # Use tqdm for progress - tracks NEW scores toward target
        pbar_total = config.max_articles if config.max_articles else None
        with tqdm(
            desc="New scores",
            unit=" new",
            dynamic_ncols=True,
            total=pbar_total,
            smoothing=0.1,
        ) as pbar:
            for batch in batch_articles(article_stream, config.batch_size):
                if shutdown_event and shutdown_event.is_set():
                    console.print("\n[yellow]Shutdown requested, saving progress...[/]")
                    break

                # DuckDB already filtered to halal symbols, just track stats
                for article in batch:
                    articles_with_symbols += 1
                    articles_after_filter += 1
                    symbols_seen.update(article.symbols)
                    dates_seen.add(article.date)

                total_articles += len(batch)

                if not batch:
                    continue

                # Use batch directly (already filtered by DuckDB)
                filtered_batch = batch

                # Load FinBERT on first actual use (if there are uncached articles)
                if not model_loaded:
                    console.print(
                        "\n  [yellow]Loading FinBERT model"
                        " (first universe match found)...[/]"
                    )
                    # Trigger model load by accessing device
                    console.print(f"  Using device: [green]{scorer.device}[/]")
                    stats["device"] = scorer.device
                    model_loaded = True
                    console.print()

                # Score batch (uses cache automatically)
                texts = [a.text for a in filtered_batch]
                _scores, cache_hits, new_scores = scorer.score_batch_with_stats(texts)

                total_cache_hits += cache_hits
                total_new_scores += new_scores

                # Store article-symbol associations ONLY if we scored new articles
                if new_scores > 0:
                    for article in filtered_batch:
                        article_hash = compute_text_hash(article.text)
                        cache.store_article_symbols(
                            article_hash, article.date, article.symbols
                        )

                batches_processed += 1

                # Calculate throughput
                elapsed = time.time() - start_time
                articles_per_sec = total_articles / elapsed if elapsed > 0 else 0

                # Update progress bar with new scores
                pbar.update(new_scores)
                pbar.set_postfix(
                    {
                        "cached": total_cache_hits,
                        "scanned": format_number(total_articles),
                        "rate": f"{articles_per_sec:.0f}/s",
                    }
                )

                # Update progress callback
                update_progress(
                    {
                        "status": "running",
                        "total_articles": total_articles,
                        "new_scores": total_new_scores,
                        "cache_hits": total_cache_hits,
                        "batches_processed": batches_processed,
                        "elapsed_seconds": elapsed,
                    }
                )

                # Check if we've scored enough NEW articles
                if config.max_articles and total_new_scores >= config.max_articles:
                    console.print(
                        f"\n  [green]✓ Reached {config.max_articles:,}"
                        " new articles scored[/]"
                    )
                    break

                # Detailed stats every N batches
                if batches_processed % DETAILED_STATS_INTERVAL == 0:
                    now = time.time()
                    interval_articles = total_articles - last_stats_articles
                    interval_time = now - last_stats_time
                    interval_rate = (
                        interval_articles / interval_time if interval_time > 0 else 0
                    )

                    # Estimate remaining time (based on new scores if we have a target)
                    if config.max_articles and total_new_scores > 0:
                        new_per_sec = total_new_scores / elapsed if elapsed > 0 else 0
                        remaining_new = config.max_articles - total_new_scores
                        eta_seconds = (
                            remaining_new / new_per_sec if new_per_sec > 0 else 0
                        )
                    else:
                        remaining_articles = ESTIMATED_TOTAL_ARTICLES - total_articles
                        eta_seconds = (
                            remaining_articles / articles_per_sec
                            if articles_per_sec > 0
                            else 0
                        )

                    # Cache hit rate
                    total_scored = total_cache_hits + total_new_scores
                    cache_hit_rate = (
                        (total_cache_hits / total_scored * 100)
                        if total_scored > 0
                        else 0
                    )
                    progress_pct = (total_articles / ESTIMATED_TOTAL_ARTICLES) * 100

                    console.print(
                        f"\n  [dim]─── Batch {batches_processed} Stats ───[/]"
                    )
                    console.print(
                        f"  Scanned: [cyan]{format_number(total_articles)}[/]"
                        f" articles ({progress_pct:.2f}% of dataset)"
                    )
                    console.print(
                        f"  Elapsed: [cyan]{format_duration(elapsed)}[/]"
                        f" | ETA: [yellow]{format_duration(eta_seconds)}[/]"
                    )
                    console.print(
                        f"  Rate: [cyan]{articles_per_sec:.0f}[/] articles/sec"
                        f" (last interval: {interval_rate:.0f}/sec)"
                    )
                    console.print(
                        f"  Matched {universe_name}: [green]{articles_after_filter:,}[/]"
                        f" ({100 * articles_after_filter / articles_with_symbols:.1f}%)"
                    )
                    console.print(
                        f"  Cache: [green]{total_cache_hits:,}[/] hits"
                        f" / [yellow]{total_new_scores:,}[/] new"
                        f" ({cache_hit_rate:.1f}% hit rate)"
                    )
                    console.print(
                        f"  Symbols found: [green]{len(symbols_seen)}[/]"
                        f" | Date range: {min(dates_seen) if dates_seen else 'N/A'}"
                        f" → {max(dates_seen) if dates_seen else 'N/A'}"
                    )
                    console.print(
                        f"  Article-symbol pairs in DB:"
                        f" [yellow]{cache.article_symbols_count:,}[/]"
                    )
                    console.print()

                    last_stats_time = now
                    last_stats_articles = total_articles

                # Checkpoint
                if batches_processed % config.checkpoint_interval == 0:
                    checkpoint = {
                        "batches_processed": batches_processed,
                        "articles_processed": total_articles,
                        "symbols_found": len(symbols_seen),
                        "date_range": {
                            "min": min(dates_seen) if dates_seen else None,
                            "max": max(dates_seen) if dates_seen else None,
                        },
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
                    with open(checkpoint_file, "w") as f:
                        json.dump(checkpoint, f)
                    console.print(
                        f"  [dim]💾 Checkpoint saved (batch {batches_processed})[/]"
                    )

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Interrupted! Saving progress...[/]")

    # Aggregate via SQL and write results
    console.print("\n[bold blue]Aggregating sentiments (SQL)...[/]")
    console.print(
        f"  Article-symbol pairs in DB: [cyan]{cache.article_symbols_count:,}[/]"
    )
    console.print(f"  Sentiment threshold: [cyan]{config.sentiment_threshold}[/]")
    daily_sentiments = cache.aggregate_daily_sentiment(config.sentiment_threshold)
    console.print(
        f"  Generated [green]{len(daily_sentiments)}[/] daily sentiment records"
    )

    # Write output file(s) - INCREMENTAL: merge with existing instead of overwriting
    console.print("\n[bold blue]Writing output (incremental)...[/]")
    parquet_path = config.output_dir / config.output_parquet
    output_paths: dict[str, Any] = {}

    # Convert daily sentiments to row dicts for append_to_parquet
    new_rows = [s.to_dict() for s in daily_sentiments]

    # Use incremental append (merge + deduplicate) instead of overwrite
    rows_written = append_to_parquet(new_rows, parquet_path)
    output_paths["parquet"] = parquet_path
    console.print(f"  PARQUET: [green]{parquet_path}[/] ({rows_written} new/updated)")

    # Optionally write CSV (still overwrites since it's for debugging)
    if config.output_csv:
        csv_path = config.output_dir / config.output_csv
        writer.write(daily_sentiments)
        csv_output = writer.finalize(
            parquet_filename=None,  # Skip parquet, already written
            csv_filename=config.output_csv,
        )
        if "csv" in csv_output:
            output_paths["csv"] = csv_output["csv"]
            console.print(f"  CSV: [green]{csv_output['csv']}[/]")

    # Upload to HuggingFace if configured and not local_only
    hf_upload_url = None
    hf_news_repo = get_hf_news_sentiment_repo()

    if not config.local_only and hf_news_repo:
        console.print("\n[bold blue]Uploading to HuggingFace...[/]")
        console.print(f"  Repository: [cyan]{hf_news_repo}[/]")

        try:
            from huggingface_hub import HfApi

            api = HfApi()

            # Create repo if it doesn't exist
            try:
                api.repo_info(repo_id=hf_news_repo, repo_type="dataset")
            except Exception:
                console.print(f"  Creating repository: [yellow]{hf_news_repo}[/]")
                api.create_repo(
                    repo_id=hf_news_repo, repo_type="dataset", exist_ok=True
                )

            # Upload parquet file if it exists
            if "parquet" in output_paths:
                parquet_path = output_paths["parquet"]
                api.upload_file(
                    path_or_fileobj=str(parquet_path),
                    path_in_repo="data/daily_sentiment.parquet",
                    repo_id=hf_news_repo,
                    repo_type="dataset",
                    commit_message=f"Update sentiment data: {len(daily_sentiments)} records",
                )
                hf_upload_url = f"https://huggingface.co/datasets/{hf_news_repo}"
                console.print(f"  [green]✓ Uploaded to {hf_upload_url}[/]")

        except Exception as e:
            console.print(f"  [red]✗ HuggingFace upload failed: {e}[/]")
            console.print(
                "  [dim]Local files are still saved."
                " Run manually with push script if needed.[/]"
            )

    elif config.local_only:
        console.print("\n[dim]HuggingFace upload skipped (--local-only)[/]")
    elif not hf_news_repo:
        console.print(
            "\n[dim]HuggingFace upload skipped (HF_NEWS_SENTIMENT_REPO not set)[/]"
        )

    # Get output stats (from parquet if available, else from csv)
    if "parquet" in output_paths:
        output_stats = read_parquet_stats(output_paths["parquet"])
    elif "csv" in output_paths:
        # Read stats from CSV
        import pandas as pd

        csv_path = output_paths["csv"]
        df = pd.read_csv(csv_path)
        output_stats = {
            "row_count": len(df),
            "date_min": str(df["date"].min()) if len(df) > 0 else None,
            "date_max": str(df["date"].max()) if len(df) > 0 else None,
            "symbol_count": df["symbol"].nunique() if len(df) > 0 else 0,
            "file_size_mb": round(csv_path.stat().st_size / (1024 * 1024), 2),
        }
    else:
        output_stats = {"row_count": 0}

    # Close cache and get final stats
    final_cache_size = cache.stats.total_entries
    new_cache_entries = final_cache_size - initial_cache_size
    total_scored = total_cache_hits + total_new_scores
    cache_hit_rate = (total_cache_hits / total_scored * 100) if total_scored > 0 else 0
    cache.close()

    # Finalize stats
    elapsed = time.time() - start_time
    output_info: dict[str, Any] = {
        "paths": {fmt: str(path) for fmt, path in output_paths.items()},
        "hf_url": hf_upload_url,
        **output_stats,
    }
    # Set device to N/A if model was never loaded
    if "device" not in stats:
        stats["device"] = "N/A (model not needed)"
    stats.update(
        {
            "completed_at": datetime.now(UTC).isoformat(),
            "elapsed_seconds": round(elapsed, 2),
            "articles": {
                "total_processed": total_articles,
                "with_symbols": articles_with_symbols,
                "after_universe_filter": articles_after_filter,
            },
            "cache": {
                "hits": total_cache_hits,
                "new_scores": total_new_scores,
                "hit_rate_percent": round(cache_hit_rate, 2),
                "total_cached": final_cache_size,
                "new_entries": new_cache_entries,
            },
            "batches_processed": batches_processed,
            "output": output_info,
        }
    )

    # Clean up checkpoint
    if checkpoint_file.exists():
        checkpoint_file.unlink()

    # Final progress update
    update_progress(
        {
            "status": "completed",
            "total_articles": total_articles,
            "new_scores": total_new_scores,
            "cache_hits": total_cache_hits,
            "batches_processed": batches_processed,
            "elapsed_seconds": elapsed,
            "output": output_info,
        }
    )

    return stats
