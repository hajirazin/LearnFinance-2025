"""Main entrypoint for news sentiment ETL pipeline.

Can be run as:
1. CLI: python -m news_sentiment_etl.main [options]
2. Cloud Function: deploy with handler() as entrypoint
"""

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from news_sentiment_etl.core.cache import SentimentCache, compute_article_hash
from news_sentiment_etl.core.config import ETLConfig, get_hf_news_sentiment_repo
from news_sentiment_etl.core.dataset import batch_articles, stream_articles
from news_sentiment_etl.core.sentiment import FinBERTScorer
from news_sentiment_etl.core.symbol_filter import UniverseFilter
from news_sentiment_etl.output.parquet_writer import ParquetWriter, read_parquet_stats

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


def run_pipeline(config: ETLConfig) -> dict:
    """Run the full ETL pipeline.

    Args:
        config: ETL configuration

    Returns:
        Dict with pipeline results and statistics
    """
    start_time = time.time()
    stats = {
        "started_at": datetime.now(UTC).isoformat(),
        "config": {
            "dataset": config.dataset_name,
            "batch_size": config.batch_size,
            "sentiment_threshold": config.sentiment_threshold,
            "filter_to_halal": config.filter_to_halal,
            "max_articles": config.max_articles,
        },
    }

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
    if config.filter_to_halal:
        console.print("  Fetching halal universe from ETFs...")
        universe = UniverseFilter.from_halal_universe()
        console.print(f"  Found [green]{universe.symbol_count}[/] halal symbols")
        stats["universe"] = {
            "type": "halal",
            "symbol_count": universe.symbol_count,
            "fetched_at": universe.fetched_at,
        }
    else:
        universe = UniverseFilter.allow_all()
        stats["universe"] = {"type": "all"}

    # Sentiment cache
    cache = SentimentCache(config.cache_dir)
    cache.log_status()
    initial_cache_size = cache.stats.total_entries

    # FinBERT scorer (lazy-loaded on first use, with cache)
    console.print("  FinBERT model will load on first use")
    scorer = FinBERTScorer(use_gpu=config.use_gpu, cache=cache)
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
        console.print(f"  Target: [yellow]{config.max_articles:,}[/] NEW articles to score")
    console.print()

    # Stream and process articles with DuckDB pre-filtering
    try:
        # Pass halal symbols to DuckDB for SQL-level filtering
        halal_symbols = universe.symbols if config.filter_to_halal else None
        
        # Get cached hashes to skip already-processed articles
        cached_hashes = cache.get_all_cached_hashes()
        
        article_stream = stream_articles(
            config, 
            halal_symbols=halal_symbols,
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
                    console.print(f"\n  [yellow]Loading FinBERT model (first halal match found)...[/]")
                    # Trigger model load by accessing device
                    console.print(f"  Using device: [green]{scorer.device}[/]")
                    stats["device"] = scorer.device
                    model_loaded = True
                    console.print()

                # Score batch (uses cache automatically)
                texts = [a.text for a in filtered_batch]
                scores, cache_hits, new_scores = scorer.score_batch(texts)

                total_cache_hits += cache_hits
                total_new_scores += new_scores

                # Store article-symbol associations ONLY if we scored new articles
                if new_scores > 0:
                    for article in filtered_batch:
                        article_hash = compute_article_hash(article.text)
                        cache.store_article_symbols(article_hash, article.date, article.symbols)

                batches_processed += 1
                
                # Calculate throughput
                elapsed = time.time() - start_time
                articles_per_sec = total_articles / elapsed if elapsed > 0 else 0
                
                # Update progress bar with new scores
                pbar.update(new_scores)
                pbar.set_postfix({
                    "cached": total_cache_hits,
                    "scanned": format_number(total_articles),
                    "rate": f"{articles_per_sec:.0f}/s",
                })

                # Check if we've scored enough NEW articles
                if config.max_articles and total_new_scores >= config.max_articles:
                    console.print(f"\n  [green]✓ Reached {config.max_articles:,} new articles scored[/]")
                    break

                # Detailed stats every N batches
                if batches_processed % DETAILED_STATS_INTERVAL == 0:
                    now = time.time()
                    interval_articles = total_articles - last_stats_articles
                    interval_time = now - last_stats_time
                    interval_rate = interval_articles / interval_time if interval_time > 0 else 0

                    # Estimate remaining time (based on new scores if we have a target)
                    if config.max_articles and total_new_scores > 0:
                        new_per_sec = total_new_scores / elapsed if elapsed > 0 else 0
                        remaining_new = config.max_articles - total_new_scores
                        eta_seconds = remaining_new / new_per_sec if new_per_sec > 0 else 0
                    else:
                        remaining_articles = ESTIMATED_TOTAL_ARTICLES - total_articles
                        eta_seconds = remaining_articles / articles_per_sec if articles_per_sec > 0 else 0

                    # Cache hit rate
                    total_scored = total_cache_hits + total_new_scores
                    cache_hit_rate = (total_cache_hits / total_scored * 100) if total_scored > 0 else 0
                    progress_pct = (total_articles / ESTIMATED_TOTAL_ARTICLES) * 100

                    console.print(f"\n  [dim]─── Batch {batches_processed} Stats ───[/]")
                    console.print(f"  Scanned: [cyan]{format_number(total_articles)}[/] articles ({progress_pct:.2f}% of dataset)")
                    console.print(f"  Elapsed: [cyan]{format_duration(elapsed)}[/] | ETA: [yellow]{format_duration(eta_seconds)}[/]")
                    console.print(f"  Rate: [cyan]{articles_per_sec:.0f}[/] articles/sec (last interval: {interval_rate:.0f}/sec)")
                    console.print(f"  Matched halal: [green]{articles_after_filter:,}[/] ({100*articles_after_filter/articles_with_symbols:.1f}%)")
                    console.print(f"  Cache: [green]{total_cache_hits:,}[/] hits / [yellow]{total_new_scores:,}[/] new ({cache_hit_rate:.1f}% hit rate)")
                    console.print(f"  Symbols found: [green]{len(symbols_seen)}[/] | Date range: {min(dates_seen) if dates_seen else 'N/A'} → {max(dates_seen) if dates_seen else 'N/A'}")
                    console.print(f"  Article-symbol pairs in DB: [yellow]{cache.article_symbols_count:,}[/]")
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
                    console.print(f"  [dim]💾 Checkpoint saved (batch {batches_processed})[/]")

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Interrupted! Saving progress...[/]")

    # Aggregate via SQL and write results
    console.print("\n[bold blue]Aggregating sentiments (SQL)...[/]")
    console.print(f"  Article-symbol pairs in DB: [cyan]{cache.article_symbols_count:,}[/]")
    console.print(f"  Sentiment threshold: [cyan]{config.sentiment_threshold}[/]")
    daily_sentiments = cache.aggregate_daily_sentiment(config.sentiment_threshold)
    console.print(f"  Generated [green]{len(daily_sentiments)}[/] daily sentiment records")

    # Write output file(s)
    console.print("\n[bold blue]Writing output...[/]")
    writer.write(daily_sentiments)
    output_paths = writer.finalize(
        parquet_filename=config.output_parquet,
        csv_filename=config.output_csv,
    )

    for fmt, path in output_paths.items():
        console.print(f"  {fmt.upper()}: [green]{path}[/]")

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
                api.create_repo(repo_id=hf_news_repo, repo_type="dataset", exist_ok=True)

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
            console.print("  [dim]Local files are still saved. Run manually with push script if needed.[/]")

    elif config.local_only:
        console.print("\n[dim]HuggingFace upload skipped (--local-only)[/]")
    elif not hf_news_repo:
        console.print("\n[dim]HuggingFace upload skipped (HF_NEWS_SENTIMENT_REPO not set)[/]")

    # Get output stats (from parquet if available, else from csv)
    if "parquet" in output_paths:
        output_stats = read_parquet_stats(output_paths["parquet"])
    else:
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

    # Close cache and get final stats
    final_cache_size = cache.stats.total_entries
    new_cache_entries = final_cache_size - initial_cache_size
    total_scored = total_cache_hits + total_new_scores
    cache_hit_rate = (total_cache_hits / total_scored * 100) if total_scored > 0 else 0
    cache.close()

    # Finalize stats
    elapsed = time.time() - start_time
    output_info = {
        "paths": {fmt: str(path) for fmt, path in output_paths.items()},
        "hf_url": hf_upload_url,
        **output_stats,
    }
    # Set device to N/A if model was never loaded
    if "device" not in stats:
        stats["device"] = "N/A (model not needed)"
    stats.update({
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
    })

    # Clean up checkpoint
    if checkpoint_file.exists():
        checkpoint_file.unlink()

    return stats


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Process HuggingFace financial news into daily sentiment scores"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/output"),
        help="Output directory for Parquet files",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for FinBERT processing",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Sentiment threshold for bounded filtering",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=None,
        help="Maximum articles to process (for testing)",
    )
    parser.add_argument(
        "--no-halal-filter",
        action="store_true",
        help="Disable filtering to halal universe",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU usage (disable GPU)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token for gated datasets",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Skip HuggingFace upload even if HF_NEWS_SENTIMENT_REPO is set",
    )
    parser.add_argument(
        "--parquet",
        type=str,
        default=None,
        metavar="FILENAME",
        help="Output Parquet filename (e.g., 'output.parquet'). Default: daily_sentiment.parquet",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        metavar="FILENAME",
        help="Output CSV filename (e.g., 'output.csv'). If specified, CSV will be generated.",
    )

    args = parser.parse_args()

    # Determine output formats:
    # - If neither --parquet nor --csv specified: default to parquet only
    # - If --parquet specified: output parquet with that name
    # - If --csv specified: output csv with that name
    # - If both specified: output both
    if args.parquet is None and args.csv is None:
        output_parquet = "daily_sentiment.parquet"
        output_csv = None
    else:
        output_parquet = args.parquet
        output_csv = args.csv

    # Build config
    config = ETLConfig(
        output_dir=args.output_dir,
        output_parquet=output_parquet,
        output_csv=output_csv,
        batch_size=args.batch_size,
        sentiment_threshold=args.threshold,
        max_articles=args.max_articles,
        filter_to_halal=not args.no_halal_filter,
        use_gpu=False if args.cpu else None,
        hf_token=args.hf_token,
        local_only=args.local_only,
    )

    console.print()
    console.print("[bold cyan]╔══════════════════════════════════════════╗[/]")
    console.print("[bold cyan]║     News Sentiment ETL Pipeline          ║[/]")
    console.print("[bold cyan]╚══════════════════════════════════════════╝[/]")
    console.print()

    try:
        stats = run_pipeline(config)

        console.print("\n[bold green]✅ Pipeline completed![/]")
        console.print()

        # Create summary table
        table = Table(title="Pipeline Summary", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="dim")
        table.add_column("Value", justify="right")

        table.add_row("Articles Processed", f"{stats['articles']['total_processed']:,}")
        table.add_row("Articles with Halal Symbols", f"{stats['articles']['after_universe_filter']:,}")
        table.add_row("Cache Hits", f"{stats['cache']['hits']:,}")
        table.add_row("New Scores", f"{stats['cache']['new_scores']:,}")
        table.add_row("Cache Hit Rate", f"{stats['cache']['hit_rate_percent']:.1f}%")
        table.add_row("Total Cached", f"{stats['cache']['total_cached']:,}")
        table.add_row("Output Rows", f"{stats['output']['row_count']:,}")
        table.add_row("Unique Symbols", str(stats['output']['symbol_count']))
        table.add_row("Date Range", f"{stats['output']['date_min']} → {stats['output']['date_max']}")
        table.add_row("File Size", f"{stats['output']['file_size_mb']} MB")
        table.add_row("Total Time", format_duration(stats['elapsed_seconds']))
        table.add_row("Avg Rate", f"{stats['articles']['total_processed'] / stats['elapsed_seconds']:.0f} articles/sec")
        table.add_row("Device Used", stats['device'])

        console.print(table)

        # Save stats
        stats_path = config.output_dir / "pipeline_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        console.print(f"\n📊 Stats saved to: [green]{stats_path}[/]")
        for fmt, path in stats['output']['paths'].items():
            console.print(f"📁 {fmt.upper()}: [green]{path}[/]")

        return 0

    except Exception as e:
        console.print(f"\n[bold red]Error:[/] {e}")
        import traceback
        traceback.print_exc()
        return 1


# Cloud Function handler
def handler(request):
    """Google Cloud Function entrypoint.

    Args:
        request: Flask request object

    Returns:
        JSON response with pipeline results
    """
    # Parse request
    request_json = request.get_json(silent=True) or {}

    config = ETLConfig(
        output_dir=Path(request_json.get("output_dir", "/tmp/output")),
        output_parquet=request_json.get("output_parquet", "daily_sentiment.parquet"),
        output_csv=request_json.get("output_csv"),
        batch_size=request_json.get("batch_size", 256),
        sentiment_threshold=request_json.get("threshold", 0.1),
        max_articles=request_json.get("max_articles"),
        filter_to_halal=request_json.get("filter_to_halal", True),
        use_gpu=request_json.get("use_gpu"),
        hf_token=request_json.get("hf_token"),
        local_only=request_json.get("local_only", False),
    )

    try:
        stats = run_pipeline(config)
        return {"status": "success", **stats}
    except Exception as e:
        return {"status": "error", "error": str(e)}, 500


if __name__ == "__main__":
    sys.exit(main())

