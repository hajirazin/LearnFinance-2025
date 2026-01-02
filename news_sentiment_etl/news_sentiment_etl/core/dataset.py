"""HuggingFace dataset streaming loader with symbol extraction.

Uses DuckDB for fast pre-filtering of parquet files.
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import duckdb
from huggingface_hub import snapshot_download
from rich.console import Console

from news_sentiment_etl.core.config import ETLConfig

console = Console()


def ensure_local_dataset(config: ETLConfig) -> Path:
    """Download dataset to local directory. HuggingFace handles caching/delta/resume.

    On first run: downloads full dataset with progress bars.
    On subsequent runs: instantly skips if files match (ETag check).
    If source changed: downloads only delta.
    If interrupted: auto-resumes from where it stopped.

    Args:
        config: ETL configuration

    Returns:
        Path to the local dataset directory
    """
    local_dir = config.data_input_dir / "financial-news-multisource"

    console.print("[bold blue]📥 Checking dataset...[/]")
    console.print(f"  Repository: [cyan]{config.dataset_name}[/]")
    console.print(f"  Local path: [cyan]{local_dir}[/]")
    console.print()

    # HuggingFace shows its own progress bars during download
    # If files exist and match remote, it skips instantly
    local_path = snapshot_download(
        repo_id=config.dataset_name,
        repo_type="dataset",
        local_dir=local_dir,
        token=config.hf_token,
    )

    console.print("[bold green]✓ Dataset ready![/]")
    console.print()

    return Path(local_path)


@dataclass
class NewsArticle:
    """A single news article from the dataset.

    Attributes:
        date: Publication date (trading date from extra_fields if available)
        text: Article text (title + body)
        symbols: List of stock symbols mentioned
        dataset_source: Which subset this came from
        raw_date: Original date string from dataset
    """

    date: str  # YYYY-MM-DD format
    text: str
    symbols: list[str]
    dataset_source: str
    raw_date: str


def _parse_extra_fields(extra_fields_str: str | None) -> dict:
    """Parse the extra_fields JSON string.

    Args:
        extra_fields_str: JSON string from dataset

    Returns:
        Parsed dictionary or empty dict on error
    """
    if not extra_fields_str:
        return {}
    try:
        return json.loads(extra_fields_str)
    except (json.JSONDecodeError, TypeError):
        return {}


def _extract_symbols(extra_fields: dict) -> list[str]:
    """Extract stock symbols from extra_fields.

    The dataset stores symbols in various fields:
    - stocks: list of symbols
    - tickers: alternative field name
    - symbol: single symbol

    Args:
        extra_fields: Parsed extra_fields dict

    Returns:
        List of uppercase stock symbols
    """
    symbols = []

    # Try different field names
    if "stocks" in extra_fields:
        stocks = extra_fields["stocks"]
        if isinstance(stocks, list):
            symbols.extend(stocks)
        elif isinstance(stocks, str):
            symbols.append(stocks)

    if "tickers" in extra_fields:
        tickers = extra_fields["tickers"]
        if isinstance(tickers, list):
            symbols.extend(tickers)
        elif isinstance(tickers, str):
            symbols.append(tickers)

    if "symbol" in extra_fields:
        symbol = extra_fields["symbol"]
        if isinstance(symbol, str):
            symbols.append(symbol)

    # Normalize: uppercase, remove duplicates, filter empty
    normalized = []
    seen = set()
    for s in symbols:
        if s and isinstance(s, str):
            upper = s.upper().strip()
            if upper and upper not in seen:
                seen.add(upper)
                normalized.append(upper)

    return normalized


def _extract_date(row: dict, extra_fields: dict) -> str:
    """Extract the trading date from a row.

    Prefers date_trading from extra_fields (backtest-safe),
    falls back to the top-level date field.

    Args:
        row: Dataset row
        extra_fields: Parsed extra_fields dict

    Returns:
        Date string in YYYY-MM-DD format
    """
    # Prefer trading date for backtest safety
    trading_date = extra_fields.get("date_trading")
    if trading_date:
        # Format: "2011-04-20T13:30:00Z" -> "2011-04-20"
        try:
            dt = datetime.fromisoformat(trading_date.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d")
        except (ValueError, AttributeError):
            pass

    # Fall back to top-level date
    date_str = row.get("date", "")
    if date_str:
        try:
            # Format: "2011-04-19T00:00:00Z" -> "2011-04-19"
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d")
        except (ValueError, AttributeError):
            pass

    return ""


class DuckDBArticleStream:
    """DuckDB-based article streamer with eager initialization.
    
    Initializes DuckDB query and prints status BEFORE iteration starts,
    avoiding output buffering issues with tqdm.
    """
    
    def __init__(
        self,
        config: ETLConfig,
        halal_symbols: set[str] | None = None,
        cached_hashes: set[str] | None = None,
    ):
        self.config = config
        self.halal_symbols = halal_symbols
        self.cached_hashes = cached_hashes
        self.con = None
        self.result = None
        self.total_matching = 0
        
        # Initialize eagerly (not lazily in generator)
        self._initialize()
    
    def _initialize(self):
        """Initialize DuckDB connection and run COUNT query."""
        # Ensure dataset is downloaded locally (HF handles caching/resume)
        local_path = ensure_local_dataset(self.config)

        console.print("[bold blue]🦆 Loading dataset with DuckDB...[/]")

        # Build the parquet glob pattern
        parquet_pattern = str(local_path / "data" / "*" / "*.parquet")
        console.print(f"  Source: [cyan]{parquet_pattern}[/]")

        # Build SQL WHERE clause for halal symbols
        if self.halal_symbols:
            symbol_conditions = " OR ".join(
                f"extra_fields LIKE '%\"{sym}\"%'" for sym in self.halal_symbols
            )
            where_clause = f"""
                WHERE extra_fields IS NOT NULL
                AND LENGTH(text) >= 10
                AND ({symbol_conditions})
            """
            console.print(f"  Filtering to [green]{len(self.halal_symbols)}[/] halal symbols in SQL")
        else:
            where_clause = """
                WHERE extra_fields IS NOT NULL
                AND LENGTH(text) >= 10
                AND (
                    extra_fields LIKE '%"stocks"%' OR
                    extra_fields LIKE '%"tickers"%' OR
                    extra_fields LIKE '%"symbol"%'
                )
            """
            console.print("  Filtering to rows with symbols")

        # Connect and count
        self.con = duckdb.connect()
        count_query = f"SELECT COUNT(*) FROM '{parquet_pattern}' {where_clause}"
        self.total_matching = self.con.execute(count_query).fetchone()[0]
        console.print(f"  Found [cyan]{self.total_matching:,}[/] halal articles (from 57M)")
        
        # Show cache filtering info
        if self.cached_hashes:
            console.print(f"  Cache has [yellow]{len(self.cached_hashes):,}[/] scored articles to skip")
            estimated_new = max(0, self.total_matching - len(self.cached_hashes))
            console.print(f"  Estimated new articles: [green]~{estimated_new:,}[/]")
        else:
            console.print(f"  Cache filter: [dim]disabled[/]")

        # Prepare the main query
        query = f"SELECT date, text, extra_fields FROM '{parquet_pattern}' {where_clause}"
        console.print("[bold green]✓ DuckDB query ready![/]")
        console.print()
        
        # Execute query (cursor ready for iteration)
        self.result = self.con.execute(query)
        self._parquet_pattern = parquet_pattern
        self._where_clause = where_clause
    
    def __iter__(self) -> Iterator[NewsArticle]:
        """Iterate through articles."""
        return self._stream_articles()
    
    def _stream_articles(self) -> Iterator[NewsArticle]:
        """Stream articles from the prepared DuckDB result."""
        while True:
            chunk = self.result.fetchmany(10000)
            if not chunk:
                break

            for row in chunk:
                date_str, text, extra_fields_str = row

                extra_fields = _parse_extra_fields(extra_fields_str)
                symbols = _extract_symbols(extra_fields)
                if not symbols:
                    continue

                if self.halal_symbols:
                    symbols = [s for s in symbols if s in self.halal_symbols]
                    if not symbols:
                        continue

                date = _extract_date({"date": date_str}, extra_fields)
                if not date:
                    continue

                if self.cached_hashes:
                    import hashlib
                    article_hash = hashlib.md5(text.encode()).hexdigest()[:16]
                    if article_hash in self.cached_hashes:
                        continue

                dataset_source = extra_fields.get("dataset", "unknown")

                yield NewsArticle(
                    date=date,
                    text=text,
                    symbols=symbols,
                    dataset_source=dataset_source,
                    raw_date=date_str or "",
                )

        if self.con:
            self.con.close()
    
    def close(self):
        """Close the DuckDB connection."""
        if self.con:
            self.con.close()
            self.con = None


def stream_articles(
    config: ETLConfig,
    halal_symbols: set[str] | None = None,
    cached_hashes: set[str] | None = None,
) -> Iterator[NewsArticle]:
    """Stream articles from parquet files using DuckDB for fast pre-filtering.

    Uses SQL to filter articles at the parquet level before Python processing,
    resulting in 10-100x speedup compared to streaming all 57M articles.

    Args:
        config: ETL configuration
        halal_symbols: Set of halal symbols to filter to (optional, filters in SQL)
        cached_hashes: Set of already-cached article hashes to skip (optional)

    Yields:
        NewsArticle objects with symbols
    """
    streamer = DuckDBArticleStream(config, halal_symbols, cached_hashes)
    return iter(streamer)


def batch_articles(
    articles: Iterator[NewsArticle], batch_size: int
) -> Iterator[list[NewsArticle]]:
    """Batch articles for efficient processing.

    Args:
        articles: Iterator of NewsArticle
        batch_size: Number of articles per batch

    Yields:
        Lists of NewsArticle
    """
    batch: list[NewsArticle] = []
    for article in articles:
        batch.append(article)
        if len(batch) >= batch_size:
            yield batch
            batch = []

    # Yield remaining
    if batch:
        yield batch

